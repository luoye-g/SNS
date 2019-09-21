# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:58:18 2018

@author: yujingya
"""
import numpy as np
import random
import cv2
from skimage import  morphology
import math
from skimage import exposure

class DataEnhanment:
    def __init__(self):
        pass

    def dataEnhancement(self , img , mask):
        choose = np.random.randint(0 , 21)
        if choose < 10:
            img = self.Sharp(img , 8.3 + random.random() * 0.4)
        elif choose < 20:
            img = self.Gauss(img , random.random() + 0.1)
        else:
            return img , mask
        img = self.gamma_trans(img , random.random() * 0.8 + 0.8)
        img = self.img_noise_new(img)
        img = self.HSV_trans(img)
        img = self.RGB_trans(img)
        img , mask = self.rotate(img , mask)
        if mask is not None:
            mask.reshape(mask.shape + (1,))
        return img , mask

    def random_data_enhancement(self , img , mask , seeds = 3):

        #每次随机进行 seeds 次数据增强
        for i in range(0 , seeds):
            seed = np.random.randint(0 , 7)
            img , mask = self.data_enhancement_by_seed(img , mask , seed)

        if len(np.shape(mask)) != 3 :
            mask = np.expand_dims(mask , axis = 2)
        return img , mask

    def data_enhancement_by_seed(self , img , mask , seed):
        if seed == 0:
            img = self.Sharp(img, 8.3 + random.random() * 0.4)
        elif seed == 1:
            img = self.Gauss(img, random.random() + 0.1)
        elif seed == 2:
            img = self.gamma_trans(img, random.random() * 0.8 + 0.8)
        elif seed == 3:
            img = self.img_noise_new(img)
        elif seed == 4:
            img = self.HSV_trans(img)
        elif seed == 5:
            img = self.RGB_trans(img)
        elif seed == 6:
            img , mask = self.rotate(img, mask)
        return img , mask

    """
    linear_trans:线性变换
    """

    def linear_trans(self , img):
        imgflat = img.flatten()
        imgflat = imgflat[imgflat > 0]
        # 线性变换
        threshold_low = 0

        if len(np.bincount(imgflat)) == 0:
            return img

        threshold_high = np.argmax(np.bincount(imgflat))  # bincount检查最大索引 , 得到出现次数最大的像素点值
        img = np.float32(img)  # 转化为float
        if threshold_high <= threshold_low:  # 高阈值不能等于0
            print("Error:高阈值取值太小")
            return
        img_max = (img > threshold_high) * 255  # 将高于高阈值的置为255
        img_min = (img < threshold_low) * 0  # 低于低阈值的置为0
        img_middle = (img >= threshold_low) * (img <= threshold_high) * 1  # 中间阈值置为1
        img_middle = ((255 / (threshold_high - threshold_low)) * (img - threshold_low)) * img_middle  # 得到中间位置的图像值
        img = np.uint8(img_max + img_min + img_middle)  # 转化为uint8位图像
        # # 线性计算
        k = np.random.randint(96, 104) / 100
        b = np.random.randint(-4, 5)
        img = np.add(np.multiply(k, img), b)  # 对图像进行线性空间的偏移
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img)  # 对图像进行规范化
        return img

    """
    rotate:图片随机旋转+flip操作
    """

    def rotate(self , img , mask):
        angle = np.random.randint(0, 4) * 90
        height = img.shape[0]
        width = img.shape[1]

        if mask is not None:
            heightMask = mask.shape[0]
            widthMask = mask.shape[1]

        if angle % 180 == 0:
            scale = 1
            scaleMask = 1
        elif angle % 90 == 0:
            scale = float(max(height, width)) / min(height, width)
            if mask is not None:
                scaleMask = float(max(heightMask, widthMask)) / min(heightMask, widthMask)
        else:
            scale = math.sqrt(pow(height, 2) + pow(width, 2)) / min(height, width)
            if mask != None:
                scaleMask = math.sqrt(pow(heightMask, 2) + pow(widthMask, 2)) / min(heightMask, widthMask)
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        if mask is not None:
            rotateMatMask = cv2.getRotationMatrix2D((widthMask / 2 , heightMask / 2) , angle , scaleMask)
        img = cv2.warpAffine(img, rotateMat, (width, height))
        if mask is not None:
            mask = cv2.warpAffine(mask , rotateMatMask , (widthMask , heightMask))
        # flip操作
        n = np.random.randint(0, 3)
        if n != 2: #判断是否进行水平或垂直翻转
            img = np.flip(img, n)
            if mask is not None:
                mask = np.flip(mask , n)
        if mask is not None:
            return img.astype(np.uint8) , mask.astype(np.uint8)
        else:
            return img.astype(np.uint8) , None

    """
    contrast:对比度变换 c 建议取值0.7-1.2
    random.random()*0.5 + 0.7
    """

    def contrast(self , img, c):
        # 亮度就是每个像素所有通道都加上b
        b = 0
        rows, cols, chunnel = img.shape
        blank = np.zeros([rows, cols, chunnel], img.dtype)
        # np.zeros(img1.shape, dtype=uint8)
        dst = cv2.addWeighted(img, c, blank, 1 - c, b) #b设置为0就是直接叠加

        dst = (dst > 255) * 255 + (dst <= 255) * dst
        return np.uint8(dst)

    """
    gamma_trans:gamma变换 gamma建议取值 0.6-1.6 ， 图像亮度与对比度的调整
    """

    def gamma_trans(self , img , gamma):
        # gamma = random.random() * 0.8 + 0.8
        img = np.float32(img) / 255.
        img = exposure.adjust_gamma(img, gamma) #I = (I)^gamma
        img = img * 255
        img = (img > 255) * 255 + (img <= 255) * img
        return np.uint8(img)

    """
    img_noise:图像加入噪声
    """
    global NOISE_NUMBER
    NOISE_NUMBER = 1000

    def img_noise(self , img):
        height, weight, channel = img.shape
        for i in range(NOISE_NUMBER):
            x = np.random.randint(0, height)
            y = np.random.randint(0, weight)
            img[x, y, :] = 255
        return img

    """
    图像中加入随机噪声
    """
    def img_noise_new(self , img):
        imgShape = img.shape
        for i in range(np.random.randint(500 , 1001)):
            x = np.random.randint(2 , imgShape[0] - 3)
            y = np.random.randint(2 , imgShape[1] - 3)
            n = np.random.randint(0 , 7)
            if n == 0:
                img[x , y , :] = 255
            if n == 1:
                img[x : x + 2 , y , :] = 255
            if n == 2:
                img[x , y : y + 2 , :] = 255
            if n == 3:
                img[x : x + 2 , y : y + 2 , :] = 255
            if n == 4:
                img[x - 1 : x + 2 , y , :] = 255
            if n == 5:
                img[x , y - 1 : y + 2 , :] = 255
        return img
    """
    PCA Jittering:先计算RGB通道的均值和方差，进行归一化，然后在整个训练集上计算协方差矩阵，
    进行特征分解，得到特征向量和特征值(这里其实就是PCA分解)，在分解后的特征空间上对特征值
    做随机的微小扰动，根据下式计算得到需要对R,G,B扰动的值，把它加回到RGB上去，作者认为这样
    的做法可以获得自然图像的一些重要属性，把top-1 error又降低了1%
    建议a取值范围为(0-0.004)
    random.random()*0.004)
    """

    def PCA_Jittering(self , img, a):
        img_size = img.size / 3
        img1 = img.reshape(int(img_size), 3) #将图像变为三维向量
        img1 = np.transpose(img1) #转置
        img_cov = np.cov([img1[0], img1[1], img1[2]])  # 计算矩阵特征向量，这个是计算协方差矩阵，无偏估计
        # print(img_cov)
        lamda, p = np.linalg.eig(img_cov) #计算协方差矩阵的特征值与特征向量
        p = np.transpose(p)  # 得到特征向量的转置矩阵
        alpha1 = random.normalvariate(0, a) #产生标准正态分布
        alpha2 = random.normalvariate(0, a)
        alpha3 = random.normalvariate(0, a)
        # print(alpha1 , alpha2 , alpha3)
        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))  # 加入扰动
        add_num = np.dot(p, v) #对特征向量进行点积
        # print(add_num)
        img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]]) #图像根据协方差的特征值与特征向量的点积进行扰动
        img2 = np.swapaxes(img2, 0, 2) #将维度进行调换
        img2 = np.swapaxes(img2, 0, 1)

        img2[img2 > 255] = 255
        img2[img2 < 0] = 0

        return np.uint8(img2)

    """
    Gauss_edge:对输入的图片img的细胞边缘进行高斯滤波处理
    """

    def Gauss_edge(self , img, ks = None):  ##高斯模糊

        path = 'X:\\GXB\\sec_seg_sample\\shy_1\\positive_p1_test\\LISL\\enhaImg\\'

        # 细胞边缘模糊整体图:
        img_edge = img.copy()
        sigmas = random.random() * 1.5 + 0.5  # 0.5-2 产生随机数
        img_edge_Gauss = cv2.GaussianBlur(img_edge, (int(6 * np.ceil(sigmas) + 1), int(6 * np.ceil(sigmas) + 1)),
                                          sigmas) #sigmas为标准差
        # 得到细胞前景
        threColor = 8
        threVol = 1024 // 4
        wj1 = img.max(axis=2)
        wj2 = img.min(axis=2)
        wj3 = wj1 - wj2
        imgBin = wj3 > threColor
        imgBin = morphology.remove_small_objects(imgBin, min_size = threVol)
        imgBin = np.uint8(imgBin)
        # cv2.imwrite(path + 'imgBin1.tif' , imgBin * 255)
        # 去除孔洞
        kernel = np.ones((5, 5), np.uint8)
        imgBin = cv2.dilate(imgBin, kernel, iterations=1) #对图像进行膨胀
        imgBin = cv2.erode(imgBin, kernel, iterations=1) #对图像进行腐蚀
        # cv2.imwrite(path + 'imgBin2.tif', imgBin * 255)
        # 得到边缘部分大前景:
        kernel = np.ones((15, 15), np.uint8)
        imgBin_big = cv2.dilate(imgBin, kernel, iterations=1) #膨胀
        kernel = np.ones((40, 40), np.uint8)
        imgBin_small = cv2.erode(imgBin, kernel, iterations=1) #腐蚀
        imgBin_edge_big = imgBin_big - imgBin_small
        # cv2.imwrite(path + 'imgBin_edge_big.tif', imgBin_edge_big * 255)
        # 得到边缘部分小前景:
        kernel = np.ones((5, 5), np.uint8)
        imgBin_big_temp = cv2.dilate(imgBin, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        imgBin_small_temp = cv2.erode(imgBin, kernel, iterations=1)
        imgBin_edge_small = imgBin_big_temp - imgBin_small_temp
        # cv2.imwrite(path + 'imgBin_edge_small.tif', imgBin_edge_small * 255)
        # 在小前景取某一小边缘:
        ind = np.flatnonzero(imgBin_edge_small.copy())
        if np.size(ind) == 0:
            return img

        ind = ind[np.random.randint(np.size(ind) , size = random.choice([3, 4, 5]))] #取不定个数
        indRow, indCol = np.unravel_index(ind, np.shape(imgBin_edge_small))  # 取1-2个点，转化为行列坐标
        imgmap = np.zeros(np.shape(imgBin_edge_small), np.uint8) #得到imgmap
        for k in range(len(indRow)):
            imgmap[int(indRow[k] - 40):int(indRow[k] + 40), int(indCol[k] - 40):int(indCol[k] + 40)] = 1 #随机选了区域置1
        # cv2.imwrite(path + 'imgmap.tif', imgmap * 255)


        imgBin_edge = np.multiply(imgBin_edge_big, imgmap)
        # cv2.imwrite(path + 'imgBin_edge.tif', imgBin_edge * 255)
        sigmas = 5
        imgBin_edge = cv2.GaussianBlur(imgBin_edge, (int(6 * np.ceil(sigmas) + 1), int(6 * np.ceil(sigmas) + 1)),
                                       sigmas) #对选中的区域进行高斯平滑
        img_edge_Gauss[imgBin_edge == 0] = 0
        img[imgBin_edge == 1] = 0  # 原地操作
        img_new = img + img_edge_Gauss #得到边缘平滑后的图像
        # 整体模糊让边缘不突兀
        sigmas = random.random() * 0.1 + 0.1  # 0.1-0.2 #设置随机θ，再次对新图像进行平滑
        img_new = cv2.GaussianBlur(img_new, (int(6 * np.ceil(sigmas) + 1), int(6 * np.ceil(sigmas) + 1)), sigmas)
        return img_new

    """
    Gauss:对输入的图片img整体进行高斯滤波处理
    """

    def Gauss(self , img , sigmas): #0.1-1.1
        # 整体模糊让边缘不突兀
        # sigmas = random.random() + 0.1  # 0.1-1.1
        img_new = cv2.GaussianBlur(img, (int(6 * np.ceil(sigmas) + 1), int(6 * np.ceil(sigmas) + 1)), sigmas)
        return img_new

    """
    Sharp:对输入的图片img进行锐化
    建议滤波器卷积核中间值：8.3-8.7
    """

    def Sharp(self , img , sigmas):
        # sigmas = random.random() * 0.4 + 8.3  # 8.3-8.7
        kernel = np.array([[-1, -1, -1], [-1, sigmas, -1], [-1, -1, -1]], np.float32) / (sigmas - 8)  # 锐化
        img_new = cv2.filter2D(img, -1, kernel=kernel)
        img_new = (img_new > 255) * 255 + (img_new <= 255) * img_new
        return np.uint8(img_new)

    """
    HSV_trans:对输入的图片img进行hsv空间的变换，h：色相，s：饱和度，v：亮度
    若做GAN可以适当缩小或者关闭h的变换 
    """

    def HSV_trans(self , img, h_change=1, s_change=1, v_change=1):  ##hsv变换
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #将图像转化到HSV空间中去
        hsv = np.float64(hsv)
        # hsv[...,0] = 180
        if h_change != 0:  # random.random()
            k = random.random() * 0.1 + 0.95  # 0.95-1.05
            b = random.random() * 6 - 3  # -3/3
            hsv[..., 0] = k * hsv[..., 0] + b
            hsv[..., 0][hsv[..., 0] <= 0] = 0
            hsv[..., 0][hsv[..., 0] >= 180] = 180
        if s_change != 0:
            k = random.random() * 0.8 + 0.7  # 0.7-1.5
            b = random.random() * 20 - 10  # -10/10
            hsv[..., 1] = k * hsv[..., 1] + b
            hsv[..., 1][hsv[..., 1] <= 0] = 1
            hsv[..., 1][hsv[..., 1] >= 255] = 255
        if v_change != 0:
            k = random.random() * 0.45 + 0.75  # 0.75-1.2
            b = random.random() * 18 - 10  # -10-8
            hsv[..., 2] = k * hsv[..., 2] + b
            hsv[..., 2][hsv[..., 2] <= 0] = 1
            hsv[..., 2][hsv[..., 2] >= 255] = 255
        hsv = np.uint8(hsv)
        img_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_new

    """
    RGB_trans:对输入的图片img进行RGB通道随机转换
    """

    def RGB_trans(self , img):
        index = [i for i in range(3)]
        random.shuffle(index)
        img = np.stack((img[:, :, index[0]], img[:, :, index[1]], img[:, :, index[2]]), axis=2)
        return img