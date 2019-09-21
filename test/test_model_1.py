'''
    病变核分割测试
'''
from models.unet_u import unet_4down
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import multi_gpu_model
from read_sample.SampleRead import SampleRead
import numpy as np
import os
import cv2
from skimage import morphology
import scipy.ndimage as ndi

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
    :parameter set
'''

gpu_num = 1
input_shape = (256 , 256 , 3)
lr = 1e-4
loss_function  = 'categorical_crossentropy'
acc_function = 'categorical_accuracy'
classes = 3

def generate_model(weight_path = None):

    if gpu_num == 1:
        model = unet_4down(input_shape = (256 , 256 , 3) , output_channel = classes)
        model.compile(optimizer=Adam(lr=lr), loss=[loss_function], metrics=[acc_function])
    else:
        with tf.device('/cpu:0'):
            model = unet_4down(input_shape = (256 , 256 , 3) , output_channel = classes)
            model.compile(optimizer=Adam(lr=lr), loss=[loss_function], metrics=[acc_function])
        model = multi_gpu_model(model, gpus = gpu_num)

    if weight_path:
        model.load_weights(weight_path)
    return model


def post_operate(img):
    thre_vol = 150
    img = ndi.binary_fill_holes(img) # 填洞

    img = np.uint8(img * 255.)
    contours , _ = cv2.findContours(img , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    f_contours = []
    for i in range(len(contours)):
        contours[i] = cv2.convexHull(contours[i]) # 对轮廓进行凸操作处理
        if cv2.contourArea(contours[i]) > thre_vol:
            f_contours.append(contours[i])
    img = cv2.fillPoly(img , f_contours , 255)
    return img / 255.

def test_img_save(test_imgs , test_masks , predict_masks , names , test_save_path):

    s_k_ious , n_k_ious = [] , []

    for i in range(0 , len(test_imgs)):

        img = np.uint8(test_imgs[i] * 255.)
        p_mask = predict_masks[i]
        p_img = img.copy()

        wj = p_mask.max(axis = 2)
        p_mask[: , : , 0] = p_mask[: , : , 0] == wj
        p_mask[: , : , 1] = p_mask[: , : , 1] == wj
        p_mask[: , : , 2] = p_mask[: , : , 2] == wj

        # 对预测的结果进行处理
        p_mask[: , : , 0] = post_operate(np.uint8(p_mask[: , : , 0]))
        p_mask[: , : , 1] = post_operate(np.uint8(p_mask[: , : , 1]))

        # 进行核分割iou的统计
        mask = test_masks[i] > 0
        p_mask = p_mask > 0

        if np.sum(mask[: , : , 0]) == 0 and np.sum(p_mask[: , : , 0]) == 0:
            s_k_iou = 1
        elif np.sum(mask[: , : , 0]) == 0 or np.sum(p_mask[: , : , 0]) == 0:
            s_k_iou = 0
        else:
            s_k_iou = np.sum(mask[: , : ,0] * p_mask[: , : , 0]) / np.sum(mask[: , : , 0] + p_mask[: , : , 0])

        if np.sum(mask[: , : , 1]) == 0 and np.sum(p_mask[: , : , 1]) == 0:
            n_k_iou = 1
        elif np.sum(mask[: , : , 1]) == 0 or np.sum(p_mask[: , : , 1]) == 0:
            n_k_iou = 0
        else:
            n_k_iou = np.sum(mask[: , : , 1] * p_mask[: , : , 1]) / np.sum(mask[: , : , 1] + p_mask[: , : , 1])

        s_k_ious.append(s_k_iou)
        n_k_ious.append(n_k_iou)

        mask = np.uint8(test_masks[i] * 255.)
        contours , _ = cv2.findContours(mask[: , : , 0] , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img , contours , -1 , (255 , 0 , 0) , 1)
        contours, _ = cv2.findContours(mask[:, :, 1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (0 , 255 , 0), 1)

        # print('multi channel : %d %d %d' % (np.sum(p_mask[: , : , 0]) , np.sum(p_mask[: , : , 1]) , np.sum(p_mask[: , : , 2])))
        p_mask = np.uint8(p_mask * 255)
        contours , _ = cv2.findContours(p_mask[: , : , 0] , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        p_img = cv2.drawContours(p_img , contours , -1 , (255 , 0 , 0) , 1)
        contours, _ = cv2.findContours(p_mask[:, :, 1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_img = cv2.drawContours(p_img, contours, -1, (0 , 255 , 0), 1)

        combined_img = cv2.hconcat([img , mask , p_img , p_mask])
        cv2.imwrite(test_save_path + names[i] , combined_img)
    return np.mean(s_k_ious) , np.mean(n_k_ious)

if __name__ == '__main__':
    test_nums = 7829
    test_weights = ['kernel_seg_6_6800.h5' , 'kernel_seg_6_6850.h5' , 'kernel_seg_6_6900.h5' ,
                    'kernel_seg_6_6950.h5' , 'kernel_seg_6_7000.h5' , 'kernel_seg_6_7050.h5']
    save_path = r'X:\GXB\SNS\test_result\unet_4down'
    for weight in test_weights:

        weight_path = r'X:\GXB\SNS\train_log\train_kernel_2\weights' + '/' + weight
        unet_model = generate_model(weight_path = weight_path)

        train_file = r'X:\GXB\SNS\data\train.txt'
        test_file = r'X:\GXB\SNS\data\test.txt'

        sr = SampleRead(train_file , test_file)

        test_imgs , test_masks , names = sr.generate_test_data(test_nums = test_nums)

        predict_masks = unet_model.predict(test_imgs , verbose = 0)

        if not os.path.exists(save_path + '/' + weight):
            os.makedirs(save_path + '/' + weight)

        s_k_iou , n_k_iou = test_img_save(test_imgs , test_masks , predict_masks , names , save_path + '/' + weight + '/')
        print(weight , s_k_iou , n_k_iou)
