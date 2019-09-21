import os
import openslide
from utils_pac.mask_decoder import csv_decoder
import numpy as np
import cv2
import random
# from test.wsi_model_2 import generate_model , post_operate
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

wsi_path = r'H:\TCTDATA\SZSQ_originaldata\Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us\svs'
csv_path = r'H:\TCTDATA\SZSQ_originaldata\Labelfiles\xml_Tongji_5th\csv'
save_path = r'X:\GXB\SNS\test_result\unet_4dwon_hard\mask'
hard_css = ['tj190619608' , 'tj19061317' , 'tj190610618' , 'tj190604601' , 'tj190423329']

sample_range = 100

# 图片的存储
def sample_save(ors , points , nums , block_size , level , save_path , name):
    x, y, w, h = cv2.boundingRect(np.array(points))

    # if w < 10 and h < 10:  # 不应该会这么小
    #     return

    # try:
        #    在标注范围内进行随机撒点 ， 范围太大，重新确定范围

        # xs = random.sample(range(-sample_range, sample_range), nums)
        # ys = random.sample(range(-sample_range, sample_range), nums)
    xs , ys = [] , []

    xs.append(0 + x - 256)
    ys.append(0 + y - 256)


    # xs = np.array(xs) + x + int(w // 2)
    # ys = np.array(ys) + y + int(h // 2)

    #     从原图中读取区域
    for x, y in zip(xs, ys):
        # x -= int(block_size // 2)
        # y -= int(block_size // 2)
        # img = ors.read_region((x, y), level, (block_size, block_size))
        img = ors.read_region((x , y) , level , (w + 512, h + 512))
        img = np.array(img)
        img = img[:, :, 0: 3]
        img = img[:, :, ::-1]

        # img = cv2.resize(img , (256 , 256) , interpolation = cv2.INTER_CUBIC)

        contours = [[[256 , 256] , [256 + w , 256] , [256 + w , 256 + h] , [256 , 256 + h]]]
        print(contours)
        img = cv2.drawContours(img , np.array(contours) , -1 , (0 , 0 , 255) , 2)

        cv2.imwrite(
            save_path + '/' + name + '_' + str(x) + '_' + str(y) + '_' + str(
                level) + '.tif', img)
    # except:
    #     print(save_path + '/' + name, points)


def sample_make():
    for hc in hard_css:

        wsi = wsi_path + '/' + hc + '.sdpc.svs'

        mask_dict = csv_decoder(csv_path , hc + '.sdpc')
        print(np.shape(mask_dict['Positive']))
        wsi_ors = openslide.OpenSlide(wsi)
        for j in range(0, len(mask_dict['Positive'])):
                sample_save(wsi_ors, mask_dict['Positive'][j], 4 , 416 , 0 , save_path, hc)

        wsi_ors.close()

def test_img_save(imgs , masks , name , cpath):

    for img , mask , p in zip(imgs , masks):
        img = np.uint8(img * 255.)
        p_img = img.copy()
        p_mask = np.uint8(mask * 255)
        wj = mask.max(axis = 2)
        s_mask = mask[: , : , 0]
        mask[: , : , 0] = mask[: , : , 0] == wj
        mask[: , : , 1] = mask[: , : , 1] == wj
        mask[: , : , 2] = mask[: , : , 2] == wj

        s_mask = mask[: , : , 0] * s_mask
        mask[: , : , 0] , contours0 = post_operate(np.uint8(mask[:, :, 0]))
        if len(contours0) == 0: # 没有分割出病变核区域则不进行保存
            continue
        mask[: , : , 1] , contours1 = post_operate(np.uint8(mask[: , : , 1]))
        mask = np.uint8(mask * 255)

        p_img = cv2.drawContours(p_img , contours0 , -1 , (255, 0, 0) , 1)
        p_img = cv2.drawContours(p_img , contours1 , -1 , (0 , 255 , 0) , 1)

        s_mask = np.stack([s_mask , s_mask , s_mask] , axis = 2)
        s_mask = np.uint8(s_mask * 255)
        combined_img = cv2.hconcat([img , p_img , mask , s_mask , p_mask])

        temp1 = s_mask[: , : , 0] / 255
        temp2 = p_mask[: , : , 0] / 255 * temp1

        try:
            mean_grade = np.sum(temp2) / np.sum(temp1)
        except:
            mean_grade = 0

        cv2.imwrite(cpath + '/%f_%s' % (mean_grade , name) , combined_img)

def grade_predict():

    weight_path = r'X:\GXB\SNS\train_log\train_kernel_2\weights\kernel_seg_6_6800.h5'
    unet_model = generate_model(weight_path = weight_path)

    img_names = os.listdir(save_path)
    img_names = [x for x in img_names if x.find('.tif') != -1]

    imgs = []
    for name in img_names:
        imgs.append(cv2.imread(save_path + '/' + name))

    imgs = np.float32(imgs) / 255.
    p_masks = unet_model.predict(imgs , batch_size = 10 , verbose = 1)

    r_save_path = r'X:\GXB\SNS\test_result\unet_4dwon_hard\hard_imgs_predict'
    test_img_save(imgs, p_masks, img_names, r_save_path)

if __name__ == '__main__':
    pass
