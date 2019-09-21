import os
import openslide
import cv2
import numpy as np

# wsi_path = r'H:\TCTDATA\SZSQ_originaldata\Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us\svs'
# path = r'X:\GXB\SNS\test_result\unet_4down_wsi_reco'
#
# wsi_names = os.listdir(path)
#
# ratio = 0.293 / 0.1803
# block_size = 256
#
# for wsi_name in wsi_names:
#
#     img_names = os.listdir(path + '/' + wsi_name)
#     img_names = [x for x in img_names if x.find('.tif') != -1]
#
#     ors = openslide.open_slide(wsi_path + '/' + wsi_name)
#
#     for name in img_names:
#         name = name[ : name.find('.tif')]
#         uints = name.split('_')
#
#         x = int(uints[2])
#         y = int(uints[3])
#
#         ex_size = int(256 * ratio)
#         ori_img = ors.read_region((x - ex_size , y - ex_size) , 0 , (ex_size * 3 , ex_size * 3))
#         ori_img = np.array(ori_img)
#         ori_img = ori_img[: , : , 0 : 3]
#         ori_img = ori_img[: , : , ::-1]
#         ori_img = cv2.resize(ori_img , (block_size * 3 , block_size * 3) , interpolation = cv2.INTER_CUBIC)
#         img = cv2.imread(path + '/' + wsi_name + '/' + name + '.tif')
#         img = img[: , 256 * 2 : 256 * 3 , 0]
#
#         c_img = ori_img.copy()
#         contours , _ = cv2.findContours(img , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             contour = np.array(contour) + block_size
#             c_img = cv2.drawContours(c_img , [contour] , -1 , (255 , 0 , 0) , 2)
#
#         com_img = cv2.hconcat((ori_img , c_img))
#         cv2.imwrite(path + '/' + wsi_name + '/' + name + '_l.tif' , com_img)
#     ors.close()


# points = np.load('X:\GXB\SNS\data1\Shengfuyou_2th\Positive\Contours\p_100_0067118 1009188_12738_129002_0.npy')
# print(points)

path = r'X:/GXB/SNS/data1/temp'

names = os.listdir('%s/Contours' % (path))
names = [x for x in names if x.find('.npy') != -1]

for name in names:
    name = name[ : name.find('.')]
    if not os.path.exists('%s/Img/%s.tif' % (path , name)):
        print(name , 'img not exit ..')
    if not os.path.exists('%s/Contours/%s.npy' % (path , name)):
        print(name , 'npy not exit ...')
    if not os.path.exists('%s/K_Contours/%s.txt' % (path , name)):
        print(name, 'txt not exit ...')