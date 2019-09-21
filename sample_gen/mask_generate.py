# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:05:40 2019

@author: lixu
"""
import os
import numpy as np
from nucleusSeg.nucleusSegmentationStageOne import nucleusSegmentationStageOne
from nucleusSeg.utils import GetImgMultiprocess
import cv2
from multiprocessing import cpu_count
from multiprocessing import dummy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_size = (256, 256, 3)  # 0.293分辨率
gpu_num = 1
batch_size = 16
weight_path = 'F:/GetSample/py_project/SNS/nucleusSeg/block380.h5'

path = 'X:/GXB/SNS/data'

cs_3d = ['Shengfuyou_1th/Positive', 'Shengfuyou_1th/Negative',
         'Shengfuyou_2th/Positive', 'Shengfuyou_2th/Negative']

cs_szsq_tj = ['SZSQ_originaldata/Tongji_3th/positive/tongji_3th_positive_40x/svs',
              'SZSQ_originaldata/Tongji_3th/negative/tongji_3th_negtive_40x/svs',
              'SZSQ_originaldata/Tongji_4th/positive/svs',
              'SZSQ_originaldata/Tongji_4th/negative/svs',
              'SZSQ_originaldata/Tongji_4th/positive/temp']

save_path = 'X:/GXB/SNS/data/mask1'


def save_mask(r, c, a, name, cs):
    mask = np.zeros((256, 256, 3), dtype=np.uint8)

    if len(a) > 0:

        b = np.array(a) > 1000
        temp1 = mask[:, :, 0]
        temp2 = mask[:, :, 1]
        if np.sum(b) > 0:

            for con, are in zip(c, b):
                if are:
                    temp1 = cv2.fillPoly(temp1.copy(), con, 255)
                else:
                    temp2 = cv2.fillPoly(temp2.copy(), con, 255)

        else:
            ind = a.index(max(a))
            for i in range(0, len(a)):
                if i == ind:
                    temp1 = cv2.fillPoly(temp1.copy(), c[i], 255)
                else:
                    temp2 = cv2.fillPoly(temp2.copy(), c[i], 255)

        mask[:, :, 0] = temp1
        mask[:, :, 1] = temp2

        mask[:, :, 2] = 255 - mask[:, :, 0] - mask[:, :, 1]

    else:
        mask[:, :, 2] = 255

    if not os.path.exists(save_path + '/' + cs):
        os.makedirs(save_path + '/' + cs)
    print(save_path + '/' + cs + '/' + name)
    cv2.imwrite(save_path + '/' + cs + '/' + name, mask)


# 推理与后处理
segment = nucleusSegmentationStageOne(weight_path, input_size, gpu_num, batch_size)

for cs in cs_szsq_tj[4: 5]:

    pool = dummy.Pool(cpu_count())

    img_path = path + '/' + cs
    testImageList = os.listdir(img_path)
    testImageList_names = [x for x in testImageList if x.find('.tif') != -1]
    testImageList = [os.path.join(img_path, test) for test in testImageList_names]
    testImg = GetImgMultiprocess(testImageList)

    # #testImg的shape为(imgNum, H, W, C)
    results, contours, areas = segment.predict(testImg)

    #  save result
    for r, c, a, name in zip(results, contours, areas, testImageList_names):
        # print(name)
        save_mask(r, c, a, name, cs)

    pool.close()
    pool.join()


