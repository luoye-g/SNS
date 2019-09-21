# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:05:40 2019

@author: lixu
"""
# import os
# import numpy as np
from nucleusSeg.nucleusSegmentationStageOne import nucleusSegmentationStageOne
# from nucleusSeg.utils import GetImgMultiprocess

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
get kernel seg model
'''
def kernel_seg_model():
    input_size=(256,256,3)  #0.293分辨率
    gpu_num = 1
    batch_size = 16
    weight_path = '../nucleusSeg/block380.h5'
    segment = nucleusSegmentationStageOne(weight_path, input_size, gpu_num, batch_size)
    return segment


# \\192.168.0.105\data_sda2

# testImageList = os.listdir(r'X:\GXB\SNS\nucleusSegmentationStageOne\test\img')
# testImageList = [x for x in testImageList if x.find('.tif') != -1]
# testImageList = [os.path.join(r'X:\GXB\SNS\nucleusSegmentationStageOne\test\img', test) for test in testImageList]
# testImg = GetImgMultiprocess(testImageList)
#
# #推理与后处理
# segment = nucleusSegmentationStageOne(weight_path, input_size, gpu_num, batch_size)
# # #testImg的shape为(imgNum, H, W, C)
# results, contours, areas = segment.predict(testImg)
#
# pass

#results的shape为(imgNum, H, W)
#contours的shape为(imgNum, nucleusNum, pointNum, 1, 2)
#areas的shape为(imgNum, nucleusNum)

