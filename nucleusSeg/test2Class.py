# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:05:40 2019

@author: lixu
"""
import os
import cv2
from nucleusSeg.utils import GetImgMultiprocess
from nucleusSeg import nucleusSegmentationStageOne

gpu_num = 1                                                                 
batch_size = 1  
input_size=(256,256,3)  #0.293分辨率                                               
weight_path = 'block380.h5' 
os.environ['CUDA_VISIBLE_DEVICES']='0'
segment = nucleusSegmentationStageOne(weight_path, input_size, gpu_num, batch_size)                                    

testImgDir = r'F:\2ClassSegmentationDataset\image\half_positive'
maskDir = r'F:\2ClassSegmentationDataset\mask\half_positive'
slideList = os.listdir(testImgDir)
for slide in slideList:
    maskPath = os.path.join(maskDir, slide)
    if not os.path.exists(maskPath):
        os.makedirs(maskPath)
    testImgPath = os.path.join(testImgDir, slide)
    testImgList = os.listdir(testImgPath)
    testImgList = [os.path.join(testImgPath, testImg) for testImg in testImgList]

    testImg = GetImgMultiprocess(testImgList)
    if testImg.shape==(0,):
        continue
    #testImg = cv2.resize(testImg, (256, 256))
    results, contours, areas = segment.predict(testImg)
    
    for ith, result in enumerate(results):
        cv2.imwrite(os.path.join(maskPath, testImgList[ith].split('\\')[-1]), result)
        

