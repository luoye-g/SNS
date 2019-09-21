# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:13:14 2019

@author: lixu
"""
import cv2
from skimage import io
import numpy as np
import multiprocessing.dummy as multiprocessing


def GetImg(dirList, ith):
    path = dirList[ith]
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
    except:
        print(path)
    img = ((np.float32(img) / 255.) - 0.5)*2.
    return img

def GetImgMultiprocess(dirList, threadNum=20):
    pool = multiprocessing.Pool(threadNum)
    imgTotals = []
    for ith in range(len(dirList)):
        imgTotal = pool.apply_async(GetImg, args=(dirList,ith))
        imgTotal = imgTotal.get()
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotals = np.array(imgTotals)
    return imgTotals