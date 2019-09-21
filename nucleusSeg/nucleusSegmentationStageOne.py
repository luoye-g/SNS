# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:47:19 2019

@author: lixu
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.utils import multi_gpu_model

def unet(input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    return model



class nucleusSegmentationStageOne:   
    def __init__(self, weight_path, input_size=(256, 256, 3), gpu_num=1, batch_size=16, removeSmallRegion=100, removeConvex=1.1):
        self.weight_path = weight_path
        self.input_size=input_size
        self.gpu_num = gpu_num
        self.batch_size = batch_size
        self.removeSmallRegion = removeSmallRegion
        self.removeConvex = removeConvex

        self._initModel()
    def _initModel(self):
        if self.gpu_num>1:
            with tf.device('/cpu:0'):
                model = unet(input_size=self.input_size)
                if self.weight_path is not None:
                    print('Load weights %s' % self.weight_path)
                    model.load_weights(self.weight_path)
            self.parallel_model = multi_gpu_model(model, gpus=self.gpu_num)
        else:
            self.parallel_model = unet(input_size=self.input_size)
            if self.weight_path is not None:
                print('Load weights %s' % self.weight_path)
                self.parallel_model.compile(optimizer=Adam(lr=0.001), loss=['categorical_crossentropy'],
                                            metrics=['categorical_accuracy'])
                self.parallel_model.load_weights(self.weight_path)

    def _postProcess(self, result):
        allResults = []
        allContours = []
        allAreas = []
        
        for i in range(len(result)):
            saveResult = result[i, :,:,0]
            saveResult[saveResult>0.5]=int(255)
            saveResult[saveResult<0.5]=int(0)
    
            finalContours = []
            finalArea = []
            finalResult = np.zeros(saveResult.shape , dtype = np.uint8)
    
            contours, _ = cv2.findContours(saveResult.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):   
                #面积
                area = cv2.contourArea(contours[j])
                #凸包面积
                convexHull = cv2.convexHull(contours[j])
                convexArea = cv2.contourArea(convexHull)
                #去除小的连通域
                if area>self.removeSmallRegion and convexArea/area<self.removeConvex:
                    finalContours.append(contours[j])
                    finalArea.append(area)
            finalResult = cv2.fillPoly(finalResult, finalContours, 255)
            allResults.append(finalResult)
            allContours.append(finalContours)
            allAreas.append(finalArea)
        return np.array(allResults), allContours, allAreas   

    def predict(self, image):
        # image = (image-0.5)*2
        result = self.parallel_model.predict(image, batch_size=self.batch_size*self.gpu_num, verbose=0)
        results, contours, areas = self._postProcess(result)
        return results, contours, areas

    def predict_ori(self , imgs):
        resuls = self.parallel_model.predict(imgs , batch_size = self.batch_size * self.gpu_num , verbose = 0)
        return resuls
        
