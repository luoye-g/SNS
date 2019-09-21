import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from nucleusSeg.test import kernel_seg_model


ks_model = kernel_seg_model()

t_ratio = 0.293
t_bs = 256
redun_ratio = 1 / 4.

p_path = 'Z:/GXB/SNS/data1'
sub_paths = ['SZSQ_originaldata/Shengfuyou_3th/positive/Shengfuyou_3th_positive_40X']
sub_ratis = [0.1803]

# nums_per_ep = 1000

def split_precit(img, ks_model , kers_p , name):

    size = img.shape[ : 2]
    r_size = int(t_bs - redun_ratio * t_bs)
    w_nums = int((size[1] - t_bs) / r_size) + 1
    h_nums = int((size[0] - t_bs) / r_size) + 1

    imgs = []
    for i in range(w_nums):
        for j in range(h_nums):
            imgs.append(img[j * r_size : j * r_size + t_bs , i * r_size : i * r_size + t_bs , :])
    imgs = (np.float32(imgs) / 255. - 0.5) * 2

    results = ks_model.predict_ori(imgs)
    result_img = np.zeros(size , dtype = np.float32)
    for i in range(w_nums):
        for j in range(h_nums):
            res = results[i * h_nums + j , : , : , 0]
            result_img[j * r_size: j * r_size + t_bs, i * r_size: i * r_size + t_bs] = res
    result_img = np.expand_dims(result_img , axis = 0)
    result_img = np.expand_dims(result_img , axis = 3)
    results, contours, areas = ks_model._postProcess(result_img)
    return results[0] , contours[0]

def list_save(path , list_data):

    with open(path , 'w') as file:
        for d in list_data:
            if len(np.shape(d)) > 2:
                d = d[: , 0  ,:]
            for item in d:
                file.write('%d_%d-' % (item[0] , item[1]))
            file.write('\n')

def list_read(path):
    with open(path , 'r') as file:
        d_list = []
        for line in file:
            line = line[ : -1]
            line = line.split('-')
            d = []
            for item in line:
                uints = item.split('_')
                if len(uints) > 1:
                    d.append([int(uints[0]) , int(uints[1])])
            d_list.append(d)
        return d_list

for s_r , s_p in zip(sub_ratis , sub_paths):

    imgs_p = '%s/%s/Img' % (p_path , s_p)
    kers_p = '%s/%s/K_Img' % (p_path , s_p)
    cons_p = '%s/%s/K_Contours' % (p_path , s_p)
    t_p = '%s/%s/t' % (p_path , s_p)

    if not os.path.exists(kers_p):
        os.makedirs(kers_p)
    if not os.path.exists(cons_p):
        os.makedirs(cons_p)

    names = os.listdir(imgs_p)
    names = [x for x in names if x.find('.tif') != -1]

    for i , name in enumerate(names):
        name = name[ : name.find('.tif')]
        img = cv2.imread('%s/%s.tif' % (imgs_p , name))

        img_re = cv2.resize(img , None , fx = s_r / t_ratio , fy = s_r / t_ratio , interpolation = cv2.INTER_CUBIC)
        size = img_re.shape[ : 2]
        w_r = (size[1] - t_bs) % int((1 - redun_ratio) * t_bs)
        w_h = (size[0] - t_bs) % int((1 - redun_ratio) * t_bs)
        if w_r != 0:
            w_r = int((1 - redun_ratio) * t_bs) - w_r
        if w_h != 0:
            w_h = int((1 - redun_ratio) * t_bs) - w_h

        img_re = cv2.copyMakeBorder(img_re, 0 , w_h , 0 , w_r , cv2.BORDER_CONSTANT, value=[0, 0, 0])
        result , contours = split_precit(img_re , ks_model , kers_p , name)
        for k in range(len(contours)):
            contours[k] = np.ceil(np.array(contours[k]) * t_ratio / s_r).astype(np.int)
        list_save('%s/%s.txt' % (cons_p , name) , contours)

        img = cv2.drawContours(img , contours , -1 , (0 , 0 , 255) , 1)
        cv2.imwrite('%s/%s.tif' % (kers_p , name) , img)