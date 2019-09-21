import os
import cv2
from multiprocessing import dummy
from multiprocessing import cpu_count
import numpy as np
from utils_pac.msort import MSort

block_size = 256


# 进行读入图片的处理
def post_handle(img_path):
    img = cv2.imread(img_path)

    # 取 单 通道即可
    img3 = img[: , 2 * block_size : 3 * block_size , 0]
    img4 = img[: , 3 * block_size : 4 * block_size , 0]

    contours , _ = cv2.findContours(img3 , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    mean_area = np.mean(areas)

    # 得到平均分数
    img3 = np.float32(img3) / 255.
    img4 = np.float32(img4) / 255. * img3

    mean_grade = np.sum(img4) / np.sum(img3)
    print(mean_area , mean_grade)
    return mean_area , mean_grade

def recommand(save_path , wsi_path ,  names , mean_value):

    new_names, new_mean_value = [], []

    min_area = np.min(mean_value[:, 0])
    max_area = np.max(mean_value[:, 0])

    for name, mv in zip(names, mean_value):

        if mv[0] > 1000 and mv[1] > 0.98:
            mv[0] = (mv[0] - min_area) / (max_area - min_area)
            new_names.append(name)
            # print(mv[0] , mv[1] , 0.5 * mv[0] + 0.5 * mv[1])
            new_mean_value.append([mv[0], mv[1], 0.05 * mv[0] + 0.95 * mv[1]])
    new_mean_value = np.array(new_mean_value)

    # reco_grades , reco_names = [] , []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # areas , names = MSort().heap_sort_s(new_mean_value[: , 0].copy() , new_names.copy())
    # for area , name in zip(areas[len(areas) - 16 : ] , names[len(areas) - 16 : ]):
    #     print(area , name)
    #
    grades , names = MSort().heap_sort_s(new_mean_value[: , 1].copy() , new_names.copy())
    for grade , name in zip(grades[len(names) - 16 : ] , names[len(names) - 16 : ]):
        print(grade , name)
        img = cv2.imread(wsi_path + '/' + name)
        cv2.imwrite(save_path + '/%f_%s' % (grade, name), img)

    # con_grades, names = MSort().heap_sort_s(new_mean_value[:, 2].copy(), new_names.copy())
    # for grade, name in zip(con_grades[len(names) - 16:], names[len(names) - 16:]):
    #     reco_grades.append(grade)
    #     reco_names.append(name)
    #     img = cv2.imread(wsi_path + '/' + name)
    #     cv2.imwrite(save_path + '/%f_%s' % (grade , name) , img)



reco_path = r'X:\GXB\SNS\test_result\unet_4down_wsi_reco'

path = r'X:\GXB\SNS\test_result\unet_4down_wsi'
wsi_names = os.listdir(path)

for wsi_name in wsi_names[1 : ]:
    print(wsi_name)
    img_names = os.listdir(path + '/' + wsi_name)
    img_names = [x for x in img_names if x.find('.tif') != -1]

    mean_value = []
    names = []
    pool = dummy.Pool(cpu_count())

    for img_name in img_names:
        t = pool.apply_async(post_handle , args = (path + '/%s/%s' % (wsi_name , img_name) , ))
        mean_value.append(t.get())
        names.append(img_name)
    pool.close()
    pool.join()

    mean_value = np.array(mean_value)
    print(np.shape(mean_value))

    recommand(reco_path + '/' + wsi_name , path + '/' + wsi_name , names, mean_value)