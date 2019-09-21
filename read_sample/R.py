import os
import cv2
import numpy as np
import random
from nucleusSeg.nucleusSegmentationStageOne import nucleusSegmentationStageOne

'''
init kernel seg model
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_size=(256,256,3)  #0.293分辨率
gpu_num = 1
batch_size = 16
weight_path = '../nucleusSeg/block380.h5'
segment = nucleusSegmentationStageOne(weight_path, input_size, gpu_num, batch_size)
'''
fixed params setting
'''
bs = 256 # block size at 0.293
bs_ratio = 0.293
sample_range = 50
random_nums = 2

path = 'X:/GXB/SNS/data1/'
imgs_path = ['Shengfuyou_1th/Positive/' , 'Shengfuyou_2th/Positive/']
ratios = [0.243 , 0.243]

for i in range(1 , len(imgs_path)):

    # the path of store sample
    if not os.path.exists('%s%st_sample' % (path , imgs_path[i])):
        os.makedirs('%s%st_sample' % (path , imgs_path[i]))
    # get anno img names
    anno_names = os.listdir('%s%sContours' % (path , imgs_path[i]))
    anno_names = [x for x in anno_names if x.find('.npy') != -1 and x.find('p_') != -1]

    block_size = int(np.ceil(bs * ratios[i] / bs_ratio))
    block_radius = int(block_size // 2)

    for anno_name in anno_names:

        name = anno_name[anno_name.find('p_') + 2 : anno_name.find('.npy')]
        # get the properties of img block
        uints = name.split('_')
        wsi_name = ''
        lu = len(uints)
        for l in range(lu - 3):
            wsi_name += uints[l]
        x , y = int(uints[lu - 3]) , int(uints[lu - 2])
        level = int(uints[lu - 1])

        img = cv2.imread('%s%sImg/%s.tif' % (path , imgs_path[i] , name))
        points = np.load('%s%sContours/p_%s.npy' % (path , imgs_path[i] , name))

        crop_imgs = []
        crop_imgs_names = []
        crop_imgs_points = []
        for point in points:
            c_x , c_y = point[0] , point[1]

            xs = random.sample(range(c_x - sample_range, c_x + sample_range), random_nums)
            ys = random.sample(range(c_y - sample_range, c_y + sample_range), random_nums)
            xs.append(c_x)
            ys.append(c_y)
            xs = np.array(xs) - block_radius
            ys = np.array(ys) - block_radius

            for tx , ty in zip(xs , ys):
                crop_points = points.copy()
                crop_points[: , 0] -= tx
                crop_points[: , 1] -= ty
                crop_points = (crop_points * bs_ratio / ratios[i]).astype(np.int)

                crop_img = img[ty : ty + block_size , tx : tx + block_size , :]
                crop_img = cv2.resize(crop_img , (bs , bs) , interpolation = cv2.INTER_CUBIC)
                crop_img_name = '%s_%d_%d_%d' % (wsi_name , tx + x , ty + y , level)

                crop_imgs.append(crop_img)
                crop_imgs_names.append(crop_img_name)
                crop_imgs_points.append(crop_points)

        crop_imgs = (np.float32(crop_imgs) / 255. - 0.5) * 2
        try:
            results , r_contours , r_areas = segment.predict(crop_imgs)
        except:
            print(name , 'error')
            continue
        crop_imgs = np.uint8((crop_imgs / 2 + 0.5) * 255)

        for j in range(len(crop_imgs)):
            result = cv2.cvtColor(results[i] , cv2.COLOR_GRAY2BGR)
            result[: , : , :] = 0
            contours = r_contours[j]
            contour_img = crop_imgs[j].copy()
            s_flag = False
            for contour in contours:
                bx , by , bw , bh = cv2.boundingRect(contour)
                l , r , t , d = bx , bx + bw , by , by + bh
                p_flag = False
                for p in crop_imgs_points[j]:
                    if p[0] < r and p[0] > l and p[1] < d and p[1] > t:
                        p_flag = True
                        s_flag = True
                        break
                if p_flag:
                    result[: , : , 0] = cv2.fillPoly(result[: , : , 0].copy() , [contour] ,  255)
                    contour_img = cv2.drawContours(contour_img , [contour] , -1 , (0 , 0 , 255) , 1)
                else:
                    result[: , : , 1] = cv2.fillPoly(result[: , : , 1].copy() , [contour], 255)
                    contour_img = cv2.drawContours(contour_img , [contour], -1, (0, 255, 0), 1)

            if s_flag:
                temp = cv2.hconcat([crop_imgs[j] , contour_img , result])
                cv2.imwrite('%s%st_sample/%s.tif' % (path , imgs_path[i] , crop_imgs_names[j]) , temp)
