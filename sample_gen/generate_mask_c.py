import os
import cv2
from multiprocessing import dummy
from multiprocessing import cpu_count

path = 'X:/GXB/SNS/data'

mask_path = r'X:\GXB\SNS\data\mask'
mask_c_path = r'X:\GXB\SNS\data\mask_c'

cs = ['SZSQ_originaldata/Tongji_3th/positive/tongji_3th_positive_40x/svs',
      'SZSQ_originaldata/Tongji_4th/positive/svs' ,
      'Shengfuyou_1th/Positive' , 'Shengfuyou_2th/Positive']


def mask_c_save(img_path , mask_path , save_path , name):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    mask1 = mask[: , : , 0]
    mask2 = mask[: , : , 1]

    contours1 , _ = cv2.findContours(mask1 , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    contours2 , _ = cv2.findContours(mask2 , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    img_c = img.copy()

    img_c = cv2.drawContours(img_c , contours1 , -1 , (255 , 0 , 0) , 1)
    img_c = cv2.drawContours(img_c , contours2 , -1 , (0 , 255 , 0) , 1)

    combined_img = cv2.hconcat((img , img_c))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(save_path + '/' + name , combined_img)

if __name__ == '__main__':

    for c in cs:

        img_names = os.listdir(path + '/' + c)
        img_names = [x for x in img_names if x.find('.tif') != -1]

        pool = dummy.Pool(cpu_count())

        for name in img_names:
            args = (path + '/' + c + '/' + name , mask_path + '/' + c + '/' + name ,
                    mask_c_path + '/' + c , name)
            pool.apply_async(mask_c_save , args = args)

        pool.close()
        pool.join()
