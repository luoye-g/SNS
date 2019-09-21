import random
from multiprocessing import cpu_count
from multiprocessing import dummy
import cv2
import numpy as np
from utils_pac.data_enhancement import DataEnhanment

class SampleRead:

    def __init__(self , train_file , test_file , batch_size = 20):

        self.train_file = train_file
        self.test_file = test_file

        self.train_list = []
        self.test_list = []

        with open(self.train_file) as t:

            for line in t:
                line = line[ : -1]
                uints =line.split('\t')
                self.train_list.append(uints)


        with open(self.test_file) as t:

            for line in t:
                line = line[ : -1]
                uints = line.split('\t')
                self.test_list.append(uints)

        self.batch_size = batch_size
        self.index = 0
        self.data_flag = True

        self.de = DataEnhanment()


    def init_data_list(self , is_train = True):
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        self.index = 0
        self.data_flag = True

        if is_train:
            self.data_list = self.train_list
        else:
            self.data_list = self.test_list

    def generate_test_data(self , test_nums = 100):

        imgs , masks , names = [] , [] , []

        pool = dummy.Pool(int(cpu_count() // 2))

        random.shuffle(self.test_list)
        data_list = self.test_list[ : test_nums]

        for d in data_list:
            item = pool.apply_async(self.read_item , args = (d , )).get()
            imgs.append(item[0])
            masks.append(item[1])
            names.append(d[0][d[0].rfind('/') + 1 : ])

        pool.close()
        pool.join()

        return np.array(imgs) , np.array(masks) , names


    def read_item(self , uint , is_enhancement = False):

        img = cv2.imread(uint[0])
        mask = cv2.imread(uint[1])

        if is_enhancement: # True 时增加数据增强
            img , mask = self.de.random_data_enhancement(img , mask , seeds = 5)

        img = np.float32(img) / 255.
        mask = np.float32(mask) / 255.

        return img , mask

    def generate_data(self):

        imgs , masks = [] , []

        pool = dummy.Pool(int(cpu_count() // 2))
        k = 0

        data_list = self.data_list

        while k < self.batch_size and self.index < len(data_list):

            item = pool.apply_async(self.read_item , args = (data_list[self.index] , True)).get()
            imgs.append(item[0])
            masks.append(item[1])
            k += 1
            self.index += 1

        if self.index >= len(data_list):
            self.data_flag = False

        pool.close()
        pool.join()

        return np.array(imgs) , np.array(masks)


if __name__ == '__main__':

    train_file = r'X:\GXB\SNS\data\train.txt'
    test_file = r'X:\GXB\SNS\data\test.txt'

    sr = SampleRead(train_file , test_file)
    sr.init_data_list(is_train = True)

    # while sr.data_flag:
    #     imgs , masks = sr.generate_data()
        # print(sr.index , np.shape(imgs) , np.shape(masks))

    imgs , masks = sr.generate_data()

    t_s_path = r'X:\GXB\SNS\train_log\R'
    for i , mask in enumerate(masks):
        mask = np.uint8(mask * 255)
        m1 , m2 , m3 = mask[: , : , 0] , mask[: , : , 1] , mask[: , : , 2]
        cv2.imwrite(t_s_path + '/' + str(i) + '_1.tif' , m1)
        cv2.imwrite(t_s_path + '/' + str(i) + '_2.tif', m2)
        cv2.imwrite(t_s_path + '/' + str(i) + '_3.tif', m3)