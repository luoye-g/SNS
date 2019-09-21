'''
    WSI 病变核分割测试
'''
from models.unet_u import unet_4down
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import multi_gpu_model
import numpy as np
import os
import cv2
from skimage import morphology
import scipy.ndimage as ndi
from multiprocessing import dummy
from multiprocessing import cpu_count

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
    :parameter set
'''

gpu_num = 1
input_shape = (256 , 256 , 3)
lr = 1e-4
loss_function  = 'categorical_crossentropy'
acc_function = 'categorical_accuracy'
classes = 3

def generate_model(weight_path = None):

    if gpu_num == 1:
        model = unet_4down(input_shape = (256 , 256 , 3) , output_channel = classes)
        model.compile(optimizer=Adam(lr=lr), loss=[loss_function], metrics=[acc_function])
    else:
        with tf.device('/cpu:0'):
            model = unet_4down(input_shape = (256 , 256 , 3) , output_channel = classes)
            model.compile(optimizer=Adam(lr=lr), loss=[loss_function], metrics=[acc_function])
        model = multi_gpu_model(model, gpus = gpu_num)

    if weight_path:
        model.load_weights(weight_path)
    return model


def post_operate(img):
    thre_vol = 150
    img = ndi.binary_fill_holes(img) # 填洞
    img = morphology.remove_small_objects(img, min_size=100) # 移除较小连通域
    img = np.uint8(img * 255.)
    contours , _ = cv2.findContours(img , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    f_contours = []
    for i in range(len(contours)):
        contours[i] = cv2.convexHull(contours[i]) # 对轮廓进行凸操作处理
        if cv2.contourArea(contours[i]) > thre_vol:
            f_contours.append(contours[i])
    img = cv2.fillPoly(img , f_contours , 255)
    return img / 255. , f_contours

def test_img_save(imgs , masks , name , plist , cpath):

    for img , mask , p in zip(imgs , masks , plist):
        img = np.uint8(img * 255.)
        p_img = img.copy()
        p_mask = np.uint8(mask * 255)
        wj = mask.max(axis = 2)
        s_mask = mask[: , : , 0]
        mask[: , : , 0] = mask[: , : , 0] == wj
        mask[: , : , 1] = mask[: , : , 1] == wj
        mask[: , : , 2] = mask[: , : , 2] == wj

        s_mask = mask[: , : , 0] * s_mask
        mask[: , : , 0] , contours0 = post_operate(np.uint8(mask[:, :, 0]))
        if len(contours0) == 0: # 没有分割出病变核区域则不进行保存
            continue
        mask[: , : , 1] , contours1 = post_operate(np.uint8(mask[: , : , 1]))
        mask = np.uint8(mask * 255)

        p_img = cv2.drawContours(p_img , contours0 , -1 , (255, 0, 0) , 1)
        p_img = cv2.drawContours(p_img , contours1 , -1 , (0 , 255 , 0) , 1)

        s_mask = np.stack([s_mask , s_mask , s_mask] , axis = 2)
        s_mask = np.uint8(s_mask * 255)
        combined_img = cv2.hconcat([img , p_img , mask , s_mask , p_mask])
        cv2.imwrite(cpath + '/%s_%d_%d.tif' % (name , p[0] , p[1]) , combined_img)

class WSIReader():

    def __init__(self , wsi_path , block_size = 256 , ratio = 0.293 , redun_ratio = 1. / 4):

        '''
        :param wsi_path: 切片路径
        :param block_size: 在ratio (um/pixel)倍率下的取块大小
        :param ratio: 默认的取块倍率
        :param redun_ratio: 取块的冗余大小比例
        '''

        import openslide

        self.ors = openslide.OpenSlide(wsi_path)
        self.size = self.ors.level_dimensions[0]
        self.block_size = block_size
        self.redun_ratio = redun_ratio
        self.cblock_size = int(np.ceil(block_size * 0.293 / ratio))
        self.redun_size = int(redun_ratio * self.cblock_size)

        self.position_list = self.init_position_list()
        self.read_flag= True
        self.read_index = 0

    def init_position_list(self):
        '''
        :return:  返回取块的位置列表
        '''

        size = self.size
        cblock_size = self.cblock_size
        redun_size = self.redun_size
        w_nums = np.floor((size[0] - cblock_size) / (cblock_size - redun_size)) + 1
        h_nums = np.floor((size[1] - cblock_size) / (cblock_size - redun_size)) + 1

        print('nums all : %d' % (w_nums * h_nums))

        position_list = []

        for i in range(int(w_nums)):
            for j in range(int(h_nums)):
                x = i * (cblock_size - redun_size)
                y = j * (cblock_size - redun_size)
                position_list.append([x , y])

        return position_list

    def read_item(self , position , level = 0):
        # print(position)
        img = self.ors.read_region(position , level , (self.cblock_size , self.cblock_size))
        img = np.array(img)
        img = img[: , : , 0 : 3]
        img = img[: , : , ::-1]
        img = cv2.resize(img , (self.block_size , self.block_size) , interpolation = cv2.INTER_CUBIC)
        img = np.float32(img) / 255.
        return img

    def read_by_position_list(self , batch_size = 20):
        '''
        :param batch_size: 每次读取的小块数量
        :return: 返回读取的图像和其对应的读取位置
        '''
        imgs = []
        plist = []
        pool = dummy.Pool(cpu_count())
        i = 0
        while self.read_index < len(self.position_list) and i < batch_size:
            t = pool.apply_async(self.read_item , args = (self.position_list[self.read_index] , )).get()
            imgs.append(t)
            plist.append(self.position_list[self.read_index])
            i += 1
            self.read_index += 1
        pool.close()
        pool.join()

        if self.read_index >= len(self.position_list):
            self.read_flag = False

        return np.array(imgs) , np.array(plist)

    def close_wsi(self):
        self.ors.close()

if __name__ == '__main__':

    # 模型的加载
    weight_path = r'X:\GXB\SNS\train_log\train_kernel_2\weights\kernel_seg_6_6800.h5'
    unet_model = generate_model(weight_path=weight_path)

    test_wsi_path = r'H:\TCTDATA\SZSQ_originaldata\Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us\svs'

    save_path = r'X:\GXB\SNS\test_result\unet_4down_wsi'

    filter_names = ['tj190423329' , 'tj19042447' , 'tj190521313' , 'tj190522607' , 'tj190524610' , 'tj190529612' ,
                    'tj190531318' , 'tj190531604', 'tj190603606' , 'tj19060450' , 'tj190604601' , 'tj1906056090' ,
                    'tj19060566' , 'tj190610307' , 'tj190610603' , 'tj190610618' , 'tj19061317' , 'tj19061836' ,
                    'tj190619303' , 'tj190619608' , 'tj190619609' , 'tj19060460']

    new_filter_names = ['tj190619303' , 'tj190619608' , 'tj190619609']

    wsi_names = os.listdir(test_wsi_path)
    wsi_names = [x for x in wsi_names if x.find('.svs') != -1]

    for i , wsi_name in enumerate(wsi_names):
        if wsi_name[ : wsi_name.find('.sdpc.svs')] in new_filter_names:

            cpath = save_path + '/' + wsi_name
            if not os.path.exists(cpath):
                os.makedirs(cpath)

            try:

                print('handling WSI %s' % (wsi_name))

                wsi_reader = WSIReader(test_wsi_path + '/' + wsi_name , ratio = 0.1803)

                while wsi_reader.read_flag:
                    imgs , plist = wsi_reader.read_by_position_list(batch_size = 1000)
                    predict_masks = unet_model.predict(imgs , verbose = 0)
                    test_img_save(imgs , predict_masks , wsi_name , plist, cpath)

                wsi_reader.close_wsi()
            except:

                print('%s read exception ... ' % wsi_name)