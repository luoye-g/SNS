'''
    增加数据增强
'''

'''
    病变核分割
'''
from models.unet_u import unet_4down
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import multi_gpu_model
from read_sample.SampleRead import SampleRead
import numpy as np
from tensorboardX import SummaryWriter
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


log_path = 'X:/GXB/SNS/train_log/unet4down_enhan/'
if not os.path.exists(log_path + 'imgs/'):
    os.makedirs(log_path + 'imgs/')
if not os.path.exists(log_path + 'weights/'):
    os.makedirs(log_path + 'weights/')

log = SummaryWriter(log_path + 'run_log')

'''
    :parameter set
'''

gpu_num = 1
input_shape = (256 , 256 , 3)
lr = 1e-4
loss_function  = 'categorical_crossentropy'
acc_function = 'categorical_accuracy'
classes = 3
train_epoch = 100
batch_size = 8
log_step = 50
step = 0

save_log_img_nums = 10

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

def log_img_save(test_imgs , test_masks , predict_masks , epoch , batch_i):

    for i in range(0 , save_log_img_nums):

        img = np.uint8(test_imgs[i] * 255.)
        mask = np.uint8(test_masks[i] * 255.)
        p_mask = predict_masks[i]
        p_img = img.copy()

        contours , _ = cv2.findContours(mask[: , : , 0] , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img , contours , -1 , (255 , 0 , 0) , 1)
        contours, _ = cv2.findContours(mask[:, :, 1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (0 , 255 , 0), 1)

        wj = p_mask.max(axis = 2)
        p_mask[: , : , 0] = p_mask[: , : , 0] == wj
        p_mask[: , : , 1] = p_mask[: , : , 1] == wj
        p_mask[: , : , 2] = p_mask[: , : , 2] == wj
        # print('multi channel : %d %d %d' % (np.sum(p_mask[: , : , 0]) , np.sum(p_mask[: , : , 1]) , np.sum(p_mask[: , : , 2])))
        p_mask = np.uint8(p_mask * 255)

        contours , _ = cv2.findContours(p_mask[: , : , 0] , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        p_img = cv2.drawContours(p_img , contours , -1 , (255 , 0 , 0) , 1)
        contours, _ = cv2.findContours(p_mask[:, :, 1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_img = cv2.drawContours(p_img, contours, -1, (0 , 255 , 0), 1)

        combined_img = cv2.hconcat([img , mask , p_img , p_mask])
        cv2.imwrite(log_path + 'imgs/%d_%d_%d.tif' % (epoch , batch_i , i) , combined_img)

if __name__ == '__main__':

    unet_model = generate_model(None)

    train_file = r'X:\GXB\SNS\data\train.txt'
    test_file = r'X:\GXB\SNS\data\test.txt'

    sr = SampleRead(train_file , test_file , batch_size = batch_size)

    for epoch in range(1 , train_epoch + 1):

        sr.init_data_list(is_train = True)

        batch_i = 0
        while sr.data_flag:
            imgs, masks = sr.generate_data()
            # print(sr.index, np.shape(imgs), np.shape(masks))
            class_weight = [1 , 0.5 , 0.5]
            loss = unet_model.train_on_batch(imgs , masks , class_weight = class_weight)
            print('[Epoch %d/%d] [Batch %d] [loss: %f] [accuracy: %.2f%%]' % (
            epoch, train_epoch, batch_i , loss[0] , loss[1] * 100))

            if batch_i % log_step == 0:
                unet_model.save_weights(log_path + 'weights/' + 'kernel_seg_' + str(epoch) + '_' + str(batch_i) + '.h5')

                # 在测试集上进行模型的评估
                test_imgs , test_masks , _   = sr.generate_test_data()
                v_loss = unet_model.evaluate(test_imgs , test_masks , batch_size = batch_size ,  verbose = 0)

                predict_masks = unet_model.predict(test_imgs , batch_size = batch_size , verbose = 0)
                log_img_save(test_imgs , test_masks , predict_masks , epoch , batch_i)

                # 进行日志的记录
                log.add_scalars('scalar/loss', {'loss': loss[0] , 'v_loss' : v_loss[0]}, step)
                log.add_scalars('scalar/acc', {'acc': loss[1] , 'v_acc' : v_loss[1]}, step)

                print('[Epoch %d/%d] [Batch %d] [v_loss: %f] [v_accuracy: %.2f%%]' % (
                    epoch, train_epoch, batch_i, v_loss[0], v_loss[1] * 100))

            batch_i += 1
            step += 1
