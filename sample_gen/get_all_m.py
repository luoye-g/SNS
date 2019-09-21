import os
import cv2
import openslide
import random
from multiprocessing import cpu_count
from multiprocessing import dummy
from utils_pac.mask_decoder import *


sample_range = int(100 // 2)

path = 'H:/TCTDATA'

# 3d sfy1 and sfy2 20x 0.243
cs_3d = ['Shengfuyou_1th/Positive' , 'Shengfuyou_1th/Negative' ,
         'Shengfuyou_2th/Positive' , 'Shengfuyou_2th/Negative']
csv_3d = ['Shengfuyou_1th/Labelfiles/csv_P' , 'Shengfuyou_1th/Labelfiles/csv_N' ,
          'Shengfuyou_2th/LabelFiles/csv_P' , 'None']

# szsq tongji3 and tongji4 40x 0.1803
cs_szsq_tj = ['SZSQ_originaldata/Tongji_3th/positive/tongji_3th_positive_40x/svs' ,
              'SZSQ_originaldata/Tongji_3th/negative/tongji_3th_negtive_40x/svs' ,
              'SZSQ_originaldata/Tongji_4th/positive/svs' ,
              'SZSQ_originaldata/Tongji_4th/negative/svs' ,
              'SZSQ_originaldata/Shengfuyou_3th/positive/Shengfuyou_3th_positive_40X/svs']

xml_szsq_tj = ['SZSQ_originaldata/Labelfiles/xml_Tongji_3th' ,
               'None' ,
               'SZSQ_originaldata/Labelfiles/xml_Tongji_4th' ,
               'None' ,
               'SZSQ_originaldata/Labelfiles/xml_Shengfuyou_3th']

# 图片的存储
def sample_save(ors , points , nums , block_size , level , save_path , cs , name):
    x, y, w, h = cv2.boundingRect(np.array(points))

    if w < 10 and h < 10:  # 不应该会这么小
        return

    try:
        #    在标注范围内进行随机撒点 ， 范围太大，重新确定范围
        position = (x - block_size, y - block_size)

        img = ors.read_region(position , level , (w + 2 * block_size, h + 2 * block_size))
        img = np.array(img)
        img = img[:, :, 0: 3]
        img = img[:, :, ::-1]
        # try:
        points = np.array(points)
        points[: , 0] -= position[0]
        points[: , 1] -= position[1]
        c_img = img.copy()
        c_img = cv2.drawContours(c_img , [points] , -1 , (0 , 0 , 255) , 2)
        # except:
        #     print('draw contours error')

        if not os.path.exists(save_path + '/' + cs + '/Img/'):
            os.makedirs(save_path + '/' + cs + '/Img/')

        cv2.imwrite(
            save_path + '/' + cs + '/Img/' + name + '_' + str(position[0]) + '_' + str(position[1]) + '_' + str(
                level) + '.tif', img)

        if not os.path.exists(save_path + '/' + cs + '/C_Img/'):
            os.makedirs(save_path + '/' + cs + '/C_Img/')

        cv2.imwrite(
            save_path + '/' + cs + '/C_Img/' + name + '_' + str(position[0]) + '_' + str(position[1]) + '_' + str(
                level) + '.tif', c_img)

        if not os.path.exists(save_path + '/' + cs + '/Contours/'):
            os.makedirs(save_path + '/' + cs + '/Contours/')
        np.save(
            save_path + '/' + cs + '/Contours/' + name + '_' + str(position[0]) + '_' + str(position[1]) + '_' + str(
                level), points)
    except:
        print(path + '/' + cs + '/' + name, points)



# generate sample 3d
def generate_sample_3d():
    # 对 sfy1-2的数据取0.243下的309大小块，对应于0.293下的256大小
    block_size = 309
    nums = 2  # 设置样本数量
    level = 0
    post_index = '.mrxs'
    save_path = 'X:/GXB/SNS/data1'
    for i in range(0, len(csv_3d)):

        if csv_3d[i] is not 'None':

            csv_names = os.listdir(path + '/' + csv_3d[i])
            for name in csv_names:
                try:
                    ors = openslide.OpenSlide(path + '/' + cs_3d[i] + '/' + name + post_index)
                    mask_dict = csv_decoder(path + '/' + csv_3d[i], name, level=level)
                    #         遍历每个阳性标签进行样本的制作
                    pool = dummy.Pool(int(cpu_count() // 2))
                    for j in range(0 , len(mask_dict['Positive'])):
                        # if j <= 200:
                        pool.apply_async(sample_save ,
                                         args = (ors , mask_dict['Positive'][j] , nums , block_size , level , save_path , cs_3d[i] , name))
                    pool.close()
                    pool.join()

                    ors.close()
                except:
                    print(csv_3d[i], name)
        else:
            pass

# generate sample szsq
def generate_sample_szsq():
    # 对 tongji3-4的数据取0.1803下的416大小块 ， 对应于0.293下的256
    block_size = 416
    nums = 2  # 设置样本数量
    level = 0
    post_index = '.svs'
    save_path = 'X:/GXB/SNS/data1'

    for i in range(4 , 5):

        if xml_szsq_tj[i] is not 'None':

            xml_names = os.listdir(path + '/' + xml_szsq_tj[i])
            for name in xml_names:
                try:
                    # print(path + '/' + cs_szsq_tj[i] + '/' + name[ : name.find('.xml')] + post_index)
                    ors = openslide.OpenSlide(path + '/' + cs_szsq_tj[i] + '/' + name[ : name.find(' ')] + post_index)
                    mask_dict = xml_decoder(path + '/' + xml_szsq_tj[i] , name, level=level)
                    #         遍历每个阳性标签进行样本的制作
                    pool = dummy.Pool(int(cpu_count() // 2))
                    for j in range(0, len(mask_dict['Positive'])):
                        # if j <= 200:
                        pool.apply_async(sample_save,
                                         args=(
                                         ors, mask_dict['Positive'][j], nums, block_size, level, save_path, cs_szsq_tj[i],
                                         name))
                    pool.close()
                    pool.join()

                    ors.close()
                except:
                    print(xml_szsq_tj[i] , name)
        else:
            pass


if __name__ == '__main__':

    # generate_sample_3d()

    generate_sample_szsq()

    # resize_img()
    # split_data()