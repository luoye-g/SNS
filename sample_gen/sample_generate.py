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
              ''
              'SZSQ_originaldata/Tongji_4th/positive/svs' ,
              'SZSQ_originaldata/Tongji_4th/negative/svs']

xml_szsq_tj = ['SZSQ_originaldata/Labelfiles/xml_Tongji_3th' ,
               'None' ,
               'SZSQ_originaldata/Labelfiles/xml_Tongji_4th' ,
               'None']

# 图片的存储
def sample_save(ors , points , nums , block_size , level , save_path , cs , name):
    x, y, w, h = cv2.boundingRect(np.array(points))

    if w < 10 and h < 10:  # 不应该会这么小
        return

    try:
        #    在标注范围内进行随机撒点 ， 范围太大，重新确定范围

        xs = random.sample(range(-sample_range, sample_range), nums)
        ys = random.sample(range(-sample_range, sample_range), nums)

        xs.append(0)
        ys.append(0)

        xs = np.array(xs) + x + int(w // 2)
        ys = np.array(ys) + y + int(h // 2)

        #     从原图中读取区域
        for x, y in zip(xs, ys):
            x -= int(block_size // 2)
            y -= int(block_size // 2)
            img = ors.read_region((x, y), level, (block_size, block_size))
            img = np.array(img)
            img = img[:, :, 0: 3]
            img = img[:, :, ::-1]

            if not os.path.exists(save_path + '/' + cs):
                os.makedirs(save_path + '/' + cs)

            cv2.imwrite(
                save_path + '/' + cs + '/' + name + '_' + str(x) + '_' + str(y) + '_' + str(
                    level) + '.tif', img)
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
    post_index = '.sdpc.svs'
    save_path = 'X:/GXB/SNS/data'

    for i in range(2, len(xml_szsq_tj)):

        if xml_szsq_tj[i] is not 'None':

            xml_names = os.listdir(path + '/' + xml_szsq_tj[i])
            for name in xml_names:
                try:

                    ors = openslide.OpenSlide(path + '/' + cs_szsq_tj[i] + '/' + name[ : name.find('.xml')] + post_index)
                    mask_dict = xml_decoder(path + '/' + xml_szsq_tj[i] , name, level=level)
                    #         遍历每个阳性标签进行样本的制作
                    pool = dummy.Pool(int(cpu_count() // 2))
                    for j in range(0, len(mask_dict['Positive'])):
                        if j <= 200:
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

# resize img
def resize_img():
    save_path = 'X:/GXB/SNS/data'

    for cs in cs_3d[2 : ]:

        names = os.listdir(save_path + '/' + cs)
        names = [x for x in names if x.find('.tif') != -1]

        for name in names:

            img = cv2.imread(save_path + '/' + cs + '/' + name)
            img = cv2.resize(img , (256 , 256) , interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(save_path + '/' + cs + '/' + name , img)

# 区分训练时的训练集和测试
def split_data():
    path = 'X:/GXB/SNS/data'

    train_file = open(r'X:\GXB\SNS\data\train.txt' , 'w')
    test_file = open(r'X:\GXB\SNS\data\test.txt' , 'w')

    for cs in cs_3d + cs_szsq_tj:
        try:
            img_names = os.listdir(path + '/' + cs)
            img_names = [x for x in img_names if x.find('.tif') != -1]
            l = len(img_names)
            train_names = img_names[ : int(9 * l // 10)]
            test_names = img_names[int(9 * l // 10) : ]

            for name in train_names:
                train_file.write(path + '/' + cs + '/' + name + '\t' + path + '/mask/' + cs + '/' + name + '\n')
            for name in test_names:
                test_file.write(path + '/' + cs + '/' + name + '\t' + path + '/mask/' + cs + '/' + name + '\n')
        except:
            print(cs)

    train_file.close()
    test_file.close()

if __name__ == '__main__':

    generate_sample_3d()

    # generate_sample_szsq()

    # resize_img()
    # split_data()