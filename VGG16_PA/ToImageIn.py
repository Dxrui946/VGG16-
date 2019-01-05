#  -.- coding:utf8 -.-   By_Pastore

import os
import math
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


train_dir = '/Users/apple/Documents/Mycode/PA_Data/Data_Over6'

Black_grass = []
Charlock = []
Cleavers = []
Common_Chickweed = []
Common_wheat = []
Fat_Hen = []
Loose_Silky_bent = []
Maize = []
Scentless_Mayweed = []
Shepherds_Purse = []
Small_flowered_Cranesbill = []
Sugar_beet = []

label_Black_grass = []
label_Charlock = []
label_Cleavers = []
label_Common_Chickweed = []
label_Common_wheat = []
label_Fat_Hen = []
label_Loose_Silky_bent = []
label_Maize = []
label_Scentless_Mayweed = []
label_Shepherds_Purse = []
label_Small_flowered_Cranesbill = []
label_Sugar_beet = []


# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/Black_grass'):
        Black_grass.append(file_dir + '/Black_grass' + '/' + file)
        label_Black_grass.append(0)
    for file in os.listdir(file_dir + '/Charlock'):
        Charlock.append(file_dir + '/Charlock' + '/' + file)
        label_Charlock.append(1)
    for file in os.listdir(file_dir + '/Cleavers'):
        Cleavers.append(file_dir + '/Cleavers' + '/' + file)
        label_Cleavers.append(2)
    for file in os.listdir(file_dir + '/Common_Chickweed'):
        Common_Chickweed.append(file_dir + '/Common_Chickweed' + '/' + file)
        label_Common_Chickweed.append(3)
    for file in os.listdir(file_dir + '/Common_wheat'):
        Common_wheat.append(file_dir + '/Common_wheat' + '/' + file)
        label_Common_wheat.append(4)
    for file in os.listdir(file_dir + '/Fat_Hen'):
        Fat_Hen.append(file_dir + '/Fat_Hen' + '/' + file)
        label_Fat_Hen.append(5)
    for file in os.listdir(file_dir + '/Loose_Silky_bent'):
        Loose_Silky_bent.append(file_dir + '/Loose_Silky_bent' + '/' + file)
        label_Loose_Silky_bent.append(6)
    for file in os.listdir(file_dir + '/Maize'):
        Maize.append(file_dir + '/Maize' + '/' + file)
        label_Maize.append(7)
    for file in os.listdir(file_dir + '/Scentless_Mayweed'):
        Scentless_Mayweed.append(file_dir + '/Scentless_Mayweed' + '/' + file)
        label_Scentless_Mayweed.append(8)
    for file in os.listdir(file_dir + '/Shepherds_Purse'):
        Shepherds_Purse.append(file_dir + '/Shepherds_Purse' + '/' + file)
        label_Shepherds_Purse.append(9)
    for file in os.listdir(file_dir + '/Small_flowered_Cranesbill'):
        Small_flowered_Cranesbill.append(file_dir + '/Small_flowered_Cranesbill' + '/' + file)
        label_Small_flowered_Cranesbill.append(10)
    for file in os.listdir(file_dir + '/Sugar_beet'):
        Sugar_beet.append(file_dir + '/Sugar_beet' + '/' + file)
        label_Sugar_beet.append(11)

    image_list = np.hstack((Black_grass, Charlock, Cleavers, Common_Chickweed,
                            Common_wheat, Fat_Hen, Loose_Silky_bent, Maize,Scentless_Mayweed,
                            Shepherds_Purse, Small_flowered_Cranesbill,Sugar_beet))
    label_list = np.hstack((label_Black_grass,label_Charlock,label_Cleavers, label_Common_Chickweed, label_Common_wheat,
                            label_Fat_Hen,label_Loose_Silky_bent,label_Maize,label_Scentless_Mayweed,
                            label_Shepherds_Purse,label_Small_flowered_Cranesbill,label_Sugar_beet))
    # print(image_list)
    # print(label_list)

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)


    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)


    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


# if __name__ == '__main__':
#     Ratio = 0.3
#     theResultFile = get_files(train_dir, Ratio)
#     # print(type(theResult))
#     #训练集合与测试集合
#     theTrainImg = theResultFile[0]
#     theTrainLab = theResultFile[1]
#     theValImg = theResultFile[2]
#     theValLab = theResultFile[3]
#
#     #Batch结果
#     for i in range(3):
#         theResultBatch = get_batch(theTrainImg,theTrainLab,224, 224, 32, 596)
#         print(theResultBatch)
