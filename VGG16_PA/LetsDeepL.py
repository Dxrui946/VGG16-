#  -.- coding:utf8 -.-   By_Pastore

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#__________________________________________________________________________

#卷积层操作
#images:输入   name：本层名称    Jh：卷积核的高   Jk：卷积核的宽   Bh：步长的高  Bk：步长的宽   channels_out：输出通道数
def conv_op(images, name, Jh, Jk, Bh, Bk, channels_out):
    channels_in = images.get_shape()[-1].value
    with tf.name_scope(name) as scope:

        weights = tf.get_variable(scope + "w",shape=[Jh, Jk, channels_in, channels_out],dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[channels_out]),
                             name='biases',dtype=tf.float32)
        conv = tf.nn.conv2d(images, weights, strides=[1,Bh,Bk,1],padding='SAME')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        return activation

#最大池化层
#增加局部响应归一化   /   没有加(查阅资料发现效果不行)
def maxpool_op(images, name,Jh, Jk, Bh, Bk):
    pool_1 = tf.nn.max_pool(images,ksize=[1,Jh,Jk,1],strides=[1,Bh,Bk,1],padding='SAME',name=name)
    return pool_1

#全连接层
def fc_op(images,name, channels_out):
    channels_in = images.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+'w',shape=[channels_in,channels_out],dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(value=0.1,dtype=tf.float32,shape=[channels_out]),
                               name='biases',dtype=tf.float32)

        activation = tf.nn.relu_layer(images, weights, biases, name=scope)
        return activation


#images 为输入，   Dr_rate为控制Dropout的失活率
#输入像素：224*224*3
def inference(images,BATCH_SIZE, N_CLASSES,Dr_rate):
    #VGG16_Mode D-1     两次卷积
    #卷积   -----像素：64*64*64    参数：3*3*3*64 + 3*3*64
    #池化   -----像素：32*32*32    参数：0
    conv1_1 = conv_op(images,name='conv1_1', Jh=3, Jk=3, channels_out=64, Bh=1, Bk=1)
    conv1_2 = conv_op(conv1_1,name='conv1_2', Jh=3, Jk=3, channels_out=64, Bh=1, Bk=1)
    pool1 = maxpool_op(conv1_2,name='pool1', Jh=2, Jk=2, Bh=2, Bk=2)

    # #VGG16 Mode D-2    两次卷积
    # # 卷积   -----像素：32*32*128    参数：
    # # 池化   -----像素：32*32*32    参数：
    conv2_1 = conv_op(pool1,name='conv2_1', Jh=3, Jk=3, channels_out=128, Bh=1, Bk=1)
    # conv2_2 = conv_op(conv2_1,name='conv2_2', Jh=3, Jk=3, channels_out=128, Bh=1, Bk=1)
    pool2 = maxpool_op(conv2_1,name='pool2', Jh=2, Jk=2, Bh=2, Bk=2)

    # # VGG16 Mode D-3    三次卷积  ----处于本电脑计算量的考虑，减为两层
    # # 卷积   -----像素：32*32*128    参数：
    # # 池化   -----像素：32*32*32    参数：
    # conv3_1 = conv_op(pool2, name='conv3_1', Jh=3, Jk=3, channels_out=256, Bh=1, Bk=1)
    # conv3_2 = conv_op(conv3_1, name='conv3_2', Jh=3, Jk=3, channels_out=256, Bh=1, Bk=1)
    # conv3_3 = conv_op(conv3_2, name='conv3_2', Jh=3, Jk=3, channels_out=256, Bh=1, Bk=1)
    # pool3 = maxpool_op(conv3_3, name='pool3', Jh=2, Jk=2, Bh=2, Bk=2)

    # # VGG16 Mode D-4    三次卷积  ----处于本电脑计算量的考虑，减为一层
    # # 卷积   -----像素：32*32*128    参数：
    # # 池化   -----像素：32*32*32    参数：
    # conv4_1 = conv_op(images, name='conv4_1', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # # conv4_2 = conv_op(images, name='conv4_2', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # # conv4_3 = conv_op(images, name='conv4_2', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # pool4 = maxpool_op(conv4_1, name='pool4', Jh=2, Jk=2, Bh=2, Bk=2)



    # VGG16 Mode D-5    三次卷积  ----处于本电脑计算量的考虑，全部减去
    # 卷积   -----像素：32*32*128    参数：
    # 池化   -----像素：32*32*32    参数：
    # conv5_1 = conv_op(images, name='conv5_1', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # conv5_2 = conv_op(images, name='conv5_2', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # conv5_3 = conv_op(images, name='conv5_2', Jh=3, Jk=3, channels_out=512, Bh=1, Bk=1)
    # pool5 = maxpool_op(conv5_1, name='pool5', Jh=2, Jk=2, Bh=2, Bk=2)


    #将三维变为一维
    TheShape_last = pool2.get_shape()
    Flatten_shape = TheShape_last[1].value * TheShape_last[2].value * TheShape_last[3].value
    Reshape1 = tf.reshape(pool2,[-1,Flatten_shape],name='Reshape1')

    # dropout层
    #    with tf.variable_scope('dropout') as scope:
    #        drop_out = tf.nn.dropout(Reshape1, 0.8)

    #全连接  ————— 测试
    fc5 = fc_op(Reshape1, name='fc5', channels_out=128)
    fc5_dropout = tf.nn.dropout(fc5,keep_prob=Dr_rate,name='fc5_dropout')
    fc6 = fc_op(fc5_dropout,name='fc6',channels_out=64)
    fc6_dropout = tf.nn.dropout(fc6,keep_prob=Dr_rate,name='fc6_dropout')
    fc7 = fc_op(fc6_dropout,name='fc7',channels_out=12)


    #全连接  ______正常
    # fc5 = fc_op(Reshape1,name='fc5',channels_out=1024)
    # fc5_dropout = tf.nn.dropout(fc5,keep_prob=Dr_rate,name='fc5_dropout')
    # fc6 = fc_op(fc5_dropout,name='fc6',channels_out=1024)
    # fc6_dropout = tf.nn.dropout(fc6,keep_prob=Dr_rate,name='fc6_dropout')
    # fc7 = fc_op(fc6_dropout,name='fc7',channels_out=12)


    #Softmax回归
    #Softmax = tf.nn.softmax(fc7)

    return fc7
#+______________________________________________________________________________________________________





# #______________________________________________________________________________________________________
# def inference(images, batch_size, n_classes):
#     # 一个简单的卷积神经网络，卷积+池化层x2，全连接层x2，最后一个softmax层做分类。
#     # 卷积层1
#     # 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
#     with tf.variable_scope('conv1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
#                              name='biases', dtype=tf.float32)
#
#         conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     # 池化层1
#     # 3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
#     with tf.variable_scope('pooling1_lrn') as scope:
#         pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     # 卷积层2
#     # 16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
#     with tf.variable_scope('conv2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
#                              name='biases', dtype=tf.float32)
#
#         conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name='conv2')
#
#     # 池化层2
#     # 3x3最大池化，步长strides为2，池化后执行lrn()操作，
#     # pool2 and norm2
#     with tf.variable_scope('pooling2_lrn') as scope:
#         norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#         pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
#
#     # 全连接层3
#     # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
#     with tf.variable_scope('local3') as scope:
#         reshape = tf.reshape(pool2, shape=[batch_size, -1])
#         dim = reshape.get_shape()[1].value
#         weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
#                              name='biases', dtype=tf.float32)
#
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#
#     # 全连接层4
#     # 128个神经元，激活函数relu()
#     with tf.variable_scope('local4') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
#                              name='biases', dtype=tf.float32)
#
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
#
#     # dropout层
#     #    with tf.variable_scope('dropout') as scope:
#     #        drop_out = tf.nn.dropout(local4, 0.8)
#
#
#     # Softmax回归层
#     # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[128, n_classes], stddev=0.005, dtype=tf.float32),
#                               name='softmax_linear', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
#                              name='biases', dtype=tf.float32)
#
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
#
#     return softmax_linear


# -__________________________________________________________________________________________
# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------
# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
