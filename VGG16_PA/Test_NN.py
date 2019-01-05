# -.- coding:utf8 -.-  
# By:Pastore_Zx          Carpe Diem

# =============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import LetsDeepL
from ToImageIn import get_files
import cv2
import os


# =======================================================================
# 获取一张图片
def get_one_image(train_img):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    # n = len(train)
    # ind = np.random.randint(0, n)

    # for i in os.listdir(train):
    #     img_dir = train+'/'+i   # 随机选择测试的图片

    img = Image.open(train_img)
    plt.imshow(img)
    # imag = img.resize([64, 64])  # 由于图片在预处理阶段以及resize，因此该命令可略
    img = cv2.imread(train_img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = np.array(img)
    return image


# --------------------------------------------------------------------
# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 16
        N_CLASSES = 12

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 256, 256, 3])

        logit = LetsDeepL.inference(image, BATCH_SIZE, N_CLASSES,0.5)

        # logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[256, 256, 3])

        logs_train_dir = '/Users/apple/Documents/Mycode/PA_Data/Data_Over5'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            # print(prediction)
            max_index = np.argmax(prediction)
            # print(max_index)
            if max_index == 0:
                print('This is a Black_grass with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a Charlock with possibility %.6f' % prediction[:, 1])
            elif max_index == 2:
                print('This is a Cleavers with possibility %.6f' % prediction[:, 2])
            elif max_index == 3:
                print('This is a Common_Chickweed with possibility %.6f' % prediction[:, 3])
            elif max_index == 4:
                print('This is a Common_wheat with possibility %.6f' % prediction[:, 4])
            elif max_index == 5:
                print('This is a Fat_Hen with possibility %.6f' % prediction[:, 5])
            elif max_index == 6:
                print('This is a Loose_Silky_bent with possibility %.6f' % prediction[:, 6])
            elif max_index == 7:
                print('This is a Maize with possibility %.6f' % prediction[:, 7])
            elif max_index == 8:
                print('This is a Scentless_Mayweed with possibility %.6f' % prediction[:, 8])
            elif max_index == 9:
                print('This is a Shepherds_Purse with possibility %.6f' % prediction[:, 9])
            elif max_index == 10:
                print('This is a Small_flowered_Cranesbill with possibility %.6f' % prediction[:, 10])
            else:
                print('This is a Sugar_beet with possibility %.6f' % prediction[:, 11])
            print(end='\n')


# ------------------------------------------------------------------------

if __name__ == '__main__':
    # train_dir = 'E:/Re_train/image_data/inputdata'
    # train, train_label, val, val_label = get_files(train_dir, 0.3)
    # img = get_one_image(val)  # 通过改变参数train or val，进而验证训练集或测试集
    train_dir = '/Users/apple/Documents/Mycode/PA_Data/Data_Over5/sample_test'
    n = len(train_dir)

    for i in os.listdir(train_dir):
        theRealdir = train_dir+'/'+i
        img = get_one_image(theRealdir)
        print('This a test about '+i)
        evaluate_one_image(img)
    # ===========================================================================</span>