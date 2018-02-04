import cv2
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../Pub/')
from file import *

TRAIN_PATH = 'E:/PycharmProjects/V_CODE/Feasibility/data/ori/'
VALIDATE_PATH = 'E:/PycharmProjects/V_CODE/Feasibility/data/validate/'

IMG_HEIGHT = 60
IMG_WIDTH = 160
TRAIN_NUM = 18
LR = 1e-6


def load_single():
    name = shuffle_file(VALIDATE_PATH,)
    # print(name)
    img = cv2.imread(TRAIN_PATH+name[0], 0)
    img = np.reshape(img, [1, IMG_HEIGHT*IMG_WIDTH])
    fr = open('../data/validate.txt', 'r')
    lines = fr.readlines()
    label = np.zeros((1, 3))
    for line in lines:
        line = line.split()
        # print(line)
        if line[0]+'.jpg' == name[0]:
            label = np.array([float(line[1]), float(line[2]), float(line[3])])
    return img / 255.0, label, name
# a, b = load_single()
# print(b)


def get_data():
    names = []
    data = np.zeros((TRAIN_NUM, IMG_HEIGHT*IMG_WIDTH))
    label = np.zeros((TRAIN_NUM, 3))

    fr = open('../data/train.txt', 'r')
    index = 0
    for line in fr.readlines()[0:TRAIN_NUM]:
        temp_l = np.zeros((1, 3))
        line = line.split()
        names.append(line[0]+'.jpg')
        temp_l = np.array([float(line[1]), float(line[2]),float(line[3])])
        label[index] = temp_l
        index += 1

    for i in range(TRAIN_NUM):
        img = cv2.imread(TRAIN_PATH+names[i], 0)
        img = np.reshape(img, [1, IMG_HEIGHT*IMG_WIDTH])
        data[i] = img
    return data / 255.0, label / float(IMG_WIDTH)

# print(y)


def get_w_b(w_shape,b_shape):
    w = tf.Variable(tf.random_normal(w_shape, stddev=0.1), dtype=tf.float32)
    b = tf.Variable(tf.random_uniform(b_shape), dtype=tf.float32)
    return w, b



if __name__ == '__main__':
    X = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT * IMG_WIDTH])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    x, y = get_data()
# print(d.shape, l.shape)

    # structure: input->10->10->3
    w_1, b_1 = get_w_b([IMG_HEIGHT*IMG_WIDTH, 10], [10])
    net = tf.nn.sigmoid(tf.matmul(X, w_1)+b_1)
    w_2, b_2 = get_w_b([10, 10], [10])
    net = tf.nn.sigmoid(tf.matmul(net, w_2)+b_2)
    w_3, b_3 = get_w_b([10, 3], [3])
    Y_p = tf.nn.relu(tf.matmul(net, w_3)+b_3)

    loss = tf.reduce_mean(tf.square(Y-Y_p))
    train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)


    saver = tf.train.Saver()
    l1 = 0.0
    l2 = 1.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000000):
            sess.run(train_op, feed_dict={X: x, Y: y})
            if not _ % 9999:
                l1 = sess.run(loss, feed_dict={X: x, Y: y})
                if (l2 - l1) < 0.00001:
                    saver.save(sess, '../fix_model/'+str(_+1)+'.ckpt')
                    break
                else:
                    print(l1)
                    l2 = l1

                # img, lab, name = load_single()
                # pred = IMG_WIDTH*sess.run(Y_p, feed_dict={X: img})
# 0.000853871