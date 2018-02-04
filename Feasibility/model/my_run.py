import tensorflow as tf
import cv2
import numpy as np
import sys
sys.path.append('../../Pub/')
from file import *

def get_w_b(w_shape,b_shape):
    w = tf.Variable(tf.random_normal(w_shape, stddev=0.1), dtype=tf.float32)
    b = tf.Variable(tf.random_uniform(b_shape), dtype=tf.float32)
    return w, b





# print(img_fla.shape)
# print(160*sess.run(Y_p,feed_dict={X: img_fla}))
X = tf.placeholder(dtype=tf.float32, shape=[None, 160*60])
w_1, b_1 = get_w_b([160 * 60, 10], [10])
net = tf.nn.sigmoid(tf.matmul(X, w_1) + b_1)
w_2, b_2 = get_w_b([10, 10], [10])
net = tf.nn.sigmoid(tf.matmul(net, w_2) + b_2)
w_3, b_3 = get_w_b([10, 3], [3])
Y_p = tf.matmul(net, w_3) + b_3


saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('../fix_model/'))

names = get_file_name('../data/validate/')
for name in names:
    print(name)
    img = cv2.imread('../data/ori/'+name, 0)
    img_fla = np.reshape(img, [1, 160*60]) / 255.0
    line = (sess.run(Y_p, feed_dict={X: img_fla}))
    line *= 160
    np.reshape(line, [1, 3])
    line = list(np.floor(line[0]).astype(dtype=np.int32))
    print(line)
    for i in line:
        img[:, i] = 100
    cv2.imwrite('./result/'+name, img)