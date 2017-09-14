"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import numpy as np


#parameters
M = 100000

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")
y = 0
count = 0

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    # first image feedforward
    features1 = sess.run(vgg.conv5_1, feed_dict={images: batch1, train_mode: False})
    features1 = np.reshape(features1, (-1,512))
    gram1 = np.matmul(features1.T, features1) / features1.size
    print("first image done.")


    a = tf.reshape(vgg.conv5_1,[-1,512])
    if(y==1):
        cost = tf.reduce_sum(( gram1 - tf.matmul(tf.transpose(a), a)/(14*14*512) ) ** 2)
    else:
        cost =   tf.maximum( 0.0 , M**2 - tf.reduce_sum(( gram1 - tf.matmul(tf.transpose(a), a)/(14*14*512) ) ** 2) )
    #traing step
    train = tf.train.GradientDescentOptimizer(0.000000000001).minimize(cost)
    sess.run(train, feed_dict={images: batch2, train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # test save
    # vgg.save_npy(sess, './test-save.npy')
