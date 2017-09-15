import tensorflow as tf
import vgg19_trainable as vgg19
import utils
import numpy as np
import os
import itertools
import random


#parameters
M = 1e15

location = os.listdir("test_data/icons/png/")
combination = list(itertools.combinations(location, 2))
training_positve = []
training_negative = []
for cmb in combination:

    if(cmb[0].split("|")[1] == "angry_birds" and  cmb[1].split("|")[1] != "angry_birds"):
        training_negative.append([ list(cmb), 0])
    elif(cmb[0].split("|")[1] != "angry_birds" and  cmb[1].split("|")[1] == "angry_birds"):
        training_negative.append([ list(cmb), 0])
    elif(cmb[0].split("|")[1] == "angry_birds" and  cmb[1].split("|")[1] == "angry_birds"):
        training_positve.append([ list(cmb), 1])

training_negative = random.sample(training_negative, 100)
training_positve = random.sample(training_positve, 100)

training = training_positve + training_negative
print(len(training_negative))
print(len(training_positve))

count = 0

with tf.device('/cpu:0'), tf.Session() as sess:
    # sess = tf.Session()
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./test1.npy')
    vgg.build(images, train_mode)

    sess.run(tf.global_variables_initializer())
    for sample in training_negative:

        img1 = utils.load_image("test_data/icons/png/" + sample[0][0][0:-4] + ".png")
        img2 = utils.load_image("test_data/icons/png/" + sample[0][1][0:-4] + ".png")
        y = sample[1]

        print(count)
        count += 1
        batch1 = img1.reshape((1, 224, 224, 3))
        batch2 = img2.reshape((1, 224, 224, 3))

        # first image feedforward
        features1 = sess.run(vgg.conv5_1, feed_dict={images: batch1, train_mode: False})
        features1 = np.reshape(features1, (-1,512))
        gram1 = np.matmul(features1.T, features1) / features1.size

        a = tf.reshape(vgg.conv5_1,[-1,512])
        if(y==1):
            cost = tf.reduce_sum(( gram1 - tf.matmul(tf.transpose(a), a)/(14*14*512) ) ** 2)
        else:
            cost =   tf.maximum( 0.0 , M**2 - tf.reduce_sum(( gram1 - tf.matmul(tf.transpose(a), a)/(14*14*512) ) ** 2) )
        #traing step
        train = tf.train.GradientDescentOptimizer(1e-11).minimize(cost)
        sess.run(train, feed_dict={images: batch2, train_mode: True})

        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './test2.npy')
