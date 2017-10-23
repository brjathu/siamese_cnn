import tensorflow as tf
import vgg19_trainable as vgg19
import utils
import numpy as np
import os
import itertools
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics
import seaborn as sns

tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)

# parameters
M = 1e6
EPOCH = 50
LR = 1e-11

model = "test2"
main_dir = "/flush1/raj034/vgg19/" + model + "/"
os.system("mkdir " + main_dir)
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

LOG_FILE = open(main_dir + 'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


# train positive samples
y = 1
batch = 5


# create tf models
images1 = tf.placeholder(tf.float32, [batch, 224, 224, 3])
images2 = tf.placeholder(tf.float32, [batch, 224, 224, 3])
pos = tf.placeholder(tf.float32, [batch, 1])
neg = tf.placeholder(tf.float32, [batch, 1])
train_mode = tf.placeholder(tf.bool)
vgg1 = vgg19.Vgg19('vgg19.npy')
vgg1.build(images1, train_mode)

vgg2 = vgg19.Vgg19('vgg19.npy')
vgg2.build(images2, train_mode)

features1 = tf.reshape(vgg1.conv5_1, [batch, -1, 512])
features2 = tf.reshape(vgg2.conv5_1, [batch, -1, 512])

gram1 = tf.matmul(tf.transpose(features1, perm=[0, 2, 1]), features1) / tf.cast(tf.size(features1), tf.float32)
gram2 = tf.matmul(tf.transpose(features2, perm=[0, 2, 1]), features2) / tf.cast(tf.size(features2), tf.float32)
# logEntry("debugger 2 ========= >  " + str(features1.shape))
# logEntry("debugger 3 ========= >  " + str(gram1.shape))

cost = tf.multiply(tf.reshape(tf.reduce_sum((gram1 - gram2) / (14 * 14 * 512), axis=[1, 2]) ** 2, (batch, 1)), pos) + \
    tf.multiply(tf.reshape(M - tf.reduce_sum((gram1 - gram2) / (14 * 14 * 512), axis=[1, 2]) ** 2, (batch, 1)), neg)
# logEntry("debugger 6 ========= >  " + str(cost.shape))

# traing step
train = tf.train.GradientDescentOptimizer(LR).minimize(cost)

loss = 0
loss_graph = []
with tf.device('/cpu'), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        loss_avg = 0

        for class_style_1 in class_list:
            # saver.save(sess, 'train/model' + str(class_style_1), write_meta_graph=False)
            logEntry(str(class_style_1) + "   ========   positive samples ")

            class_1 = sorted(os.listdir("/flush1/raj034/WIKI_STYLE/" + str(class_style_1) + "/img/"))[:batch]
            class_2 = sorted(os.listdir("/flush1/raj034/WIKI_STYLE/" + str(class_style_1) + "/img/"))[batch:batch * 2]

            y_pos = np.ones((batch, 1))
            y_neg = np.zeros((batch, 1))
            count = 0
            for i in range(batch):
                logEntry(str(class_style_1) + "   ========   " + str(class_1[i]))

                img1 = utils.load_image("/flush1/raj034/WIKI_STYLE/" +
                                        str(class_style_1) + "/img/" + class_1[i])
                img2 = utils.load_image("/flush1/raj034/WIKI_STYLE/" +
                                        str(class_style_1) + "/img/" + class_2[i])

                if count == 0:
                    batch1 = img1.reshape((1, 224, 224, 3))
                    batch2 = img2.reshape((1, 224, 224, 3))
                else:
                    batch1 = np.concatenate((batch1, img1.reshape((1, 224, 224, 3))), 0)
                    batch2 = np.concatenate((batch2, img2.reshape((1, 224, 224, 3))), 0)

                count += 1
                # logEntry("debugger 0 ========= >  " + str(count) + "    " + str(batch1.shape))

            _, costp = sess.run([train, cost], feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: True})
            logEntry("debugger 4 ========= >  " + str(costp))

            # test classification again, should have a higher probability about tiger
            prob = sess.run(vgg1.prob, feed_dict={images1: batch1, train_mode: False})
            logEntry(str(class_style_1) + "      " + str(utils.print_prob(prob[0], './synset.txt')))
            prob = sess.run(vgg2.prob, feed_dict={images2: batch1, train_mode: False})
            logEntry(str(class_style_1) + "      " + str(utils.print_prob(prob[0], './synset.txt')))
            loss_avg += np.sum(costp)

        loss_graph.append(loss_avg)
        np.save(str("/flush1/raj034/vgg19/" + model + "/") + "loss.npy", loss_graph)
        plt.figure(0)
        plt.plot(np.array(loss_graph))
        plt.savefig(str("/flush1/raj034/vgg19/" + model + "/") + "loss_graph.png")
        os.system("mkdir /flush1/raj034/vgg19/" + model + "/" + str(epoch))
        vgg1.save_npy(sess, str("/flush1/raj034/vgg19/" + model + "/" + str(epoch) + "/vgg19_1.npy"))
        vgg2.save_npy(sess, str("/flush1/raj034/vgg19/" + model + "/" + str(epoch) + "/vgg19_2.npy"))
        # plt.show()

#         saver.save(sess, "model/" + str(MODEL) + "/ckpt/dnn.ckpt")
    # for class_style_1 in class_list:
    #     # saver.save(sess, 'train/model' + str(class_style_1), write_meta_graph=False)
    #     for class_style_2 in class_list:
    #         logEntry(str(class_style_1) + "   ========   " + str(class_style_2))
    #         if(class_style_1 == class_style_2):
    #             y = 1
    #             batch = 40
    #             LR = 1e-13
    #         else:
    #             y = 0
    #             batch = 5
    #             LR = 1e-14
    #         class_1 = random.sample(os.listdir(
    #             "/flush1/raj034/WIKI_STYLE/" + str(class_style_1) + "/img/"), batch)
    #         class_2 = random.sample(os.listdir(
    #             "/flush1/raj034/WIKI_STYLE/" + str(class_style_2) + "/img/"), batch)

    #         count = 0
    #         for i in range(batch):
    #             img1 = utils.load_image("/flush1/raj034/WIKI_STYLE/" +
    #                                     str(class_style_1) + "/img/" + class_1[i])
    #             img2 = utils.load_image("/flush1/raj034/WIKI_STYLE/" +
    #                                     str(class_style_2) + "/img/" + class_2[i])

    #             if count == 0:
    #                 batch1 = img1.reshape((1, 224, 224, 3))
    #                 batch2 = img2.reshape((1, 224, 224, 3))
    #             else:
    #                 batch1 = np.concatenate((batch1, img1.reshape((1, 224, 224, 3))), 0)
    #                 batch2 = np.concatenate((batch2, img2.reshape((1, 224, 224, 3))), 0)

    #             count += 1

    #         # create tf models
    #         images = tf.placeholder(tf.float32, [batch, 224, 224, 3])
    #         train_mode = tf.placeholder(tf.bool)
    #         vgg = vgg19.Vgg19('vgg19.npy')
    #         vgg.build(images, train_mode)
    #         sess.run(tf.global_variables_initializer())
    #         # saver = tf.train.Saver()
    #         # saver.restore(sess, 'train/model13')
    #         for epoch in range(EPOCH):
    #             features1 = sess.run(vgg.conv5_1, feed_dict={
    #                                  images: batch1, train_mode: False})
    #             print("debugger 1 ========= >  ", features1.shape)
    #             features1 = np.reshape(features1, (-1, 512))
    #             gram1 = np.matmul(features1.T, features1) / features1.size

    #             a = tf.reshape(vgg.conv5_1, [-1, 512])
    #             if(y == 1):
    #                 cost = tf.reduce_sum(
    #                     (gram1 - tf.matmul(tf.transpose(a), a) / (14 * 14 * 512)) ** 2)
    #             else:
    #                 cost = tf.maximum(
    #                     0.0, M**2 - tf.reduce_sum((gram1 - tf.matmul(tf.transpose(a), a) / (14 * 14 * 512)) ** 2))
    #             # traing step
    #             train = tf.train.GradientDescentOptimizer(LR).minimize(cost)
    #             _, costp = sess.run([train, cost], feed_dict={
    #                                 images: batch2, train_mode: True})

    #             # test classification again, should have a higher probability about tiger
    #             prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    #             logEntry(prob.shape)
    #             logEntry(utils.print_prob(prob[0], './synset.txt'))
    # test save
    # vgg.save_npy(sess, str('/flush1/raj034/testX.npy'))
