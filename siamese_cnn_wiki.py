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


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


# parameters
M = 100  # sqrt(0.2)
EPOCH = 50
LR = 1e-7


model = "test3"
main_dir = "/flush1/raj034/vgg19/" + model + "/"
os.system("mkdir " + main_dir)
LOG_FILE = open(main_dir + 'log.txt', 'a')

train_data = np.load("data/final_train.npy")
logEntry("debugger a ========= >  " + str(train_data.shape))
train_data = train_data[0:800, :]
logEntry("debugger b ========= >  " + str(train_data.shape))
val_data = np.load("data/final_val.npy")
test_data = np.load("data/final_test.npy")

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


# train positive samples
batch_size = 80


# create tf models
images1 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images1")
images2 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images2")

pos = tf.placeholder(tf.float32, [batch_size, 1], name="positive_samples")
neg = tf.placeholder(tf.float32, [batch_size, 1], name="negetive_samples")

train_mode = tf.placeholder(tf.bool, name="train_mode")

# vgg1 = vgg19.Vgg19('/flush1/raj034/vgg19/test2/25/vgg19_1.npy')
vgg1 = vgg19.Vgg19('vgg19.npy')
vgg1.build(images1, train_mode)

# vgg2 = vgg19.Vgg19('/flush1/raj034/vgg19/test2/25/vgg19_2.npy')
vgg2 = vgg19.Vgg19('vgg19.npy')
vgg2.build(images2, train_mode)

features1 = tf.reshape(vgg1.conv5_1, [batch_size, -1, 512], name="features1")
features2 = tf.reshape(vgg2.conv5_1, [batch_size, -1, 512], name="features2")

gram1 = tf.matmul(tf.transpose(features1, perm=[0, 2, 1]), features1, name="gram1") / (512 * 196) / (512 * 196)
gram2 = tf.matmul(tf.transpose(features2, perm=[0, 2, 1]), features2, name="gram2") / (512 * 196) / (512 * 196)
loss = tf.reshape(tf.reduce_sum((gram1 - gram2), axis=[1, 2], name="loss") ** 2, (batch_size, 1))
pos_loss = tf.multiply(tf.reshape(loss, (batch_size, 1)), pos, name="pos_loss")
neg_loss = tf.multiply(tf.where(tf.less(loss, M), M - loss, tf.zeros_like(loss)), neg, name="neg_loss")
# neg_loss = tf.multiply(tf.reshape(loss, (batch_size, 1)), neg, name="neg_loss")
# cost = loss
cost = tf.reduce_sum(pos_loss + neg_loss)

# traing step
train = tf.train.GradientDescentOptimizer(LR).minimize(cost)

distibution = np.array((batch_size, 2))
loss = 0
loss_graph = []
with tf.device('/gpu'), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        logEntry("debugger 0 ========= >  " + str(epoch) + "   ========   epoch number ")

        loss_avg = 0

        for b in range(int(train_data.shape[0] / batch_size)):
            logEntry("debugger 1 ========= >  " + str(b) + "batch of" + str(int(train_data.shape[0] / batch_size)))

            train_batch_data = train_data[b * batch_size: (b + 1) * batch_size]

            y_pos = np.reshape(train_batch_data[:, 3], (batch_size, 1))
            y_neg = (y_pos - 1) * (-1)  # change
            # logEntry("debugger 0a ========= >  " + str(y_pos))
            # logEntry("debugger 0b ========= >  " + str(y_neg))

            count = 0
            for i in range(batch_size):
                # logEntry("debugger 1 ========= >  " + str(train_batch_data[i, 0][0]) + "\t\t" + str(train_batch_data[i, 0][1]) +
                #          "\t" + str(train_batch_data[i, 1]) + "\t" + str(train_batch_data[i, 2]) + "\t" + str(train_batch_data[i, 3]))

                img1 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + train_batch_data[i, 0][0])
                img2 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + train_batch_data[i, 0][1])

                if count == 0:
                    batch1 = img1.reshape((1, 224, 224, 3))
                    batch2 = img2.reshape((1, 224, 224, 3))
                else:
                    batch1 = np.concatenate((batch1, img1.reshape((1, 224, 224, 3))), 0)
                    batch2 = np.concatenate((batch2, img2.reshape((1, 224, 224, 3))), 0)

                count += 1
                # logEntry("debugger 2 ========= >  " + str(count) + "    " + str(batch1.shape))

            _, costp = sess.run([train, cost], feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: True})
            # costp = sess.run(cost, feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: False})
            logEntry("debugger 3 ========= >  " + str(costp))

            # # code for the distribution graphs
            # logEntry("debugger 3 ========= >  " + str(np.hstack([costp, y_pos])))
            # distibution = np.vstack([distibution, np.hstack([costp, y_pos])])
            # logEntry("debugger 4 ========= >  " + str(distibution.shape))
            # np.save(main_dir + "distibution25.npy", distibution)

            # code for validation
            # # test classification again, should have a higher probability about tiger
            # prob = sess.run(vgg1.prob, feed_dict={images1: batch1, train_mode: False})
            # logEntry(str(class_style_1) + "      " + str(utils.print_prob(prob[0], './synset.txt')))
            # prob = sess.run(vgg2.prob, feed_dict={images2: batch1, train_mode: False})
            # logEntry(str(class_style_1) + "      " + str(utils.print_prob(prob[0], './synset.txt')))
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
