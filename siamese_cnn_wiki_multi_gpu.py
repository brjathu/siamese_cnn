import tensorflow as tf
import vgg19_trainable_mini as vgg19
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
M = 200  # sqrt(0.2)
EPOCH = 100
LR = 1e-5
model = "testX3"
batch_size = 80
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
num_gpu = 3


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


def plotDistribution(loss, bins=51, y=500):
    pos = loss[np.where(loss[:, 1] == 1)[0]]
    pos = pos[:, 0]

    neg = loss[np.where(loss[:, 1] == 0)[0]]
    neg = neg[:, 0]

    plt.figure(1)

    plt.subplot(1, 2, 1)

    plt.hist(pos, bins=np.linspace(0, 400, bins), facecolor='green', alpha=0.5)
    plt.title("positive loss")
    plt.axis([0, 400, 0, y])

    plt.subplot(1, 2, 2)

    plt.hist(neg, bins=np.linspace(0, 400, bins), facecolor='green', alpha=0.5)
    plt.title("negative loss")
    plt.axis([0, 400, 0, y])
    plt.savefig(main_dir + str(epoch) + "/distribution" + str(epoch) + ".png")
    plt.close()


def validation_accuracy(val_data):
    pos = val_data[np.where(val_data[:, 1] == 1)[0]]
    pos = pos[:, 0]
    p = len(np.where(pos < M)[0])

    neg = val_data[np.where(val_data[:, 1] == 0)[0]]
    neg = neg[:, 0]
    n = len(np.where(neg > M)[0])

    val_acc = (p + n) * 1.0 / (val_data.shape[0])
    return val_acc


main_dir = "/flush1/raj034/vgg19/" + model + "/"
os.system("mkdir " + main_dir)
LOG_FILE = open(main_dir + 'log.txt', 'a')

train_data = np.load("data/final_train.npy")
# logEntry("debugger a ========= >  " + str(train_data.shape))
train_data = train_data[0:800, :]
logEntry("debugger b ========= >  " + str(train_data.shape))

val_data = np.load("data/final_val.npy")
# logEntry("debugger c ========= >  " + str(val_data.shape))
val_data = val_data[0:160, :]
logEntry("debugger d ========= >  " + str(val_data.shape))

test_data = np.load("data/final_test.npy")

##################################################################################
# Tensorflow model
##################################################################################

images1 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images1")
images2 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images2")

pos = tf.placeholder(tf.float32, [batch_size, 1], name="positive_samples")
neg = tf.placeholder(tf.float32, [batch_size, 1], name="negetive_samples")

train_mode = tf.placeholder(tf.bool, name="train_mode")

vgg1 = vgg19.Vgg19('vgg19.npy')
vgg1.build(images1, train_mode)

vgg2 = vgg19.Vgg19('vgg19.npy')
vgg2.build(images2, train_mode)

features1 = tf.reshape(vgg1.conv3_1, [batch_size, -1, 256], name="features1")
features2 = tf.reshape(vgg2.conv3_1, [batch_size, -1, 256], name="features2")

gram1 = tf.matmul(tf.transpose(features1, perm=[0, 2, 1]), features1, name="gram1") / (256 * 196) / (256 * 196)
gram2 = tf.matmul(tf.transpose(features2, perm=[0, 2, 1]), features2, name="gram2") / (256 * 196) / (256 * 196)
loss = tf.reshape(tf.reduce_sum((gram1 - gram2), axis=[1, 2]) ** 2, (batch_size, 1), name="loss")
pos_loss = tf.multiply(tf.reshape(loss, (batch_size, 1)), pos, name="pos_loss")
neg_loss = tf.multiply(tf.where(tf.less(loss, M), M - loss, tf.zeros_like(loss)), neg, name="neg_loss")
cost = tf.reduce_sum(pos_loss + neg_loss)

# traing step
train = tf.train.AdamOptimizer(LR).minimize(tf.reduce_mean(cost), colocate_gradients_with_ops=True)

loss_graph = []
val_graph = []

with tf.device('/gpu'), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        logEntry("debugger 0 ========= >  " + str(epoch) + "   ========   epoch number ")

        loss_avg = 0
        val_avg = 0
        distibution = np.array((batch_size, 2))
        validation = np.array((batch_size, 2))

        # training
        for b in range(int(train_data.shape[0] / batch_size)):
            logEntry("debugger 1 ========= >  " + str(b) + "batch of" + str(int(train_data.shape[0] / batch_size)))
            train_batch_data = train_data[b * batch_size: (b + 1) * batch_size]
            y_pos = np.reshape(train_batch_data[:, 3], (batch_size, 1))
            y_neg = (y_pos - 1) * (-1)
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

            _, costp = sess.run([train, cost], feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: True})
            logEntry("debugger 3 ========= >  " + str(costp))

            loss_avg += np.sum(costp)

        # validation
        for b in range(int(val_data.shape[0] / batch_size)):
            logEntry("#validation debugger 1 ========= >  " + str(b) + "batch of" + str(int(val_data.shape[0] / batch_size)))
            val_batch_data = val_data[b * batch_size: (b + 1) * batch_size]
            # logEntry("#validation debugger 2 ========= >  " + str(b) + "batch of" + str(val_batch_data))

            y_pos = np.reshape(val_batch_data[:, 3], (batch_size, 1))
            y_neg = (y_pos - 1) * (-1)  # change
            count = 0
            for i in range(batch_size):
                img1 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + val_batch_data[i, 0][0])
                img2 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + val_batch_data[i, 0][1])

                if count == 0:
                    batch1 = img1.reshape((1, 224, 224, 3))
                    batch2 = img2.reshape((1, 224, 224, 3))
                else:
                    batch1 = np.concatenate((batch1, img1.reshape((1, 224, 224, 3))), 0)
                    batch2 = np.concatenate((batch2, img2.reshape((1, 224, 224, 3))), 0)

                count += 1
            lossp = sess.run(loss, feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: False})
            logEntry("#validation debugger 3 ========= >  " + str(b) + "batch of" + str(lossp.shape))

            # for the distribution graphs
            distibution = np.vstack([distibution, np.hstack([lossp, y_pos])])

        loss_graph.append(loss_avg)
        np.save(main_dir + "loss.npy", loss_graph)
        plt.figure(0)
        plt.plot(np.array(loss_graph))
        plt.savefig(main_dir + "loss_graph.png")

        val_graph.append(validation_accuracy(distibution))
        np.save(main_dir + "val.npy", val_graph)
        plt.figure(1)
        plt.plot(np.array(val_graph))
        plt.savefig(main_dir + "val_graph.png")

        os.system("mkdir " + main_dir + str(epoch))
        vgg1.save_npy(sess, main_dir + str(epoch) + "/vgg19_1.npy")
        vgg2.save_npy(sess, main_dir + str(epoch) + "/vgg19_2.npy")

        np.save(main_dir + str(epoch) + "/distibution.npy", distibution)
        plotDistribution(distibution, 101, val_data.shape[0])

        # plt.show()
