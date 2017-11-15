import tensorflow as tf
import vgg19_trainable as vgg19
import utils
import numpy as np
import os
import itertools
import random
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics
import seaborn as sns
from operator import itemgetter
from test_vgg19_batch import validation_testing

tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)


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

# parameters
M = 0.5  # sqrt(0.2)
EPOCH = 300
LR = 1e-4
# train positive samples
batch_size = 32

# model = "testP16"  # LR=1e-6, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum , 10000, layer 5_3
model = "testP15"  # LR=1e-4, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum , 10000, vgg training from the begining
# model = "testP14"  # LR=1e-6, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum , 10000, vgg training from the begining
# model = "testP13"  # LR=1e-6, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum , all
# model = "testP12"  # LR=1e-6, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum , 25000
# model = "testP11"  # LR=1e-6, M=1, BS=32, neg_loss=tf.maximum, cost=reduce_sum
# model = "testP10"  # LR=1e-5, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum
# model = "testP9"  # LR=1e-5, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum, optimizer = gradeientdesent
# model = "testP8"  # LR=1e-6, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum
# model = "testP7"  # LR=1e-7, M=0.5, BS=32, neg_loss=tf.maximum, cost=reduce_sum
main_dir = "/flush1/raj034/vgg19/" + model + "/"
os.system("mkdir " + main_dir)
LOG_FILE = open(main_dir + 'log.txt', 'a')

train_data = np.load("data/new_train_data.npy", encoding='latin1')
logEntry("debugger a ========= >  " + str(train_data.shape))
train_data = train_data[0:10000, :]
logEntry("debugger b ========= >  " + str(train_data.shape))

val_data = np.load("data/new_val_data.npy", encoding='latin1')
logEntry("debugger c ========= >  " + str(val_data.shape))
val_data = val_data[0:1000, :]
# logEntry("debugger d ========= >  " + str(val_data.shape))

test_data = np.load("data/final_test.npy", encoding='latin1')

# get all the images at first to reduce network traffic
pre_load = np.load("pre_load_images_train.npy", encoding='latin1')
pre_load_images = pre_load.item()

pre_load = np.load("pre_load_images_val.npy", encoding='latin1')
pre_load_images_val = pre_load.item()


class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


# create tf models
images1 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images1")
images2 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images2")

pos = tf.placeholder(tf.float32, [batch_size, 1], name="positive_samples")
# neg = tf.placeholder(tf.float32, [batch_size, 1], name="negetive_samples")

train_mode = tf.placeholder(tf.bool, name="train_mode")

# vgg1 = vgg19.Vgg19("vgg19.npy")
vgg1 = vgg19.Vgg19()
vgg1.build(images1, train_mode)

# vgg2 = vgg19.Vgg19("vgg19.npy")
vgg2 = vgg19.Vgg19()
vgg2.build(images2, train_mode)

features1 = tf.reshape(vgg1.conv5_1, [batch_size, -1, 512], name="features1")
features2 = tf.reshape(vgg2.conv5_1, [batch_size, -1, 512], name="features2")

neg = (pos - 1) * (-1)
gram1 = tf.matmul(tf.transpose(features1, perm=[0, 2, 1]), features1, name="gram1") / (512 * 196) / (512 * 196)
gram2 = tf.matmul(tf.transpose(features2, perm=[0, 2, 1]), features2, name="gram2") / (512 * 196) / (512 * 196)
loss = tf.reshape(tf.reduce_sum((gram1 - gram2), axis=[1, 2]) ** 2, (batch_size, 1), name="loss")
pos_loss = tf.multiply(tf.reshape(loss, (batch_size, 1)), pos, name="pos_loss")
# neg_loss = tf.multiply(tf.where(tf.less(loss, M), M - loss, tf.zeros_like(loss)), neg, name="neg_loss")
neg_loss = tf.multiply(tf.maximum(0.0, M - loss), neg, name="neg_loss")
cost = tf.reduce_sum(pos_loss + neg_loss)
# cost = pos_loss + neg_loss

# traing step
train = tf.train.AdamOptimizer(LR, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)
# train = tf.train.GradientDescentOptimizer(LR).minimize(cost)

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

        total_min_batch = int(train_data.shape[0] / batch_size)
        # training
        for b in range(total_min_batch):
            start_time = time.time()
            train_batch_data = train_data[b * batch_size: (b + 1) * batch_size]
            y_pos = np.reshape(train_batch_data[:, 4], (batch_size, 1))

            batch1 = np.reshape(itemgetter(*train_batch_data[:, 0])(pre_load_images), (batch_size, 224, 224, 3))
            # batch1 = itemgetter(*train_batch_data[:, 0])(pre_load_images)
            batch2 = np.reshape(itemgetter(*train_batch_data[:, 1])(pre_load_images), (batch_size, 224, 224, 3))
            # batch2 = itemgetter(*train_batch_data[:, 1])(pre_load_images)

            _, costp = sess.run([train, cost], feed_dict={images1: batch1, images2: batch2, pos: y_pos, train_mode: True})
            # costp = sess.run(cost, feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg, train_mode: False})
            logEntry("debugger 3 ========= >  " + "epoch = " + str(epoch) + "  mini =" + str(b) + "/" + str(total_min_batch) + "\tloss = " + str(costp))
            duration = time.time() - start_time
            logEntry("time = " + str(duration))

            loss_avg += np.sum(costp)

        # validation
        # total_val_min_batch = int(val_data.shape[0] / batch_size)
        # for b in range(total_val_min_batch):
        #     val_batch_data = val_data[b * batch_size: (b + 1) * batch_size]
        #     # logEntry("#validation debugger 2 ========= >  " + str(b) + "batch of" + str(val_batch_data))

        #     y_pos = np.reshape(val_batch_data[:, 4], (batch_size, 1))

        #     batch1 = np.reshape(itemgetter(*val_batch_data[:, 0])(pre_load_images_val), (batch_size, 224, 224, 3))
        #     batch2 = np.reshape(itemgetter(*val_batch_data[:, 1])(pre_load_images_val), (batch_size, 224, 224, 3))

        #     lossp = sess.run(loss, feed_dict={images1: batch1, images2: batch2, pos: y_pos, train_mode: False})
        #     logEntry("debugger 4 ========= >  " + "epoch = " + str(epoch) + "  mini =" + str(b) + "/" + str(total_val_min_batch))

        #     # for the distribution graphs
        #     distibution = np.vstack([distibution, np.hstack([lossp, y_pos])])

        loss_graph.append(loss_avg)
        np.save(main_dir + "loss.npy", loss_graph)
        plt.figure(0)
        plt.plot(np.array(loss_graph))
        plt.savefig(main_dir + "loss_graph.png")

        os.system("mkdir " + main_dir + str(epoch))
        vgg1.save_npy(sess, main_dir + str(epoch) + "/vgg19_1.npy")
        vgg2.save_npy(sess, main_dir + str(epoch) + "/vgg19_2.npy")

        val = validation_testing(model + "/" + str(epoch))
        val_graph.append(val)
        np.save(main_dir + "val.npy", val_graph)
        plt.figure(1)
        plt.plot(np.array(val_graph))
        plt.savefig(main_dir + "val_graph.png")

        #np.save(main_dir + str(epoch) + "/distibution.npy", distibution)
        #plotDistribution(distibution, 101, val_data.shape[0])

        plt.show()
