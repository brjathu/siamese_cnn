# Copyright (c) 2017
#
# Jathushan.Rajasegaran@gmail.com
import os
import tensorflow as tf
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)


# parameters
LR = 0.0001
EPOCH = 300
BATCH_SIZE = 100
MODEL = 5
RESULTS_PATH = "./model/" + str(MODEL) + "/"
LOG_FILE = open(RESULTS_PATH + 'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(TMP_STRING)
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


tf.reset_default_graph()

# tf placeholder
# value in the range of (0, 1)
tf_x = tf.placeholder(tf.float32, [None, 2500])
tf_y = tf.placeholder(tf.float32, [None, 25])


# # model 3
# hidden_layer1 = tf.layers.dense(tf_x, 400, tf.nn.tanh, name="hidden_layer1")
# hidden_layer2 = tf.layers.dense(hidden_layer1, 300, tf.nn.tanh, name="hidden_layer2")
# hidden_layer3 = tf.layers.dense(hidden_layer2, 200, tf.nn.tanh, name="hidden_layer3")
# hidden_layer4 = tf.layers.dense(hidden_layer3, 100, tf.nn.tanh, name="hidden_layer4")
# hidden_layer5 = tf.layers.dense(hidden_layer4, 60, tf.nn.tanh, name="hidden_layer5")
# hidden_layer6 = tf.layers.dense(hidden_layer5, 30, tf.nn.tanh, name="hidden_layer6")
# hidden_layer7 = tf.layers.dense(hidden_layer6, 10, tf.nn.tanh, name="hidden_layer7")
# hidden_layer8 = tf.layers.dense(hidden_layer7, 5, tf.nn.tanh, name="hidden_layer9")

# model 4
hidden_layer1 = tf.layers.dense(tf_x, 2000, tf.nn.tanh, name="hidden_layer1")
hidden_layer2 = tf.layers.dense(hidden_layer1, 1500, tf.nn.tanh, name="hidden_layer2")
hidden_layer3 = tf.layers.dense(hidden_layer2, 1024, tf.nn.tanh, name="hidden_layer3")
hidden_layer4 = tf.layers.dense(hidden_layer3, 512, tf.nn.tanh, name="hidden_layer4")
hidden_layer5 = tf.layers.dense(hidden_layer4, 256, tf.nn.tanh, name="hidden_layer5")
hidden_layer6 = tf.layers.dense(hidden_layer5, 128, tf.nn.tanh, name="hidden_layer6")
hidden_layer7 = tf.layers.dense(hidden_layer6, 64, tf.nn.tanh, name="hidden_layer7")
hidden_layer8 = tf.layers.dense(hidden_layer7, 25, tf.nn.tanh, name="hidden_layer8")
# hidden_layer9 = tf.layers.dense(hidden_layer8, 5, tf.nn.tanh, name="hidden_layer9")

softmax_layer = tf.nn.softmax(hidden_layer8, name="softmax")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax_layer, labels=tf_y))
# loss = tf.losses.mean_squared_error(labels=tf_y, predictions=softmax_layer)
train = tf.train.AdamOptimizer(LR).minimize(loss)


x_train = np.load("./data/train_pca.npy")
y_train = np.load("./data/train_label.npy")

x_test = np.load("./data/test_pca.npy")
y_test = np.load("./data/test_label.npy")


train_data = np.hstack([x_train, y_train])
test_data = np.hstack([x_test, y_test])

print(x_train.shape)
print(y_train.shape)
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

train2 = np.random.shuffle(train_data)
test2 = np.random.shuffle(test_data)

i = 0


loss_graph = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH):
        loss_avg = 0
        for i in range(int(x_train.shape[0] / BATCH_SIZE)):

            one_hot = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 2501]

            one = np.zeros((BATCH_SIZE, 25))
            count = 0
            for j in one_hot:
                # print(class_list.index(int(j)))
                one[count, class_list.index(int(j))] = 1
                count = count + 1
            _, loss_val = sess.run([train, loss], feed_dict={tf_x: train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 0:2500], tf_y: one})
            loss_avg += loss_val
        # print(loss_avg / i)
        # loss_graph.append(loss_avg / i)

        # test
        acc = 0
        for i in range(int(x_test.shape[0] / BATCH_SIZE)):

            one_hot = test_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 2501]

            one = np.zeros((BATCH_SIZE, 25))
            count = 0
            for j in one_hot:
                # print(class_list.index(int(j)))
                one[count, class_list.index(int(j))] = 1
                count = count + 1

            sm = sess.run(softmax_layer, feed_dict={tf_x: test_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 0:2500], tf_y: one})
            count = 0
            for s in sm:
                # print(class_list[np.argmax(s)], one_hot[count], int(class_list[np.argmax(s)]) == int(one_hot[count]))
                if(int(class_list[np.argmax(s)]) == int(one_hot[count])):
                    acc = acc + 1
                count = count + 1
        print(acc * 1.0 / (int(x_test.shape[0] / BATCH_SIZE) * BATCH_SIZE) * 100)
