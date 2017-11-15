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


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


model = "testP12/o"


main_dir = "/flush1/raj034/vgg19/" + model + "/"
LOG_FILE = open(main_dir + 'log2.txt', 'a')

logEntry("started")
train_vect = np.load(main_dir + "train_vectors.npy")
train_label = np.load(main_dir + "train_label.npy")

logEntry(train_vect.shape)
test_vectors = np.load(main_dir + "test_vectors.npy")
test_label = np.load(main_dir + "test_label.npy")
logEntry(test_vectors.shape)

logEntry("loaded")


M = 0.5
count1 = 0
count2 = 0
total = 0
for i in range(1250):

    l2 = np.sum((train_vect - test_vectors[i]) ** 2, axis=1)
    # logEntry(l2[np.where(l2 < M)].shape)

    a = np.argsort(l2)
    b = train_label[a]
    # logEntry(b)
    if(b[0][1] == test_label[i][1]):
        count1 = count1 + 1

    c = b[0:10, 1]
    d = c.astype(int)

    if(range(27)[np.argmax(np.bincount(d))] == int(test_label[i][1])):
        count2 = count2 + 1

    total = total + 1

    # logEntry(str(i) + "   top1 = " + str(1.0 * count1 / total * 100) + " \t\t" + str(l2[np.where(l2 < M)].shape))
    # logEntry(str(d) + "\t " + str(test_label[i]) + str(range(26)[np.argmax(np.bincount(d))]))
    # logEntry(str(i) + "   top10 = " + str(1.0 * count2 / total * 100) + " \t\t" + str(l2[np.where(l2 < M)].shape))
    logEntry(str(i) + "   top100 = " + str(1.0 * count2 / total * 100))
