# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
import random
import numpy as np
import scipy.misc
import scipy.io
import math
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics
import seaborn as sns
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import scipy.io
import pickle

# from matplotlib.mlab import PCA

test_case = "style_test1"
model = "test1/0"

main_dir = "/flush1/raj034/vgg19/" + model + "/"
LOG_FILE = open(main_dir + 'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


def pca2(data, pc_count=None):
    return PCA(n_components=2000).fit(data).transform(data)


def main():

    train_vect = np.load(main_dir + "train_vectors.npy")
    train_label = np.load(main_dir + "train_label.npy")

    pca = PCA(n_components=2000)

    results = pca.fit_transform(train_vect)

    logEntry(results.shape)
    logEntry(results[0].shape)
    np.save(main_dir + "train_pca.npy", results)
    with open(main_dir + 'pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    ################################################################
    with open(main_dir + 'pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    var = pca.explained_variance_ratio_

    np.save(main_dir + "var_test.npy", var)
    plt.plot(np.array(var))
    plt.savefig(main_dir + "var.png")

    test_vectors = np.load(main_dir + "test_vectors.npy")
    test_label = np.load(main_dir + "test_label.npy")

    results_test = pca.transform(test_vectors)

    logEntry(results_test.shape)
    logEntry(results_test[0].shape)
    np.save(main_dir + "test_pca.npy", results_test)

if __name__ == '__main__':
    main()
