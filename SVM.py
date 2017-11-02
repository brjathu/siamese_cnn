from sklearn import svm
import numpy as np
import os
import scipy.misc
import scipy.io
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle


# preprocessing

model = "testX1/0"

main_dir = "/flush1/raj034/vgg19/" + model + "/"
LOG_FILE = open(main_dir + 'log.txt', 'a')

train = True


def plotConfusionMatrix(predictions, y, FILENAME, TITLE):
    LABELS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    confusion_matrix = metrics.confusion_matrix(predictions, y)
    accuracy = metrics.accuracy_score(predictions, y)
    logEntry("accuracy ==> " + str(accuracy))
    fig = plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix: " + TITLE)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(FILENAME)
    plt.clf()
    plt.close('all')

    return accuracy


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


if (train == True):
    train_vect = np.load(main_dir + "train_pca.npy")
    train_label = np.load(main_dir + "train_label.npy")
    clf = SVC(kernel='linear', gamma=1, decision_function_shape="ovr")
    # clf = LinearDiscriminantAnalysis(n_components=20)
    clf.fit(train_vect, train_label[:, 1])

    with open(main_dir + 'svm.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("=======DONE========")

    ########################################################
    #
    #         PCA for testing samples
    #
    ########################################################

    with open(main_dir + 'svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_vect = np.load(main_dir + "test_pca.npy")
    test_label = np.load(main_dir + "test_label.npy")
    prediction = clf.predict(test_vect)
    plotConfusionMatrix(prediction, test_label[:, 1], main_dir + "confusion_matrix", "wiki art classification")
