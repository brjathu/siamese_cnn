import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math
from operator import itemgetter
from sklearn.decomposition import PCA
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import metrics
import seaborn as sns
import matplotlib.image as mpimg
from sklearn.svm import SVC

tf.logging.set_verbosity(tf.logging.ERROR)


def validation_testing(model="testP4/3"):

    def logEntry(TMP_STRING):
        LOG_FILE.write(str(TMP_STRING))
        LOG_FILE.write("\n")
        LOG_FILE.flush()
        print(str(TMP_STRING))

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

    # model = "testP4/3"
    batch_size = 50
    class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    g = tf.Graph()
    main_dir = "/flush1/raj034/vgg19/" + model + "/"

    train_data = np.load("../siamese_cnn/data/train_data.npy", encoding='latin1')
    test_data = np.load("../siamese_cnn/data/val_data.npy", encoding='latin1')
    pre_load = np.load("../siamese_cnn/pre_load_images_train.npy", encoding='latin1').item()
    pre_load_val = np.load("../siamese_cnn/pre_load_images_val.npy", encoding='latin1').item()

    train_vectors = []
    test_vectors = []

    train_label = []
    test_label = []
    with g.as_default(), g.device('/gpu'), tf.Session() as sess:
        images = tf.placeholder("float", [batch_size, 224, 224, 3])

        # feed_dict = {images: batch}
        vgg = vgg19.Vgg19("/flush1/raj034/vgg19/" + model + "/vgg19_1.npy")

        LOG_FILE = open(main_dir + 'log.txt', 'a')

        with tf.name_scope("content_vgg"):
            vgg.build(images)
        for l in class_list:
            logEntry(str(l) + "traing samples")
            count = 0
            for b in range(int(train_data.shape[1] / batch_size)):
                img_name = train_data[l, b * batch_size: (b + 1) * batch_size]

                batch = itemgetter(*img_name)(pre_load)
                batch = np.array(batch)

                feat = sess.run(vgg.conv5_1, feed_dict={images: batch})
                count = 0
                for i in img_name:
                    features = np.reshape(feat[count], (-1, 512))
                    gram = np.matmul(features.T, features) / (512 * 196) / (512 * 196)
                    gram = np.array(gram[np.triu_indices(512)])
                    train_vectors.append(gram)
                    train_label.append([i, l])
                    count = count + 1
            logEntry(np.array(train_vectors).shape)

        np.save(main_dir + "train_vectors.npy", np.array(train_vectors))
        np.save(main_dir + "train_label.npy", np.array(train_label))

        for l in class_list:
            logEntry(str(l) + "testing samples")

            count = 0
            for b in range(int(test_data.shape[1] / batch_size)):
                img_name = test_data[l, b * batch_size: (b + 1) * batch_size]
                batch = itemgetter(*img_name)(pre_load_val)
                batch = np.array(batch)

                feat = sess.run(vgg.conv5_1, feed_dict={images: batch})
                count = 0
                for i in img_name:
                    features = np.reshape(feat[count], (-1, 512))
                    gram = np.matmul(features.T, features) / features.size
                    gram = np.array(gram[np.triu_indices(512)])
                    test_vectors.append(gram)
                    test_label.append([i, l])
                    count = count + 1
            logEntry(np.array(test_vectors).shape)
        np.save(main_dir + "test_vectors.npy", np.array(test_vectors))
        np.save(main_dir + "test_label.npy", np.array(test_label))
        sess.close()

    # PCA

    train_vect = np.load(main_dir + "train_vectors.npy")
    train_label = np.load(main_dir + "train_label.npy")

    pca = PCA(n_components=4096)

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

    # SVM
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
    #         SVM for testing samples
    #
    ########################################################

    with open(main_dir + 'svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_vect = np.load(main_dir + "test_pca.npy")
    test_label = np.load(main_dir + "test_label.npy")
    prediction = clf.predict(test_vect)
    accuracy = plotConfusionMatrix(prediction, test_label[:, 1], main_dir + "confusion_matrix", "wiki art classification")
    return accuracy


a = validation_testing("testP12/o")
