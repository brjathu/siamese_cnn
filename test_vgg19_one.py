import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math
from operator import itemgetter

tf.logging.set_verbosity(tf.logging.ERROR)

# os.system("rm log.txt")


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


batch_size = 50
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
test_case = "style_test1"
model = "testP4/3"
g = tf.Graph()


train_data = np.load("../siamese_cnn/data/train_data.npy", encoding='latin1')
test_data = np.load("../siamese_cnn/data/val_data.npy", encoding='latin1')
pre_load = np.load("../siamese_cnn/pre_load_images_train.npy", encoding='latin1').item()
pre_load_val = np.load("../siamese_cnn/pre_load_images_val.npy", encoding='latin1').item()


train_vectors = []
test_vectors = []

train_label = []
test_label = []
with g.as_default(), g.device('/cpu'), tf.Session() as sess:
    images = tf.placeholder("float", [batch_size, 224, 224, 3])

    # feed_dict = {images: batch}
    vgg = vgg19.Vgg19("/flush1/raj034/vgg19/" + model + "/vgg19_1.npy")
    main_dir = "/flush1/raj034/vgg19/" + model + "/"
    LOG_FILE = open(main_dir + 'log.txt', 'a')

    with tf.name_scope("content_vgg"):
        vgg.build(images)
    for l in class_list:
        logEntry(str(l) + "traing samples")
        class_num = l
        # location = os.listdir("/flush1/raj034/WIKI_STYLE/" + str(class_num) + "/img/")
        # os.system("mkdir /flush1/raj034/WIKI_STYLE/" + str(class_num) + "/features/" + test_case)
        count = 0
        for b in range(int(train_data.shape[1] / batch_size)):
            img_name = train_data[l, b * batch_size: (b + 1) * batch_size]
            # logEntry(img_name)
            batch = itemgetter(*img_name)(pre_load)
            batch = np.array(batch)

            feat = sess.run(vgg.conv5_1, feed_dict={images: batch})
            # feat = model.predict(img)
            # logEntry(feat.shape)

            count = 0
            for i in img_name:
                features = np.reshape(feat[count], (-1, 512))
                gram = np.matmul(features.T, features) / features.size
                gram = np.array(gram[np.triu_indices(512)])
                train_vectors.append(gram)
                train_label.append([i, l])
                count = count + 1
        logEntry(np.array(train_vectors).shape)

    np.save(main_dir + "train_vectors.npy", np.array(train_vectors))
    np.save(main_dir + "train_label.npy", np.array(train_label))

    batch_size = 50

    for l in class_list:
        class_num = l
        logEntry(str(l) + "testing samples")

        count = 0
        for b in range(int(test_data.shape[1] / batch_size)):
            img_name = test_data[l, b * batch_size: (b + 1) * batch_size]
            # logEntry(img_name)
            batch = itemgetter(*img_name)(pre_load_val)
            batch = np.array(batch)

            feat = sess.run(vgg.conv5_1, feed_dict={images: batch})
            # feat = model.predict(img)
            # logEntry(feat.shape)

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
