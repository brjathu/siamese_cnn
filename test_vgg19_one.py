import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math


tf.logging.set_verbosity(tf.logging.ERROR)

# os.system("rm log.txt")


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


batch_size = 1
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
test_case = "style_test1"
model = "test1/0"
g = tf.Graph()

train_vectors = []
test_vectors = []

train_label = []
test_label = []
with g.as_default(), g.device('/cpu'), tf.Session() as sess:
    images = tf.placeholder("float", [batch_size, 224, 224, 3])
    # feed_dict = {images: batch}
    vgg = vgg19.Vgg19("/flush1/raj034/vgg19/" + model + "/testX.npy")
    main_dir = "/flush1/raj034/vgg19/" + model + "/"
    LOG_FILE = open(main_dir + 'log.txt', 'a')

    with tf.name_scope("content_vgg"):
        vgg.build(images)
    for l in class_list:
        logEntry(str(l) + "traing samples")
        class_num = l
        location = os.listdir("/flush1/raj034/WIKI_STYLE/" + str(class_num) + "/img/")
        # os.system("mkdir /flush1/raj034/WIKI_STYLE/" + str(class_num) + "/features/" + test_case)
        count = 0
        for image in location:

            if(count % batch_size == 0):
                img1 = utils.load_image("/flush1/raj034/WIKI_STYLE/" + str(class_num) + "/img/" + image)
                batch = img1.reshape((1, 224, 224, 3))
            else:
                img1 = utils.load_image("/flush1/raj034/WIKI_STYLE/" + str(class_num) + "/img/" + image)
                batch1 = img1.reshape((1, 224, 224, 3))
                batch = np.concatenate((batch, batch1), 0)

            if ((count + 1) % batch_size == 0):
                # fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
                conv5_1 = sess.run(vgg.conv5_1, feed_dict={images: batch})
                for i in range(batch_size):
                    features = np.reshape(conv5_1[i], (-1, 512))
                    gram = np.matmul(features.T, features) / features.size
                    gram = np.array(gram[np.triu_indices(512)])
                    train_vectors.append(gram)
                    train_label.append([image, l])
                    # scipy.io.savemat("/flush1/raj034/WIKI_STYLE/" + str(class_num) + "/features/" + test_case + "/" +
                    #                  name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'gram': gram}, oned_as='row')
                    # scipy.io.savemat("../style/WIKI_STYLE/" + str(class_num) + "/features/style/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'conv5_1': gram}, oned_as='row')
                    # scipy.io.savemat('features/style/'+image[0:-4]+'.mat',mdict={'conv1_1': gram1,'conv2_1':gram2,'conv3_1':gram3,'conv4_1':gram4,'conv5_1':gram5},oned_as='row')
            count = count + 1

    np.save(main_dir + "train_vectors.npy", np.array(train_vectors))
    np.save(main_dir + "train_label.npy", np.array(train_label))

    for l in class_list:
        class_num = l
        location = os.listdir("/flush1/raj034/WIKI_STYLE_TEST/" + str(class_num) + "/img/")
        os.system("mkdir /flush1/raj034/WIKI_STYLE_TEST/" + str(class_num) + "/features/" + str(test_case) + "/")
        logEntry(str(l) + "testing samples")

        count = 0
        for image in location:

            if(count % batch_size == 0):
                img1 = utils.load_image("/flush1/raj034/WIKI_STYLE_TEST/" + str(class_num) + "/img/" + image)
                batch = img1.reshape((1, 224, 224, 3))
            else:
                img1 = utils.load_image("/flush1/raj034/WIKI_STYLE_TEST/" + str(class_num) + "/img/" + image)
                batch1 = img1.reshape((1, 224, 224, 3))
                batch = np.concatenate((batch, batch1), 0)

            if ((count + 1) % batch_size == 0):
                # fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
                conv5_1 = sess.run(vgg.conv5_1, feed_dict={images: batch})
                for i in range(batch_size):
                    features = np.reshape(conv5_1[i], (-1, 512))
                    gram = np.matmul(features.T, features) / features.size
                    gram = np.array(gram[np.triu_indices(512)])
                    test_vectors.append(gram)
                    test_label.append([image, l])
                    # scipy.io.savemat("/flush1/raj034/WIKI_STYLE_TEST/" + str(class_num) + "/features/" + test_case + "/" +
                    #                  name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'gram': gram}, oned_as='row')
                    # scipy.io.savemat("../style/WIKI_STYLE/" + str(class_num) + "/features/style/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'conv5_1': gram}, oned_as='row')
                    # scipy.io.savemat('features/style/'+image[0:-4]+'.mat',mdict={'conv1_1': gram1,'conv2_1':gram2,'conv3_1':gram3,'conv4_1':gram4,'conv5_1':gram5},oned_as='row')
            count = count + 1
    np.save(main_dir + "test_vectors.npy", np.array(test_vectors))
    np.save(main_dir + "test_label.npy", np.array(test_label))
