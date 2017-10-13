import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math


tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 10
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    images = tf.placeholder("float", [batch_size, 224, 224, 3])
    # feed_dict = {images: batch}
    vgg = vgg19.Vgg19("/home/brjathu/Downloads/test1.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    for l in class_list:
        print(l)
        class_num = l
        location = os.listdir("style/WIKI_STYLE/" + str(class_num) + "/img/")
        os.system("mkdir style/WIKI_STYLE/" + str(class_num) + "/features/style_test1")
        count = 0
        name = []
        for image in location:
            name.append(image)
            if(count % batch_size == 0):
                img1 = utils.load_image("style/WIKI_STYLE/" + str(class_num) + "/img/" + image)
                batch = img1.reshape((1, 224, 224, 3))
            else:
                img1 = utils.load_image("style/WIKI_STYLE/" + str(class_num) + "/img/" + image)
                batch1 = img1.reshape((1, 224, 224, 3))
                batch = np.concatenate((batch, batch1), 0)

            print(batch.shape)

            if ((count + 1) % batch_size == 0):
                # fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
                conv5_1 = sess.run(vgg.conv5_1, feed_dict={images: batch})
                for i in range(batch_size):
                    features = np.reshape(conv5_1[i], (-1, 512))
                    gram = np.matmul(features.T, features) / features.size
                    scipy.io.savemat("style/WIKI_STYLE/" + str(class_num) + "/features/style_test1/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'gram': gram}, oned_as='row')
                    # scipy.io.savemat("../style/WIKI_STYLE/" + str(class_num) + "/features/style/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'conv5_1': gram}, oned_as='row')
                    # scipy.io.savemat('features/style/'+image[0:-4]+'.mat',mdict={'conv1_1': gram1,'conv2_1':gram2,'conv3_1':gram3,'conv4_1':gram4,'conv5_1':gram5},oned_as='row')
            count = count + 1
