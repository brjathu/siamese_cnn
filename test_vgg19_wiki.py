import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math
import pickle

tf.logging.set_verbosity(tf.logging.ERROR)

location = os.listdir("test_data/icons/png/")

g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    images = tf.placeholder("float", [1, 224, 224, 3])
    vgg = vgg19.Vgg19("./test2.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    for img in location:
        print(location.index(img))
        img1 = utils.load_image("test_data/icons/png/" + img)
        batch = img1.reshape((1, 224, 224, 3))

        conv5_1 = sess.run(vgg.conv5_1, feed_dict={images: batch})

        features = np.reshape( conv5_1 , (-1,512))
        gram = np.matmul(features.T, features)/features.size
        print(gram.shape)
        scipy.io.savemat("test_data/icons/test6/" + img[0:-4] + '.mat', mdict={'conv5_1': gram}, oned_as='row')
