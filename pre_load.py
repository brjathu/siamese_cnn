import csv
import numpy as np
import os
from shutil import copyfile
import random
import itertools
import utils

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


count = 0
image_dict = dict()
train_img = np.load("data/train_data.npy", encoding='latin1')
for l in class_list:
    print(l)
    for img in train_img[l, :]:
        image = utils.load_image("/flush1/raj034/wikiart/wikiart/" + img)
        print(image.shape)

        image_dict[img] = image.reshape((1, 224, 224, 3))


print(image_dict)
np.save('my_file.npy', image_dict)
