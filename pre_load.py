import csv
import numpy as np
import os
from shutil import copyfile
import random
import itertools
import utils

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


a = np.load("data/final_val.npy", encoding="latin1")
print(a.shape)

n = []
for i in a:
    print(i)
    n.append([i[0][0], i[0][1], i[1], [2], i[3]])

b = np.array(n, dtype=object)
print(a.shape)
np.save("data/new_val_data.npy", b)

# print(np.array(new).shape)


count = 0
image_dict = dict()
train_img = np.load("data/val_data.npy", encoding='latin1')
for l in class_list:
    print(l)
    print(train_img.shape)
    for img in train_img[l, :]:
        image = utils.load_image("/flush1/raj034/wikiart/wikiart/" + img)
        # print(image.shape)

        image_dict[img] = image.reshape((224, 224, 3))


# print(image_dict)
np.save('pre_load_images_val.npy', image_dict)


a = np.load("data/final_train.npy", encoding="latin1")
print(a.shape)

n = []
for i in a:
    print(i)
    n.append([i[0][0], i[0][1], i[1], [2], i[3]])

b = np.array(n, dtype=object)
print(a.shape)
np.save("data/new_train_data.npy", b)

# print(np.array(new).shape)


count = 0
image_dict = dict()
train_img = np.load("data/train_data.npy", encoding='latin1')
for l in class_list:
    print(l)
    print(train_img.shape)
    for img in train_img[l, :]:
        image = utils.load_image("/flush1/raj034/wikiart/wikiart/" + img)
        # print(image.shape)

        image_dict[img] = image.reshape((224, 224, 3))


# print(image_dict)
np.save('pre_load_images_train.npy', image_dict)
