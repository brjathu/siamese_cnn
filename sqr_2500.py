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
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle
from sklearn.preprocessing import normalize

# preprocessing
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

# name = []
# class_name = []
# count = 0

# for l in class_list:
#     print("class ==> " + str(l))
#     location = os.listdir("../wiki/style/WIKI_STYLE/" + str(l) + "/features/style_test1/")
#     for file in location:

        # content_vect = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content4096/" + file)['f6']

        # style = scipy.io.loadmat("../wiki/style/WIKI_STYLE/" + str(l) + "/features/style_test1/" + file)['gram']

        # content_mat.append([content_vect[0]])
        # name.append(file)
        # class_name.append(l)

        # if(count == 0):
        #     style_mat = np.array( [file, style[np.triu_indices(512)], l])
        # else:
        #     style_mat = np.vstack([style_mat, [file, np.array(style[np.triu_indices(512)]), l]])
        # count = count + 1

# content_mat = np.array(content_mat, dtype=object)[:, 0]
# print((content_mat).shape)
# mean_vect = np.mean(content_mat, axis=0)
# np.save("norm/content_mat_4096.npy", content_mat)
# np.save("norm/content_mean_vect_4096.npy", mean_vect)


# print((style_mat).shape)
# mean_vect = np.mean(style_mat, axis=0)

# np.save("norm/name.npy", name)
# np.save("norm/class.npy", class_name)
# np.save("norm/style_mat2.npy", style_mat)
# np.save("norm/style_mean_vect.npy", mean_vect)
# np.save("norm/style_max_val.npy", max_val)


###########################################################################################################################
# normalizing
###########################################################################################################################


# content_mat = np.load("norm/content_mat_4096.npy")
# content_mean = np.load("norm/content_mean_vect_4096.npy")

# batch_var = np.sum((content_mat - content_mean)**2) / 2500
# print(batch_var)  #316981
# normalized_content = (content_mat - content_mean) / np.sqrt(batch_var)
# np.save("norm/normalized_content_4096.npy", normalized_content)


# style_mat = np.load("norm/style_mat.npy")
# style_mean = np.load("norm/style_mean_vect.npy")
# batch_var = np.sum((style_mat - style_mean)**2) / 2500
# print(batch_var)  #10263181.2703
# normalized_style = ((style_mat - style_mean)) / batch_var
# np.save("norm/normalized_style.npy", normalized_style)

###########################################################################################################################
# create squared difference matrix
###########################################################################################################################


# normalized_content = np.load("norm/normalized_content_4096.npy")
# print(np.max(normalized_content))
# num_sample = normalized_content.shape
# print(num_sample)
# content_sqr_matrix = np.zeros((num_sample[0], num_sample[0]))
# for i in range(num_sample[0]):
#     print(i)
#     for j in range(num_sample[0]):
#         content_sqr_matrix[i][j] = np.sum((normalized_content[i] - normalized_content[j])**2)

# np.save("norm/content_sqr_matrix_4096.npy", content_sqr_matrix)


normalized_style = np.load("norm/style_mat.npy")
print(normalized_style.shape)
num_sample = normalized_style.shape

style_sqr_matrix = np.zeros((num_sample[0], num_sample[0]))
for i in range(num_sample[0]):
    print(i)
    for j in range(num_sample[0]):
        style_sqr_matrix[i][j] = np.sum((normalized_style[i] - normalized_style[j])**2)

np.save("norm/style_sqr_matrix.npy", style_sqr_matrix)


# style_sqr_matrix = np.load("norm/style_sqr_matrix.npy")
# content_sqr_matrix = np.load("norm/content_sqr_matrix.npy")
# print(style_sqr_matrix.shape)
# print(content_sqr_matrix.shape)


# plt.imshow(content_sqr_matrix, interpolation='nearest', cmap=plt.cm.ocean,
#            extent=(0.5, np.shape(content_sqr_matrix)[0] + 0.5, 0.5, np.shape(content_sqr_matrix)[1] + 0.5))
# plt.colorbar()
# plt.show()


# graph = np.load("norm/graph.npy")


# print(graph[:, 1])

# plt.plot((graph[:, 1]) / 3)
# plt.show()
