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

name_list = np.load("norm/name.npy")
# style_mat = np.load("norm/style_mat.npy")
style_sqr_matrix = np.load("norm/style_sqr_matrix.npy")

class_list_main = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable_main = np.reshape(np.repeat(class_list_main, 100), (2500, 1))

def search(name):
    ind = np.where(name_list == name)
    res = np.hstack([np.reshape(name_list, (2500, 1)), style_sqr_matrix[ind].T, class_lable_main])
    b = np.array(sorted(res, key=lambda a_entry: float(a_entry[-2])))
    return(b[0:2])




# preprocessing
name = []
count = 0

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable = np.repeat(class_list, 100)

########################################################
#
#         PCA for testing samples
#
########################################################

correct = 0
total = 0
num_test_per_class = 5

result = np.zeros((2, 1))
result = np.array(result)
for l in class_list:
    print("class ==> " + str(l))
    location_style = os.listdir("../wiki/style/WIKI_STYLE/" + str(l) + "/features/style_test1/")
    location_style = random.sample(location_style, num_test_per_class)
    for target_file in location_style:

        target = scipy.io.loadmat("../wiki/style/WIKI_STYLE/" + str(l) + "/features/style_test1/" + target_file)['gram']  # /0.89662

        out = search(target_file)

        output = np.zeros((2, 1))
        count = 0
        for q in out:
            if(str(l) == q[-1]):
                output[count] = 1
            count = count + 1

        result = np.hstack([result, output])
        # print(result[1:])
#
#
print(np.sum(result[1:]) / (result.shape[1]-1) * 100)


# if(total != 0):
# print(correct / total * 100)

#         if (int(out) == l):
#             correct = correct + 1
#         total = total + 1

# print(correct / total * 100)

# map = confusion_matrix(y_true, y_pred)

# print(map)

# df_cm = pd.DataFrame(map, index = [i for i in class_list],
#                   columns = [i for i in class_list])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
