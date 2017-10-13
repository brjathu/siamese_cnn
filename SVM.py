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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle


# preprocessing

name = []
count = 0

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable = np.repeat(class_list, 100)

ratio = 5
max = 0
train = False
content_mean = np.load("norm/content_mean_vect_4096.npy")

if (train == True):
    for l in class_list:
        print(l)
        location = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/style")

        for file in location:

            style_file = (scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/style4096/" + file)['conv5_1'])
            # style_half = np.array(style_file[np.triu_indices(512)])

            # style_file = style_file / 7826.146
            # content_file = (scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content4096/" + file)['f6'] - content_mean) / 316981

            if (count == 0):
                final_vector = np.hstack([style_file])
            else:
                final_vector = np.vstack([final_vector, np.hstack([style_file])])

            count = count + 1
    # c_vector = np.load("norm/content_mat_4096.npy")
    print(final_vector.shape)
    # # np.save("style/style_mean.npy" , np.mean(style_all, axis = 0))
    # # np.save("style/content_mean.npy" , np.mean(content_all, axis = 0))

    clf = SVC(kernel='poly', gamma="auto", decision_function_shape="ovr")
    # clf = LinearDiscriminantAnalysis(n_components=20)
    clf.fit(final_vector, class_lable.T)

    with open('svm.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("=======DONE========")

else:
    ########################################################
    #
    #         PCA for testing samples
    #
    ########################################################

    with open('svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    # comp = np.load("style/comp.npy")
    # mean = np.load("style/mean.npy")

    # print(comp.shape)
    # print(mean.shape)

    content4096_mean = np.load("norm/content_mean_vect_4096.npy")
    correct = 0
    total = 0
    num_test_per_class = 50
    y_true = np.repeat(class_list, num_test_per_class)
    y_pred = []

    for l in class_list:
        print(l)
        location_style = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/style4096/")
        location_style = random.sample(location_style, num_test_per_class)
        for file in location_style:

            style = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/style4096/" + file)['conv5_1']
            # content = ((scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content4096/" + file)['f6']) - content4096_mean) / 316981
            # features = np.reshape(style, (1, -1))

            # style1000 = ((features - mean).dot(comp.T) - mean_s)  # /8918.6 * ratio
            # style1000 = style1000 / np.linalg.norm(style1000)
            test = np.hstack([style])

            # print(test.shape)
            # style_half = np.array(style[np.triu_indices(512)])

            res = clf.predict(test)
            print(res[0], l)
            y_pred.append(res[0])

            if(total != 0):
                print(correct / total * 100)

            if (res[0] == l):
                correct = correct + 1
            total = total + 1

    print(correct / total * 100)

    # map = confusion_matrix(y_true, y_pred)

    # print(map)

    # df_cm = pd.DataFrame(map, index=[i for i in class_list],
    #                      columns=[i for i in class_list])
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()
