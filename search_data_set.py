# Copyright (c) Jathushan Rajasegaran
# Email - brjathu@gmail.com
# University of Moratuwa
# Department of Electronics and Telecommunication

import os
import numpy as np
import scipy.misc
import scipy.io
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.special import entr

target_content = scipy.io.loadmat('playstore/features/content1000/orignal|angry_birds|angry_birds.mat')
target_style = scipy.io.loadmat('playstore/features/style/orignal|angry_birds|angry_birds.mat')
search = True


def main():

    if(search == True):

        # first dirctory
        path = "playstore/features"
        count = 0
        location = os.listdir(path + "/content1000/")
        for file in location:
            name = file.split("|")
            if(name[1] == "angry_birds"):

                content = scipy.io.loadmat(path + '/content1000/' + file)
                style = scipy.io.loadmat(path + '/style/' + file)

                # print(sum_content, sum_style)
                cos = score(style, content, 1)
                sqr = score(style, content, 2)
                if(count == 0):
                    table = np.array([file, cos[0], cos[1], sqr[0], sqr[1]])
                else:
                    table = np.vstack((table, [file, cos[0], cos[1], sqr[0], sqr[1]]))

                if(count % 100 == 0):
                    print(count)
                count = count + 1

        print("===========   DONE  ===========")
        print(table)
        count = 0
        correct = 0
        total = 10
        sorted_table = np.array(sorted(table, key=lambda a_entry: float(a_entry[3]) + 1e-15 * float(a_entry[4])))[0:total]
        fig = plt.figure()

        for name in sorted_table:
            test = name[0].split("|")[0]
            print(test)
            if(test == "orignal"):
                correct += 1
            count = count + 1
            a = fig.add_subplot(2, 5, count)
            a.axis('off')
            try:
                image1 = mpimg.imread('playstore/png/all/' + name[0][0:-4] + '.png')
                plt.imshow(image1)
            except:
                print("pass")
        print(correct / total * 100)
        # plt.show()
        # fig.savefig("me.png")


def score(style, content, type=1):

    if(type == 1):
        cosine_mat_style = (style['conv5_1'] * target_style['conv5_1'])
        sum_style = np.sum(cosine_mat_style) / np.sqrt(np.sum(style['conv5_1']**2) * np.sum(target_style['conv5_1']**2))

        cosine_mat_content = (content['prob'] * target_content['prob'])
        sum_content = np.sum(cosine_mat_content) / np.sqrt(np.sum(content['prob']**2) * np.sum(target_content['prob']**2))
    elif(type == 2):
        squred_difference_style = (((style['conv5_1'] - target_style['conv5_1']))**2)
        sum_style = np.sum(squred_difference_style)

        squred_difference_content = (((content['prob'] - target_content['prob']))**2)
        sum_content = np.sum(squred_difference_content)

    return([sum_content, sum_style])


if __name__ == '__main__':
    main()
    # normalize_vector(np.random.randn(1,1000))
