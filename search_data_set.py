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

target_style = scipy.io.loadmat('test_data/icons/test1/orignal|angry_birds|angry_birds.mat')
search = True


def main():

    if(search == True):

        # first dirctory
        path = "test_data/icons"
        count = 0
        location = os.listdir(path + "/test3/")
        for file in location:
            # name = file.split("|")
            # if(name[1] == "angry_birds"):
            style = scipy.io.loadmat(path + '/test3/' + file)

            sqr = score(style)
            if(count == 0):
                table = np.array([file, sqr])
            else:
                table = np.vstack((table, [file, sqr]))

            if(count % 100 == 0):
                print(count)
            count = count + 1

        print("===========   DONE  ===========")
        print(table)

        sorted_table = np.array(sorted(table, key=lambda a_entry: float(a_entry[1]) ))[0:10]
        fig = plt.figure()

        count = 1
        for name in sorted_table:
            a = fig.add_subplot(2, 5, count)
            a.axis('off')
            try:
                image1 = mpimg.imread('test_data/icons/png/' + name[0][0:-4] + '.png')
                plt.imshow(image1)
            except:
                print("pass")
            count += 1
        plt.show()
        # fig.savefig("me.png")


def score(style):
    squred_difference_style = (((style['conv5_1'] - target_style['conv5_1']))**2)
    sum_style = np.sum(squred_difference_style)
    return(sum_style)


if __name__ == '__main__':
    main()
    # normalize_vector(np.random.randn(1,1000))
