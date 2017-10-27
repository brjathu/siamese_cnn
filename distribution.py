import numpy as np
import random
import math
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import pylab


def plotDistribution(loss, bins=51, y=500):
    pos = loss[np.where(a[:, 1] == 1)[0]]
    pos = pos[:, 0]

    neg = loss[np.where(a[:, 1] == 0)[0]]
    neg = neg[:, 0]

    plt.figure(1)

    plt.subplot(1, 2, 1)

    plt.hist(pos, bins=np.linspace(0, 400, bins), facecolor='green', alpha=0.5)
    plt.title("positive loss")
    plt.axis([0, 400, 0, y])

    plt.subplot(1, 2, 2)

    plt.hist(neg, bins=np.linspace(0, 400, bins), facecolor='green', alpha=0.5)
    plt.title("negative loss")
    plt.axis([0, 400, 0, y])
    # plt.savefig("distribution.png")
    # plt.show()


a = np.load("distibution25.npy")
plotDistribution(a, 51, 500)
print("loaded")
