import csv
import numpy as np
import os
from shutil import copyfile
import random
import itertools

path = 'data/style_train.csv'

file = open(path, "r")
reader = csv.reader(file)
num_class = 27
train_throshold = 100
train_all_data = np.empty((num_class, 10000), dtype=np.object)

class_count = [0] * num_class
for line in reader:
    # print(line[0])
    # class_count[int(line[1])] += 1
    train_all_data[int(line[1]), class_count[int(line[1])]] = line[0]
    class_count[int(line[1])] += 1

np.save("data/train_all_data.npy", train_all_data, fix_imports=True)

#print(train_all_data)
train_data = np.empty((num_class, train_throshold), dtype=np.object)
for i in range(num_class):
    print(len(np.where(train_all_data[i, :] != np.array( None))[0]))
    train_size = min(train_throshold, len(np.where(train_all_data[i, :] != np.array(None))[0]))
    print(train_size)
    # try:
    train_data[i, 0:train_size] = random.sample(train_all_data[i, np.where(train_all_data[i, :] != np.array(None))[0]], train_size)



np.save("data/train_data.npy", train_data, fix_imports=True)
print(train_data)

train_data = np.load("data/train_data.npy")
train_combinational_data_positive = np.empty((num_class, 4950), dtype=np.object)
train_combinational_data_negative = np.empty((num_class, 4944), dtype=np.object)
selected_class = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


for i in range(num_class):
    data = train_data[i, :]
    # assert(len(np.where(data == None)[0]) > 0)
    com_class_i = list(itertools.combinations(data, 2))
    train_combinational_data_positive[i, :] = com_class_i
    negetive_com = []
    for j in range(num_class):
        if(j != i and j != 1 and j != 2):
            data2 = train_data[j, :]
            com_class_ij = random.sample(list(itertools.product(data, data2)), 206)
            for ij in com_class_ij:
                negetive_com.append([ij, i, j, 0])
    train_combinational_data_negative[i, :] = negetive_com[:4944]

final_train = []
for i in selected_class:
    for j in train_combinational_data_positive[i, :]:
        final_train.append([j, i, i, 1])
    for j in train_combinational_data_negative[i, :]:
        final_train.append(j)


final_train = np.array(final_train)
np.random.shuffle(final_train)
print(final_train.shape)
np.save("data/final_train.npy", final_train, fix_imports=True)


# validation and test datasets
path = 'data/style_val.csv'

file = open(path, "r")
reader = csv.reader(file)
num_class = 27
val_throshold = 60
val_all_data = np.empty((num_class, 10000), dtype=np.object)

class_count = [0] * num_class
for line in reader:
    # print(line[0])
    # class_count[int(line[1])] += 1
    val_all_data[int(line[1]), class_count[int(line[1])]] = line[0]
    class_count[int(line[1])] += 1

np.save("data/val_all_data.npy", val_all_data, fix_imports=True)

val_data = np.empty((num_class, val_throshold), dtype=np.object)
for i in range(num_class):
    # print(len(np.where(val_all_data[i, :] != None)[0]))
    val_size = min(val_throshold, len(np.where(val_all_data[i, :] != np.array(None))[0]))
    # print(val_size)
    # try:
    val_data[i, 0:val_size] = random.sample(val_all_data[i, np.where(val_all_data[i, :] != np.array(None))[0]], val_size)

np.save("data/val_data.npy", val_data, fix_imports=True)
print(val_data.shape)

val_data = np.load("data/val_data.npy")
val_combinational_data_positive = np.empty((num_class, 1770), dtype=np.object)
val_combinational_data_negative = np.empty((num_class, 1752), dtype=np.object)
selected_class = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


for i in range(num_class):
    data = val_data[i, :]
    # assert(len(np.where(data == None)[0]) > 0)
    com_class_i = list(itertools.combinations(data, 2))
    val_combinational_data_positive[i, :] = com_class_i
    negetive_com = []
    for j in range(num_class):
        if(j != i and j != 1 and j != 2):
            data2 = val_data[j, :]
            com_class_ij = random.sample(list(itertools.product(data, data2)), 73)
            for ij in com_class_ij:
                negetive_com.append([ij, i, j, 0])
    val_combinational_data_negative[i, :] = negetive_com[:1752]

final_val = []
for i in selected_class:
    for j in val_combinational_data_positive[i, :800]:
        final_val.append([j, i, i, 1])
    for j in val_combinational_data_negative[i, :800]:
        final_val.append(j)


final_val = np.array(final_val)
np.random.shuffle(final_val)
print(final_val.shape)
np.save("data/final_val.npy", final_val, fix_imports=True)


final_test = []
for i in selected_class:
    for j in val_combinational_data_positive[i, 800:]:
        final_test.append([j, i, i, 1])
    for j in val_combinational_data_negative[i, 800:]:
        final_test.append(j)


final_test = np.array(final_test)
np.random.shuffle(final_test)
print(final_test.shape)
np.save("data/final_test.npy", final_test, fix_imports=True)
