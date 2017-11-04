import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Queue
from operator import itemgetter

# def my_fun(a):
#     return a * a


train_data = np.load("data/new_train_data.npy", encoding='latin1')


pre_load_images = np.load("pre_load_images.npy", encoding='latin1').item()


count = 0
# test1
s = time.time()

for i in range(300):
    img1 = pre_load_images[train_data[i, 0]]
    img2 = pre_load_images[train_data[i, 1]]

    if count == 0:
        batch1 = img1
        batch2 = img2
    else:
        batch1 = np.concatenate((batch1, img1), 0)
        batch2 = np.concatenate((batch2, img2), 0)
    count += 1

print(batch1.shape)
print(time.time() - s)


# test2
s = time.time()

a = train_data[0:300, 0]
b = train_data[0:300, 1]

img11 = np.reshape(itemgetter(*a)(pre_load_images), (300, 224, 224, 3))
img22 = np.reshape(itemgetter(*b)(pre_load_images), (300, 224, 224, 3))


print(np.array(img11).shape)
print(time.time() - s)

# count = 0
# s = time.time()

# for i in range(30):
#     img1 = pre_load_images[train_data[i, 0][0]]
#     img2 = pre_load_images[train_data[i, 0][1]]

#     if count == 0:
#         batch1 = img1
#         batch2 = img2
#     else:
#         batch1 = np.concatenate((batch1, img1), 0)
#         batch2 = np.concatenate((batch2, img2), 0)
#     count += 1

# print(time.time() - s)

# # print(pre_load_images)

# # filename = filenameQ.dequeue()
# # image_bytes = tf.read_file(filename)
# decoded_image = pre_load_images
# image_queue = tf.FIFOQueue(128, [tf.uint8], None)
# enqueue_op = image_queue.enqueue(decoded_image)

# # Create a queue runner that will enqueue decoded images into `image_queue`.
# NUM_THREADS = 16
# queue_runner = tf.train.QueueRunner(
#     image_queue,
#     [enqueue_op] * NUM_THREADS,  # Each element will be run from a separate thread.
#     image_queue.close(),
#     image_queue.close(cancel_pending_enqueues=True))

# # Ensure that the queue runner threads are started when we call
# # `tf.train.start_queue_runners()` below.
# tf.train.add_queue_runner(queue_runner)

# # Dequeue the next image from the queue, for returning to the client.
# img = image_queue.dequeue()

# init_op = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in tqdm(range(10000)):
#         img.eval().mean()
