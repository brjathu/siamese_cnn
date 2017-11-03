import os
import numpy as np
import tensorflow as tf
import siamese_trainable as sia
import utils
import scipy.io
import time
import math
import re

num_gpus = 3
EPOCH = 100
batch_size = 3
model = "testP1"
main_dir = "/flush1/raj034/vgg19/" + model + "/"
# main_dir = "bck/" + model + "/"
os.system("mkdir " + main_dir)
LOG_FILE = open(main_dir + 'log.txt', 'a')


train_data = np.load("data/final_train.npy", encoding='latin1')
# logEntry("debugger a ========= >  " + str(train_data.shape))
train_data = train_data[0:800, :]


val_data = np.load("data/final_val.npy", encoding='latin1')
# logEntry("debugger c ========= >  " + str(val_data.shape))
val_data = val_data[0:160, :]

test_data = np.load("data/final_test.npy", encoding='latin1')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(str(TMP_STRING))


def tower_loss(scope, images1, images2, pos, neg):

    loss = siamese.inference(images1, images2, pos, neg)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % siamese.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


siamese = sia.sia("vgg19.npy")
with tf.Graph().as_default(), tf.device('/cpu:0'):

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(1e-5,
                                    global_step,
                                    250,
                                    0.01,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    tower_grads = []

    images1 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images1")
    images2 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name="images2")
    pos = tf.placeholder(tf.float32, [batch_size, 1], name="positive_samples")
    neg = tf.placeholder(tf.float32, [batch_size, 1], name="negetive_samples")

    split_img1 = tf.split(images1, num_gpus)
    split_img2 = tf.split(images2, num_gpus)
    split_pos = tf.split(pos, num_gpus)
    split_neg = tf.split(neg, num_gpus)

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (siamese.TOWER_NAME, i)) as scope:

                    loss = tower_loss(scope, split_img1[i], split_img2[i], split_pos[i], split_neg[i])
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

    with tf.variable_scope("opts"):
        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', lr))

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True))
    sess.run(init)

    loss_avg = 0
    val_avg = 0
    distibution = np.array((batch_size, 2))
    validation = np.array((batch_size, 2))

    for epoch in range(EPOCH):

        for mini_batch in range(int(train_data.shape[0] / batch_size)):
            start_time = time.time()
            min_batch_data = train_data[mini_batch * batch_size: (mini_batch + 1) * batch_size]
            y_pos = np.reshape(train_batch_data[:, 3], (batch_size, 1))
            y_neg = (y_pos - 1) * (-1)  # change
            count = 0

            for i in range(batch_size):
                img1 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + min_batch_data[i, 0][0])
                img2 = utils.load_image("/flush1/raj034/wikiart/wikiart/" + min_batch_data[i, 0][1])

                if count == 0:
                    batch1 = img1.reshape((1, 224, 224, 3))
                    batch2 = img2.reshape((1, 224, 224, 3))
                else:
                    batch1 = np.concatenate((batch1, img1.reshape((1, 224, 224, 3))), 0)
                    batch2 = np.concatenate((batch2, img2.reshape((1, 224, 224, 3))), 0)

                count += 1
            _, loss_value, lr_val, summary_str = sess.run([train_op, loss, lr, summary_op], feed_dict={images1: batch1, images2: batch2, pos: y_pos, neg: y_neg})
            logEntry("epoch = " + str(epoch) + "\t mini batch = " + str(mini_batch) + "\t loss =" loss_value)
            logEntry(lr_val)
            duration = time.time() - start_time
            logEntry("time = " + str(duration))
            summary_writer = tf.summary.FileWriter(main_dir + LOG_test)
            summary_writer.add_graph(sess.graph)
            summary_writer.add_summary(summary_str, epoch)

    summary_writer.close()
