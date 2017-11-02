import tensorflow as tf
import re
import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class sia:

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % self.TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def preprocss_image(self, rgb, scope):
        rgb_scaled = rgb * 255.0

        with tf.variable_scope(scope):
            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.cast(tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ]), dtype=tf.float32)
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        return bgr

    # def name(self):
    #     return 'Name: ' + self.name.name

    def inference(self, images1, images2, pos, neg):

        print(images1.shape)
        print(images2.shape)

        with tf.variable_scope("siamese"):
            # vgg19_1
            with tf.variable_scope('vgg19_1') as scope:
                bgr1 = self.preprocss_image(images1, scope)
                sia1_conv1_1 = self.conv_layer(bgr1, 3, 64, "conv1_1")
                sia1_conv1_2 = self.conv_layer(sia1_conv1_1, 64, 64, "conv1_2")
                sia1_pool1 = self.max_pool(sia1_conv1_2, 'pool1')

                sia1_conv2_1 = self.conv_layer(sia1_pool1, 64, 128, "conv2_1")
                sia1_conv2_2 = self.conv_layer(sia1_conv2_1, 128, 128, "conv2_2")
                sia1_pool2 = self.max_pool(sia1_conv2_2, 'pool2')

                sia1_conv3_1 = self.conv_layer(sia1_pool2, 128, 256, "conv3_1")
                sia1_conv3_2 = self.conv_layer(sia1_conv3_1, 256, 256, "conv3_2")
                features1 = tf.reshape(sia1_conv3_2, [self.batch_size, -1, 256], name="features1")
                # print("features1", features1.shape)
                gram1 = tf.matmul(tf.transpose(features1, perm=[0, 2, 1]), features1, name="gram1") / (256 * 3196) / (256 * 3196)
                # print("gram1", gram1.shape)

            # self._activation_summary(sia1_conv3_2)

            # vgg19_2
            with tf.variable_scope('vgg19_2') as scope:
                bgr2 = self.preprocss_image(images2, scope)
                sia2_conv1_1 = self.conv_layer(bgr2, 3, 64, "conv1_1")
                sia2_conv1_2 = self.conv_layer(sia2_conv1_1, 64, 64, "conv1_2")
                sia2_pool1 = self.max_pool(sia2_conv1_2, 'pool1')

                sia2_conv2_1 = self.conv_layer(sia2_pool1, 64, 128, "conv2_1")
                sia2_conv2_2 = self.conv_layer(sia2_conv2_1, 128, 128, "conv2_2")
                sia2_pool2 = self.max_pool(sia2_conv2_2, 'pool2')

                sia2_conv3_1 = self.conv_layer(sia2_pool2, 128, 256, "conv3_1")
                sia2_conv3_2 = self.conv_layer(sia2_conv3_1, 256, 256, "conv3_2")
                features2 = tf.reshape(sia2_conv3_2, [self.batch_size, -1, 256], name="features2")
                # print("features2", features2.shape)
                gram2 = tf.matmul(tf.transpose(features2, perm=[0, 2, 1]), features2, name="gram2") / (256 * 3196) / (256 * 3196)
                # print("gram2", gram2.shape)

            self.data_dict = None

            loss = tf.reshape(tf.reduce_sum((gram1 - gram2), axis=[1, 2]) ** 2, (self.batch_size, 1), name="loss")
            pos_loss = tf.multiply(tf.reshape(loss, (self.batch_size, 1)), pos, name="pos_loss")
            neg_loss = tf.multiply(tf.where(tf.less(loss, self.M), self.M - loss, tf.zeros_like(loss)), neg, name="neg_loss")
            loss = tf.reduce_sum(pos_loss + neg_loss, name="total_loss")

            tf.add_to_collection('losses', loss)

        return loss

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5, batch_size_per_tower=1):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
            # print(data_dict)
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.name = "siamese_cnn"
        self.TOWER_NAME = "balck_tower"
        self.batch_size = batch_size_per_tower
        self.M = 200
        print(self.trainable)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # var = tf.Variable(value, name=var_name)
            var = self._variable_on_cpu(var_name, value.shape, initializer=tf.constant_initializer(value))
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
