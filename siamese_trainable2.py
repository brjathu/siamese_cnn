import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


def _activation_summary(x):
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
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
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


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def preprocss_image(rgb):
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    return bgr


def inference(images1, images2):

    bgr1 = preprocss_image(images1)
    # vgg19_1
    with tf.variable_scope('vgg19_1') as scope:
        sia1_conv1_1 = conv_layer(bgr1, 3, 64, "conv1_1")
        sia1_conv1_2 = conv_layer(sia1_conv1_1, 64, 64, "conv1_2")
        sia1_pool1 = max_pool(sia1_conv1_2, 'pool1')

        sia1_conv2_1 = conv_layer(sia1_pool1, 64, 128, "conv2_1")
        sia1_conv2_2 = conv_layer(sia1_conv2_1, 128, 128, "conv2_2")
        sia1_pool2 = max_pool(sia1_conv2_2, 'pool2')

        sia1_conv3_1 = conv_layer(sia1_pool2, 128, 256, "conv3_1")
        sia1_conv3_2 = conv_layer(sia1_conv3_1, 256, 256, "conv3_2")

    bgr2 = preprocss_image(images2)
    # vgg19_1
    with tf.variable_scope('vgg19_2') as scope:
        sia2_conv1_1 = conv_layer(bgr2, 3, 64, "conv1_1")
        sia2_conv1_2 = conv_layer(sia2_conv1_1, 64, 64, "conv1_2")
        sia2_pool1 = max_pool(sia2_conv1_2, 'pool1')

        sia2_conv2_1 = conv_layer(sia2_pool1, 64, 128, "conv2_1")
        sia2_conv2_2 = conv_layer(sia2_conv2_1, 128, 128, "conv2_2")
        sia2_pool2 = max_pool(sia2_conv2_2, 'pool2')

        sia2_conv3_1 = conv_layer(sia2_pool2, 128, 256, "conv3_1")
        sia2_conv3_2 = conv_layer(sia2_conv3_1, 256, 256, "conv3_2")

    data_dict = None

    loss = tf.reduce_sum(sia1_conv3_2 - sia2_conv3_2)

    return loss


def __init__(vgg19_npy_path=None, trainable=True, dropout=0.5):
    if vgg19_npy_path is not None:
        data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print(data_dict)
    else:
        data_dict = None

    var_dict = {}
    trainable = trainable

    print(trainable)
    return __self__


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu


def fc_layer(bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def get_conv_var(filter_size, in_channels, out_channels, name):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name, 0, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return filters, biases


def get_fc_var(in_size, out_size, name):
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    weights = get_var(initial_value, name, 0, name + "_weights")

    initial_value = tf.truncated_normal([out_size], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return weights, biases


def get_var(initial_value, name, idx, var_name):
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    if trainable:
        # var = tf.Variable(value, name=var_name)
        var = _variable_on_cpu(var_name, initializer=tf.constant_initializer(value))
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    var_dict[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var


def save_npy(sess, npy_path="./vgg19-save.npy"):
    assert isinstance(sess, tf.Session)

    data_dict = {}

    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name][idx] = var_out

    np.save(npy_path, data_dict)
    print(("file saved", npy_path))
    return npy_path


def get_var_count():
    count = 0
    for v in list(var_dict.values()):
        count += reduce(lambda x, y: x * y, v.get_shape().as_list())
    return count
