# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: activation

import tensorflow as tf


def get_activation(activation=None):
    """
    get activation function
    :param activation:
    :return:
    """
    if activation is None:
        return None
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'softmax':
        return tf.nn.softmax
    else:
        raise Exception("Unknown activation function:%s" % activation)
