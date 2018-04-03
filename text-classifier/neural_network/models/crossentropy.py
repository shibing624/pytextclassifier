# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import tensorflow as tf


def category_crossentropy(y_true, y_pred):
    """
    get cross entropy with label and pred
    :param y_true:
    :param y_pred:
    :return:
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true, name='xentroy')
    return tf.reduce_mean(cross_entropy, name='xentroy_mean')
