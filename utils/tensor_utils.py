# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import tensorflow as tf


def zero_nil_slot(t, name=None):
    """
    Overwrite the nil_slot (first 1 rows) of the input zeros tensor
    :param t: 2D tensor
    :param name: str
    :return:
    """
    with tf.name_scope('zero_nil_slot'):
        s = tf.shape(t)[1]
        z = tf.zeros([1, s], dtype=tf.float32)
        return tf.concat(axis=0, name=name,
                         values=[z, tf.slice(t, [1, 0], [-1, -1])])


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Add gradient noise
    :param t: 2D tensor, a gradient
    :param stddev:
    :param name:
    :return: t + gaussian noise
    """
    with tf.name_scope('add_gradient_noise'):
        noise = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, noise, name=name)


def mask_tensor(input_data, len, max_len, dtype=tf.float32):
    """
    Mask
    :param input_data:
    :param len:
    :param max_len:
    :param dtype:
    :return:
    """
    mask = tf.cast(tf.sequence_mask(len, max_len), dtype)
    return tf.multiply(input_data, mask)


def get_ckpt_path(model_path):
    ckpt_path = ""
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        ckpt_file = ckpt.model_checkpoint_path.split('/')[-1]
        ckpt_path = os.path.join(model_path, ckpt_file)
    return ckpt_path
