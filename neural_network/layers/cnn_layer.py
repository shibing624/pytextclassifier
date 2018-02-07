# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import numpy as np
import tensorflow as tf
from neural_network.activation import get_activation


class CNN(object):
    def __init__(self, input_data, filter_length, nb_filter, strides=[1, 1, 1, 1],
                 padding='VALID', activation='tanh', pooling=True, name='CNN1D'):
        """
        1D conv layer
        :param input_data:
        :param filter_length:
        :param nb_filter:
        :param strides:
        :param padding:
        :param activation:
        :param pooling:
        :param name:
        """
        in_height, in_width = map(int, input_data.get_shape()[1:])
        self._input_data = tf.expand_dims(input_data, -1)  # shape = [x,x,x,1]
        self._filter_length = filter_length
        self._nb_filter = nb_filter
        self._strides = strides
        self._padding = padding
        self._activation = get_activation(activation)
        self.pooling = pooling
        self._name = name

        filter_length = self._filter_length
        nb_filter = self._nb_filter
        with tf.name_scope('%s_%d' % (name, filter_length)):
            if activation != 'relu':
                fan_in = filter_length * in_width
                fan_out = nb_filter * (in_width - filter_length + 1)
                w_bound = np.sqrt(6. / (fan_in + fan_out))
                self.weights = tf.Variable(tf.random_uniform(
                    minval=-w_bound, maxval=w_bound, dtype='float32',
                    shape=[filter_length, in_width, 1, nb_filter]), name='conv_weight')
                tf.summary.histogram('weights', self.weights)
            else:
                # init weight for relu
                w_values = tf.random_normal(shape=[filter_length, in_width, 1, nb_filter]) * tf.sqrt(
                    2. / (filter_length * in_width * nb_filter))
                self.weights = tf.Variable(w_values, name='conv_weight')
            # bias
            self.biases = tf.Variable(tf.constant(0.1, shape=[nb_filter, ]), name='conv_bias')
            tf.summary.histogram('biases', self.biases)
        self.call()

    def call(self):
        conv_output = tf.nn.conv2d(
            input=self._input_data,
            filter=self.weights,
            strides=self._strides,
            padding=self._padding
        )
        # output shape = [batch_size, new_height, 1, nb_filters]
        linear_output = tf.nn.bias_add(conv_output, self.biases)
        act_output = (linear_output if self._activation is None
                      else self._activation(linear_output))
        if self.pooling:
            # max pooling
            self._output = tf.reduce_max(tf.squeeze(act_output, [2]), 1)
        else:
            self._output = tf.squeeze(act_output, axis=2)

    @property
    def input_data(self):
        return self._input_data

    @property
    def output(self):
        return self._output

    @property
    def output_dim(self):
        return self._nb_filter

    @property
    def get_weights(self):
        return self.weights
