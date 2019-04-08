#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : highway.py
@Time     : 2018/12/9 17:36
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf


class Highway(object):
    """
    Fully connected Highway network
    """
    def __init__(self, input_x, w_shape, b_shape, activation, bias_init=0.1):
        self.input_x = input_x
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.bias_init = bias_init
        self.activation = activation

        self.w = tf.Variable(tf.truncated_normal(self.w_shape, stddev=0.1), name='weight')
        self.b = tf.Variable(tf.constant(self.bias_init, shape=self.b_shape), name='bias')

        with tf.name_scope('transform_gate'):
            self.w_T = tf.Variable(tf.truncated_normal(self.w_shape, stddev=0.1), name='weight_transform')
            self.b_T = tf.Variable(tf.constant(self.bias_init, shape=self.b_shape), name='bias_transform')

        H = activation(tf.matmul(self.input_x, self.w) + self.b, name='activation')  # fully connected
        T = tf.sigmoid(tf.matmul(self.input_x, self.w_T) + self.b_T, name='transform_gate')  # fully connected
        C = tf.sub(1.0, T, name='carry_gate')  # C = 1 - T

        y = tf.add(tf.mul(H, T), tf.mul(self.input_x, C))  # y = (H * T) + (x * C)
        return y
