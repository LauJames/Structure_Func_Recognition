#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
@Reference: https://www.jianshu.com/p/cc6407444a8c
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : attention.py
@Time     : 19-1-19 下午1:38
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Implement attention mechanism for rnn layer.
    :param inputs: Tensor. RNN outputs.
    :param attention_size: Integer.
    :param time_major: Boolean. If True, the inputs dimension is (T, B, D)
    :param return_alphas: Boolean. If True, return the alphas attention weights.
    :return: Attention-ed outputs and alphas(Optional).
    """
    if isinstance(inputs, tuple):
        # In case of Bi-RNNs not concatenate the forward and the backward RNN outputs
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T, B, D) => (B, T, D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.get_shape().as_list()[2]  # D value - hidden size of the RNN layer.

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps.
        # The shape of 'v' is (B, T, D) * (D, A) = (B, T, A), where A=Attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from 'v' is reduced with 'u' vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B, T) shape
    alphas = tf.nn.softmax(vu, name='alphas')

    # Outputs of (Bi)RNNs is reduced with attention vector
    # The result has (B, D) shape.
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return outputs
    else:
        return outputs, alphas

