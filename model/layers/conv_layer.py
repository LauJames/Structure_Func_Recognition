#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : conv_layer.py
@Time     : 19-1-12 下午2:31
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf


def conv_layer(inputs, filter_shape, is_training=False):
    """
    使用给定的参数作为输入创建卷积层
    :param inputs: Tensor 传入该层神经元作为输入
    :param filter_shape: [filter_size, embedding_dim, 1, num_filters]
    :param is_training: bool or Tensor 表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
    :returns Tensor
    一个新的卷积层
    """
    # filter_size = filter_shape[0]
    # embedding_dim = filter_shape[1]
    num_filters = filter_shape[3]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

    # 使用BN，则需要去掉bias和activation
    conv = tf.nn.conv2d(
        inputs,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv"
    )

    layer = tf.nn.conv2d(inputs,
                         W,
                         strides=[1, 1, 1, 1],
                         padding='VALID')  # (N, H, W, C)

    gamma = tf.Variable(tf.ones([num_filters]))
    beta = tf.Variable(tf.zeros([num_filters]))

    pop_mean = tf.Variable(tf.zeros([num_filters]), trainable=False)
    pop_variance = tf.Variable(tf.ones([num_filters]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
        batch_mean, batch_variance = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(tf.equal(is_training, True), batch_norm_training, batch_norm_inference)
    return tf.nn.relu(tf.nn.bias_add(batch_normalized_output, b))
