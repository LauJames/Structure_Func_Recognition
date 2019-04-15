#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Reference: https://github.com/Kyubyong/transformer/blob/master/modules.py
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : transformer.py
@Time     : 2019/1/15 11:10
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


def layer_normalize(inputs,
                    epsilon=1e-8,
                    scope="ln"):
    """
    Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
    Applies layer normalization.
    :param inputs: A tensor with 2 or more dimensions, where the first dimension has`batch_size`.
    :param epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    :param scope:  Optional scope for `variable_scope`.
    :return: A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # Batch Normalization use 0

        beta = tf.get_variable(initializer=tf.zeros_initializer(), shape=params_shape, name='beta')
        gamma = tf.get_variable(initializer=tf.zeros_initializer(), shape=params_shape, name='gamma')

        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


def postitional_encoding(inputs,
                         maxlen,
                         masking=True,
                         scope="positional_encoding",
                         reuse=None):
    """
    Positional encoding. Construct a position embedding lookup table and lookup according to the inputs (position index)
    :param inputs: Tensor. A 3D Tensor with shape of (N, T, E)
    :param maxlen: scalar. Must be >= T
    :param masking: Boolean. If true, padding postions are set to zero
    :param scope: String. Optional scope for 'variable_scope'.
    :param reuse: Boolean. Whether to reuse the weights of a previous layer by the same name.
    :return: Lookuped Position embeddings.
    """
    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=reuse):
        position_index = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        # Github issue has correct the wrong implementation of position encode.
        # position_enc = np.array([[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
        #                           for pos in range(T)])

        # In the paper, section 3.5, the embedding (before the application of sine or cosine)
        # for even positions in [0..num_units] indexed by 2*i is the same as that for odd positions.
        # take 2i as a whole part.
        position_enc = np.array([[pos / np.power(10000, (i - i % 2)/E) for i in range(E)]
                                 for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # Convert to a tensor
        outputs = tf.nn.embedding_lookup(position_enc, position_index)

        # if zero_pad:
        #     lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
        #                               lookup_table[1:, :]), 0)
        # outputs = tf.nn.embedding_lookup(lookup_table, position_index)
        #
        # if scale:
        #     outputs = outputs * num_units ** 0.5

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    """
    Applies multihead attention. N is the batch size, T is the Time length, C is the embedding dim.
    :param queries: Tensor. A 3D tensor with shape of [N, T_q, C_q].
    :param keys: Tensor. A 3D tensor with shape of [N, T_k, T_k].
    :param num_units: Integer. A scalar represent attention size.
    :param num_heads: Integer. Number of heads.
    :param dropout_rate: Float.
    :param is_training: Boolean. Controller of mechanism for dropout.
    :param causality: Boolean. If true, units that reference the future are masked.
    :param scope: Optional scope for 'variable_scope'.
    :param reuse: Boolean, whether to reuse the weight of a previous layer by the same name.
    :return: A 3D tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Batch Multiplication  (h*N, T_q, C/h) * (h*N, C/h, T_k) = (h*N, T_q, T_k)
        # (Dot product in math concept, in Paper 3.2.1 and 3.2.2)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)  # divided sqrt(K_ dimension)

        # Key masking (Remains problems. Position embeddings make keys and queries not zero.)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)  embedding sum ==0 --> 0
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)  tile for multihead
        # expand and tile for queries
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tc.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_mean(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)

        # Broadcasting. (h*N, T_q, T_k)
        outputs *= query_masks

        # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = layer_normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                # num_units=[2048, 512],
                num_units=[1024, 256],
                scope="multihead_attention",
                reuse=None):
    """
    Paper 3.3 Point-wise Feed-Forward Networks.
    Way1: Two linear transformations with a ReLU activation in between.
    Way2: Two convolutions with kernel size 1. (Here we implemented.)
    :param inputs: Tensor. A 3D tensor with shape of [N, T, C].
    :param num_units: List. A list of two integers.
    :param scope: String. Optional scope for 'variable_scope'.
    :param reuse: Boolean. Whether to reuse the weights of a previous layer by the name.
    :return: Tensor. A 3D tensor with the same shape and dtype as inputs.
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs,
                  "filters": num_units[0],
                  "kernel_size": 1,
                  "activation": tf.nn.relu,
                  "use_bias": True}

        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs,
                  "filters": num_units[1],
                  "kernel_size": 1,
                  "activation": None,
                  "use_bias": True}

        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = layer_normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    """
    Applies labels smoothing. See https://arxiv.org/abs/1512.00567.
    :param inputs: Tensor. A 3D tensor with shape of [N, T, V], where V is the number of vocabulary.
    :param epsilon: Float. Smoothing rate.
    :return: Tensor. Same dimension as inputs.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    """
    K = inputs.get_shape().as_list()[-1]  # Number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)
