#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : LSTMConv.py
@Time     : 2018/1/29 17:44
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""


import tensorflow as tf
import tensorflow.contrib as contrib


class LSTMConv(object):
    """
    A LSTM+Convolutional network for text classification.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_dim, num_layers,
                 hidden_dim, learning_rate, filter_sizes, num_filters):
        
        # Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer 指定在cpu进行
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embeded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.name_scope('cnn'):
            # CNN layer
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            # max_feature_length = sequence_length - max(enumerate(filter_sizes)) + 1
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_dim, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv"
                    )

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # Max-pooling over the  outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    # Remove channel dimension
                    pooled_reshape = tf.squeeze(pooled, [2])
                    # Cut the feature sequence at the end based on the maximum filter length
                    # pooled_reshape = pooled_reshape[:, :max_feature_length, :]
                    pooled_outputs.append(pooled_reshape)
            # Concatenate the outputs from different filters
            if len(len(filter_sizes)) > 1:
                rnn_inputs = tf.concat(pooled_outputs, -1)
            else:
                rnn_inputs = pooled_reshape

        with tf.name_scope('rnn'):
            # LSTM cell
            cell = contrib.rnn.LSTMCell(
                len(filter_sizes) * num_filters,
                forget_bias=1.0,
                state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse
            )

            # Add dropout to LSTM cell
            cell = contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

            # Stacked LSTMs
            cell = contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, dtype=tf.float32)
            # 取最后一个时序作为结果
            last = _outputs[:, -1, :]

        with tf.name_scope('score'):
            # Dense layer, followed a relu activiation layer
            fc = tf.layers.dense(last, hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # classfier
            self.logits = tf.layers.dense(fc, num_classes, name='fc2')
            # prediction
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            # loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizer
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

