#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author : Lau James
@Contact : LauJames2017@whu.edu.cn
@Project : Structure_Func_Recognition 
@File : textRNN.py
@Time : 1/20/18 10:57 AM
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""


import tensorflow as tf


class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by 2 rnn layers and 2 fully connected layers
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_dim, num_layers, hidden_dim, rnn):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Keeping track of L2 regularization loss
        # l2_loss = tf.constant(0.0)

        # Embedding layer 指定在cpu
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embeded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # RNN 不需要增加一维
            # self.embeded_chars_expanded = tf.expand_dims(self.embeded_chars, -1)

        # RNN model
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(hidden_dim)

        def dropout():
            """
            为每个rnn核后面加一个dropout层
            :return:
            """
            if rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        # end of RNN model

        with tf.name_scope('rnn'):
            # multi-layers rnn network
            cells = [dropout() for _ in range(num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embeded_chars, dtype=tf.float32)
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

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
