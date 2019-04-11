#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : self_att_TF.py
@Time     : 2019/4/8 12:29
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import tensorflow.contrib as tc
from model.layers.transformer import *


class SelfAttTF(object):
    """
    A Transformer encoder structure for text classification.
    """

    def __init__(self, sequence_length, num_classes, vocab, num_units, num_blocks, learning_rate):
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def encode(self, x, training=True):
        """
        Including embedding layer and transformer encoder layers.
        :param x: [N, T1]
        :param training: Boolean
        :return: memory: encoder outputs. [N, T1, d_model]
        """

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True)
            x_embedd = tf.nn.embedding_lookup(word_embeddings, x)

        # Scale
        x_embedd *= self.vocab.embed_dim ** 0.5

        # Position encoding
        x_embedd += postitional_encoding(self.x_embedd, self.sequence_length)

        # Dropout
        enc = tf.nn.dropout(self.x_embedd, rate=self.dropout_keep_prob)

        # Transformer Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc = multihead_attention(queries=enc,
                                          keys=enc,
                                          num_units=self.num_units,
                                          dropout_rate=self.dropout_keep_prob,
                                          is_training=training,
                                          causality=False)
                # FFN
                enc = feedforward(enc)
        return enc

    def train(self, x, y):
        """
        :return:
            loss: scalar.
            train_op: training operation.
            global_step: scalar.
            summaries: training summary node.
        """
        with tf.name_scope('score'):
            fc = tf.layers.dense(self.encode(x), self.num_units, name='fc1')
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # logits
            logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            # prediction
            y_pred = tf.argmax(tf.nn.softmax(logits), 1)

        with tf.name_scope('loss'):
            # loss
            stop_logits = tf.stop_gradient(logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=stop_logits, labels=y)
            loss = tf.reduce_mean(cross_entropy)

            # step
            global_step = tf.train.get_or_create_global_step()

            # optimizer
            train_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(y_pred, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('global_step', global_step)

        summaries = tf.summary.merge_all()

        return loss, train_optim, global_step, summaries

    def eval(self):
        pass



