#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : self_att_TF_old.py
@Time     : 2019/4/11 10:59
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import tensorflow.contrib as tc
from model.layers.transformer import *


class SelfAttTFold(object):
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

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True)
            self.x_embedd = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)

        # Scale
        self.x_embedd *= self.vocab.embed_dim ** 0.5

        # Position encoding
        self.x_embedd += postitional_encoding(self.x_embedd, self.sequence_length)

        # Dropout
        self.enc = tf.nn.dropout(self.x_embedd, keep_prob=self.dropout_keep_prob)

        # Transformer Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                self.enc = multihead_attention(queries=self.enc,
                                               keys=self.enc,
                                               num_units=self.num_units,
                                               dropout_rate=1 - self.dropout_keep_prob,
                                               causality=False)
                # FFN
                self.enc = feedforward(self.enc)

        with tf.name_scope('score'):
            self.flatten = tf.layers.flatten(self.enc)

            self.fc = tf.layers.dense(self.flatten, self.num_units, name='fc1')
            self.fc = tf.nn.dropout(self.fc, keep_prob=self.dropout_keep_prob)
            self.fc = tf.nn.relu(self.fc)

            # logits
            self.logits = tf.layers.dense(self.fc, self.num_classes, name='fc2')
            # prediction
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            # loss
            # self.stop_logits = tf.stop_gradient(self.logits)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entropy)

            # step
            # self.global_step = tf.train.get_or_create_global_step()

            # optimizer
            self.train_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

