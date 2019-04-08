#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : bidaf.py
@Time     : 19-1-13 下午2:07
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from model.layers.basic_rnn import rnn
from model.layers.match_layer import AttentionFlowMatchLayer
from model.layers.match_layer import MatchLSTMLayer
from model.layers.pointer_net import PointerNetDecoder


class BiDAF(object):
    """
    Implements the Bi-Directional Attention Flow Reading Comprehension Model
    """
    def __init__(self, vocab, args):
        # logging
        self.logger = logging.getLogger("BiDAF")

        self.algorithm = args.algorithm
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.l2_norm = args.l2_norm
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # vocab
        self.vocab = vocab

        # build the computation graph with tensorflow
        start_time = time.time()

        # Placeholders for input, output and dropout
        self.text_a = tf.placeholder(tf.int32, [None, None], name='text_a')
        self.text_b = tf.placeholder(tf.int32, [None, None], name='text_b')
        self.a_mask = tf.placeholder(tf.int32, [None], name='a_mask')
        self.b_mask = tf.placeholder(tf.int32, [None], name='b_mask')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # embedding layer
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True,
                name='word_embeddings'
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.text_a)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.text_b)

        # Encodding layer
        # Employs two Bi-LSTM to encode passage and question separately
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.a_mask, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.b_mask, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

        # Attention Flow Layer: Match
        # Core part. Get the question-aware passage encoding with either BIDAF
        if self.algorithm == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        elif self.algorithm == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented or doesn\'t exist at all.'.format(self.algorithm))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.a_mask, self.b_mask)

        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

        # Modeling layer: Fuse
        # Fuse the context information after match layer using LSTM
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.a_mask, self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

        # Output Layer: decode
        # Employs Pointer Network to get the porbs of each positon to be the start or end of the predicted answer.
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.label)[0]
            concat_passage_encodes = tf.reshape(self.fuse_p_encodes,
                                                [batch_size, -1, 2 * self.hidden_size])
            no_dup_question_encodes = tf.reshape(self.sep_q_encodes,
                                                 [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size])

        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes, no_dup_question_encodes)

        # Loss function
        self.start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_probs,
                                                                         labels=self.label)
        self.end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_probs,
                                                                       labels=self.end_label)
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))

        # regularization method:
        # L2 regularization
        # self.all_params = tf.trainable_variables()
        if self.l2_norm:
            self.variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=3e-7),
                                                             self.variables)
            self.loss += l2_loss
        # weight decay
        if self.weight_decay:
            self.var_ema = tf.train.ExponentialMovingAverage(self.weight_decay)
            # Maintains moving averages of all trainable variables
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                # 应用指数平滑的weight decay之后，再更新loss
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))
        # end of regularization

        # Optimization method
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)



