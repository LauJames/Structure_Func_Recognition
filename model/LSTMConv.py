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


class LSTMConv(object):
    """
    A LSTM+Convolutional network for text classification.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_dim, num_layers,
                 hidden_dim, learning_rate):
        
        # Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32)
