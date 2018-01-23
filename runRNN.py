#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : runRNN.py
@Time     : 2018/1/23 15:39
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
from sklearn import metrics
from textRNN_new import TextRnnNew
import dataHelper

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.01, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "./data/labeled_data_part",
                       "Data source for the  data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("seq_length", 600, "sequence length (default: 600)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_layers", 2, "number of layers (default: 2)")
tf.flags.DEFINE_integer("hidden_dim", 128, "neural numbers of hidden layer (default: 128)")
tf.flags.DEFINE_string("rnn_type", 'gru', "rnn type (default: gru)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default:1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def evaluate(x_dev, y_dev, sess):
    """
    Evaluates model on a dev set
    :param x_dev:
    :param y_dev:
    :return:
    """
    data_len = len(x_dev)
    batch_eval = dataHelper.batch_iter_eval(x_dev, y_dev)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch_eval, y_batch_eval in batch_eval:
        batch_len = len(x_batch_eval)
        feed_dict = {
            model.input_x: x_batch_eval,
            model.input_y: y_batch_eval,
            model.dropout_keep_prob: 1.0
        }
        loss, accuracy = sess.run(
            [model.loss, model.accuracy],
            feed_dict)
        total_loss += loss * batch_len
        total_acc += accuracy * batch_len
    time_str = datetime.datetime.now().isoformat()
    print("{}: loss {:g}, acc {:g}".format(time_str, total_loss / data_len, total_acc / data_len))


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("Please input: python3 runRNN.py [train/test]")

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    model = TextRnnNew(
        sequence_length=FLAGS.sequence_length,
        num_classes=FLAGS.num_classes,
        vocab_size=FLAGS.vocab_size,
        embedding_dim=FLAGS.embedding_dim,
        num_layers=FLAGS.num_layers,
        hidden_dim=FLAGS.hidden_dim,
        rnn=FLAGS.rnn_type,
        learning_rate=FLAGS.learning_rate
    )
    if sys.argv[1] == 'train':
        train()
    else:
        test()
