#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : runTF_header.py
@Time     : 2019/4/11 11:39
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import sys
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
import csv
from sklearn import metrics
from model.self_att_TF_old import SelfAttTFold
from data import dataHelper
from data.vocab import Vocab


# Data loading params
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "./data/header3500",
                       "Data source for the train data.")
tf.flags.DEFINE_string("test_data_file", "./data/header500",
                       "Data source for the test data.")
tf.flags.DEFINE_string("tensorboard_dir", "tensorboard_dir/TF_header", "saving path of tensorboard")
tf.flags.DEFINE_string("save_dir", "checkpoints/TF_header", "save base dir")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("seq_length", 15, "sequence length (default: 600)")
# tf.flags.DEFINE_integer("vocab_size", 8000, "vocabulary size (default: 5000)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_integer("num_layers", 2, "number of layers (default: 2)")
tf.flags.DEFINE_integer("hidden_dim", 512, "neural numbers of hidden layer (default: 128)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default:1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
save_path = os.path.join(FLAGS.save_dir, 'best_validation')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


def prepare():
    # Load data
    print("Loading data...")
    start_time = time.time()
    headers, labels = dataHelper.get_header(FLAGS.train_data)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    print("Building vocabulary and embedding table ...")
    start_time = time.time()
    vocab = Vocab(lower=True)
    for word in dataHelper.word_iter(headers):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filter_num = unfiltered_vocab_size - vocab.size()
    print("After filter {} tokens, the final vocab size is {}".format(filter_num, vocab.size()))

    print("Assigning embeddings ...")
    # Pre-train
    # vocab.load_pretrained_embeddings('./data/glove.vectors.300d.noheader.txt')
    vocab.randomly_init_embeddings(embed_dim=300)

    print("Saving vocab ...")
    with open(os.path.join(FLAGS.save_dir, 'vocab.data')) as fout:
        pickle.dump(vocab, fout)

    print("Done with preparing!")
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train():
    print("Configuring TensorBoard and Saver ...")
    tensorboard_dir = FLAGS.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    model = SelfAttTFold(
        sequence_length=FLAGS.seq_length,
        num_classes=FLAGS.num_classes,
        # vocab=
        num_units=FLAGS.num_units,
        num_blocks=FLAGS.num_blocks,
        learning_rate=FLAGS.learning_rate
    )

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # Configuring Saver

    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Load data
    print("Load vocab")
    with open(os.path.join(FLAGS.save_dir, 'vocab.data')) as pkl_in:
        vocab = pickle.dump(pkl_in)

    print("Loading data ...")
    start_time = time.time()
    x_train, y_train, x_dev, y_dev = dataHelper.get_header(FLAGS.train_data)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # Create Session
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5,
        allow_growth=True
    )
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options
    )
    session = tf.Session(config=session_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and deviation ...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_dev = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_imporvement = 15000  # 如果超过30000论未提升，提前结束训练

    tag = False

    for epoch in range(FLAGS.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = dataHelper.batch_iter_per_epoch(x_train, y_train, FLAGS.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob, model=model)
            if total_batch % FLAGS.checkpoint_every == 0:
                # write to tensorboard scalar
                summary = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(summary, total_batch)

            if total_batch % FLAGS.evaluate_every == 0:
                # print performance on train set and dev set
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_dev, acc_dev = evaluate(x_dev, y_dev, session)

                if acc_dev > best_acc_dev:
                    # save best result
                    best_acc_dev = acc_dev
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                print('Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, ''Val Acc: '
                      '{4:>7.2%}, Time: {5} {6}'
                      .format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_imporvement:
                # having no improvement for a long time
                print("No optimization for a long time, auto-stopping...")
                tag = True
                break
        if tag:  # early stopping
            break


def predict():
    pass


if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #      raise ValueError("Please input: python3 runRNN_header.py [train/test]")
    # print("\nParameters:")
    # for key in sorted(FLAGS.__flags.keys()):
    #     print("{}={}".format(key.upper(), FLAGS.__flags[key].value))
    # print("")
    #

    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'train':
        train()
    else:
        predict()

