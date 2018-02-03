#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author : Lau James
@Contact : LauJames2017@whu.edu.cn
@Project : Structure_Func_Recognition 
@File : runCLSTM_header.py
@Time : 2/3/18 10:31 PM
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import csv
from sklearn import metrics
from model.CLSTMmodel import CLSTM
from data import dataHelper

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "./data/header3500",
                       "Data source for the train data.")
tf.flags.DEFINE_string("test_data_file", "./data/header500",
                       "Data source for the test data.")
tf.flags.DEFINE_string("tensorboard_dir", "tensorboard_dir/textCLSTM_header", "saving path of tensorboard")
tf.flags.DEFINE_string("save_dir", "checkpoints/textCLSTM_header", "save base dir")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("seq_length", 15, "sequence length (default: 600)")
tf.flags.DEFINE_integer("vocab_size", 8000, "vocabulary size (default: 5000)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_layers", 2, "number of layers (default: 2)")
tf.flags.DEFINE_integer("hidden_dim", 128, "neural numbers of hidden layer (default: 128)")
tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate (default:1e-3)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

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
FLAGS._parse_flags()
save_path = os.path.join(FLAGS.save_dir, 'best_validation')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


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
    # time_str = datetime.datetime.now().isoformat()
    # print("{}: loss {:g}, acc {:g}".format(time_str, total_loss / data_len, total_acc / data_len))
    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver ...")
    tensorboard_dir = FLAGS.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # Configuring Saver

    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Load data
    print("Loading data...")
    start_time = time.time()
    x_train, y_train, x_dev, y_dev = dataHelper.load_header_data(FLAGS.train_data_file, FLAGS.dev_sample_percentage, FLAGS.save_dir, FLAGS.seq_length)
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
            feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob)
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


def test():
    print("Loading test data ...")
    start_time = time.time()
    x_raw, y_test = dataHelper.get_para_label(FLAGS.test_data_file)
    # y_test = np.argmax(y_test, axis=1)
    vocab_path = os.path.join(FLAGS.save_dir, "vocab")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    print('Testing ...')
    loss_test, acc_test = evaluate(x_test, y_test, session)
    print('Test loss: {0:>6.2}, Test acc: {1:>7.2%}'.format(loss_test, acc_test))

    x_test_batches = dataHelper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
    all_predictions = []
    all_predict_prob = []
    count = 0  # concatenate第一次不能为空，需要加个判断来赋all_predict_prob值
    for x_test_batch in x_test_batches:
        batch_predictions, batch_predict_prob = session.run([model.y_pred, model.prob],
                                                            feed_dict={
                                                                model.input_x: x_test_batch,
                                                                model.dropout_keep_prob: 1.0
                                                            })
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        if count == 0:
            all_predict_prob = batch_predict_prob
        else:
            all_predict_prob = np.concatenate([all_predict_prob, batch_predict_prob])
        count = 1

    # Evaluation indexes
    y_test = np.argmax(y_test, axis=1)
    print("Precision, Recall, F1-Score ...")
    print(metrics.classification_report(y_test, all_predictions, target_names=['Introduction', 'Related work',
                                                                               'Methods', 'Experiment', 'Conclusion']))
    # Confusion Matrix
    print("Confusion Matrix ...")
    print(metrics.confusion_matrix(y_test, all_predictions))

    out_dir = os.path.join(FLAGS.save_dir, 'predict_prob.csv')
    print("Saving evaluation to {0}".format(out_dir))
    with open(out_dir, 'w') as f:
        csv.writer(f).writerows(all_predict_prob)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("Please input: python3 runCLSTM.py [train/test]")

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    model = CLSTM(
        sequence_length=FLAGS.seq_length,
        num_classes=FLAGS.num_classes,
        vocab_size=FLAGS.vocab_size,
        embedding_dim=FLAGS.embedding_dim,
        num_layers=FLAGS.num_layers,
        hidden_dim=FLAGS.hidden_dim,
        learning_rate=FLAGS.learning_rate,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters
    )
    if sys.argv[1] == 'train':
        train()
    else:
        test()
