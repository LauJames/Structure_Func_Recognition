# -*- coding:utf8 -*-
import tensorflow as tf
import time
import datetime
import os
import sys
import numpy as np
import argparse
from sklearn import metrics
import  csv
from  model.BiLSTM import BiLSTM
from data import dataLoader
# data loading parameters
# 可以使用python argparser.AgumentParser
# parse = argparse.ArgumentParser()
# parse.add_argument(dest='train_file_path', default='/Users/j_yee/Desktop/LSTM/header3500',
#                    help='data source for the train data')
# parse.add_argument(dest='test_file_path', default='/Users/j_yee/Desktop/LSTM/header500',
#                    help='data source for the test data')
# args = parse.parse_args()

tf.flags.DEFINE_string(name='train_file_path', default='./data/header3500',
                       help='data source for the train data')
tf.flags.DEFINE_string(name='test_file_path', default='./data/header500',
                       help='data source for the test data')
tf.flags.DEFINE_string(name='save_dir', default='./checkpoints/save',
                       help='save base dir')
tf.flags.DEFINE_float(name='dev_percentage', default=0.1, help='percentage of validation set ')
# tf.flags.DEFINE_integer(name='vector_max_length', default=15, help='length of vocabulary vector')

# model hyperparameters
tf.flags.DEFINE_integer(name='sequence_length', default=15, help='length of input sequence (x)')
tf.flags.DEFINE_integer(name='num_classes', default=5, help='number of classes')
tf.flags.DEFINE_integer(name='vocab_size', default=8000, help='vocabulary size')
tf.flags.DEFINE_integer(name='embedding_dim', default=256, help='dimensionlity of character embedding')
tf.flags.DEFINE_integer(name='num_layers', default=2, help='number of layers')
tf.flags.DEFINE_integer(name='hidden_dim', default=128, help='neural number of hidden layer')
tf.flags.DEFINE_float(name='dropout_keep_prob', default=0.8, help='dropout keep probability')
tf.flags.DEFINE_float(name='learning_rate', default=1e-3, help='learning rate')

# trainning parameters
tf.flags.DEFINE_integer(name='batch_size', default=64, help='batch size')
tf.flags.DEFINE_integer(name='num_epoches', default=200, help='number of training epoches')
tf.flags.DEFINE_integer(name='evaluate_every', default=100, help='evaluate model on dev set after this steps')
# tf.flags.DEFINE_integer(name='checkpoint_every', default=500, help='save model after this steps')
# tf.flags.DEFINE_integer(name='num_checkpoints', default=5, help='number of checkpoints to store')

FLAGS = tf.app.flags.FLAGS
# 可以从命令行接受参数：
# FLAGS._parse_flags()


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, dropout_keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: dropout_keep_prob
    }
    return feed_dict


def evaluate(x_dev, y_dev, session):
    data_len = len(x_dev)
    batch_dev = dataLoader.batch_iterator_dev(x_dev, y_dev)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch_dev, y_batch_dev in batch_dev:
        batch_len = len(x_batch_dev)
        feed_dict = {
            model.input_x: x_batch_dev,
            model.input_y: y_batch_dev,
            model.dropout_keep_prob: 1.0  # 测试集/验证集不需要dropout？
        }
        loss, accuracy, logits, cross_entropy = session.run(
            [model.loss, model.accuracy, model.logits, model.cross_entropy],
            feed_dict
        )
        total_loss += loss * batch_len
        total_acc += accuracy * batch_len

    return total_loss / data_len, total_acc / data_len  # loss个数/总个数


save_path = os.path.join(FLAGS.save_dir, 'best_validation')  # 模型存放地址


def train():
    # configuring saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # oad data
    print("loading training data...")
    start_time_load = time.time()
    x_train, y_train, x_dev, y_dev = dataLoader.load_header_training_data(data_file_dir=FLAGS.train_file_path,
                                                                          save_vocab_dir=FLAGS.save_dir,
                                                                          dev_percentage=FLAGS.dev_percentage,
                                                                          sequence_length=FLAGS.sequence_length)
    time_dif = get_time_dif(start_time_load)
    print("time used: ", time_dif)

    # create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("training and deciation...")
    start_time_train = time.time()
    total_batch = 0  # 训练的总批次
    best_acc_dev = 0.0  # 验证集最佳准确率
    last_improved = 0  # 上一次有提升的批次
    require_improvement = 15000  # 超过1500轮未提升，提前结束训练

    # 循环epoches
    # 每个epoch遍历所有batch（train set 切分 batch） 每一个batch计算 loss_train和acc_train
    # 同时使用dev set 进行evaluate，计算loss_dev和acc_dev
    # 比较 acc_dev 和best_acc_dev ,若有提升，更新best_acc_dev和last_improvement，保存模型
    #
    # 判断是否超过require_improvement
    stop_flag = False
    for epoch in range(FLAGS.num_epoches):
        print("epoch: " + str(epoch+1))
        batch_itarator = dataLoader.batch_iterator(x_train, y_train, FLAGS.batch_size, isShuffle=True)
        for x_batch, y_batch in batch_itarator:
            feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob)
            # loss_train, acc_train, fw_last, bw_last, logits, cross_entropy = session.run([model.loss, model.accuracy,
            #                                                                               model.output_fw_last,
            #                                                                               model.output_bw_last,
            #                                                                               model.logits,
            #                                                                               model.cross_entropy],
            #                                                                              feed_dict)
            loss_train, acc_train, logits, cross_entropy = session.run([model.loss, model.accuracy,
                                                                        model.logits,
                                                                        model.cross_entropy],
                                                                       feed_dict)
            if total_batch % FLAGS.evaluate_every == 0:
                # 使用dev进行evaluate
                loss_dev, acc_dev = evaluate(x_dev, y_dev, session)
                if acc_dev > best_acc_dev:
                    best_acc_dev = acc_dev
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)  # 保存模型
                    improved_str = '*'
                else:
                    improved_str = ''
            time_dif = get_time_dif(start_time_train)
            print('Iter: {0}, Train Loss: {1}, Train Acc: {2}, Val Loss: {3}, ''Val Acc: '
                  '{4}, Time: {5} {6}'
                  .format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))
            # 优化
            session.run(model.optimizer, feed_dict)

            total_batch += 1
            if total_batch > require_improvement:
                stop_flag = True
                print('no optimization for a long time, auto-stopping...')
                break
        if stop_flag:
            break
    print('finish training')


def predict():
    # load data
    print('load testing data...')
    start_time_load = time.time()

    test_x, test_y = dataLoader.load_header_testing_data(data_file_dir=FLAGS.train_file_path,
                                                         save_vocab_dir=FLAGS.save_dir,
                                                         sequence_length=FLAGS.sequence_length)

    # create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    print('testing...')
    test_loss, test_acc = evaluate(test_x, test_y, session)
    print('Test loss: {0:>6.2}, Test acc: {1:>7.2%}'.format(test_loss, test_acc))
    batch_itarator = dataLoader.batch_iterator(test_x, test_y, False)
    all_y_pred = []
    all_prob = []
    count = 0  # concatenate() 二维空数组不能直接连接
    for x_batch, y_batch in batch_itarator:
        batch_y_pred, batch_prob = session.run([model.y_pred, model.prob],
                                               feed_dict={
                                                 model.input_x: x_batch,
                                                 model.dropout_keep_prob: 1.0
                                             })
        all_y_pred = np.concatenate(0, [all_y_pred, batch_y_pred])
        if count == 0:
          all_prob = batch_prob
        else:
            all_prob = np.concatenate(all_prob, batch_prob)
        count = 1

    # evaluate
    # y_test = np.argmax(y_test, axis=1)
    print('precision, recall, F1 score')
    print(metrics.classification_report(y_pred=all_y_pred, y_true=test_y, target_names=['Introduction', 'Relaterd work',
                                                                                        'Methods', 'Experiment',
                                                                                        'Conclusion']))
    # write probability to csv
    out_dir = os.path.join(FLAGS.save_dir, 'predict_probs')
    print('saving predict probabilities to {0}'.format(out_dir))
    with open(out_dir,'w') as f:
        csv.writer(f).writerows(all_prob)

    time_dif = get_time_dif(start_time_load)
    print('time used:', time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("Please input: python3 runLSTM.py [train/test]")

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
    model = BiLSTM(sequence_length=FLAGS.sequence_length,
                   num_classes=FLAGS.num_classes,
                   vocab_size=FLAGS.vocab_size,
                   embedding_dim=FLAGS.embedding_dim,
                   num_layers=FLAGS.num_layers,
                   hidden_dim=FLAGS.hidden_dim,
                   learning_rate=FLAGS.learning_rate)
    if sys.argv[1] == 'train':
        train()
        # print('this for trian()')
    else:
        predict()
        # print('this is for test()')

