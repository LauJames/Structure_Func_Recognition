#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : testSVM.py
@Time     : 18-12-14 下午5:25
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os

import tensorflow as tf

from data import dataLoader_dt

header_data_file_dir = './data/header3500'
header_data_test_file_dir = './data/header500'
section_data_file_dir = './data/section3500'
section_data_test_file_dir = './data/section500'
paragraph_data_file_dir = './data/paragraph3500'
paragraph_data_test_file_dir = './data/paragraph500'

save_train_vacab_dir = './SVM/save/train'
save_test_vacab_dir = './SVM/save/test'

if not os.path.exists(save_train_vacab_dir):
    os.makedirs(save_train_vacab_dir)
if not os.path.exists(save_test_vacab_dir):
    os.makedirs(save_test_vacab_dir)

def test():
    x_header_test, y_header_test, x_header_train, y_header_train = dataLoader_dt.load_header_data(header_data_test_file_dir, save_test_vacab_dir,
                                                                        0.9, 15)
    # x_header_train, y_header_train, _, _ = dataLoader_dt.load_header_data(header_data_file_dir, save_train_vacab_dir, 0,
    #                                                                       15)
    # x_header_test, y_header_test, _, _ = dataLoader_dt.load_header_data(header_data_test_file_dir, save_test_vacab_dir,
    #                                                                     0, 15)
    # 词向量
    embedding_words = tf.Variable(tf.random_uniform([8000, 256], -1.0, 1.0), name='embedding_words')
    x_train = tf.nn.embedding_lookup(embedding_words, x_header_train)
    x_test = tf.nn.embedding_lookup(embedding_words, x_header_test)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    x_header_train = session.run(x_train)
    x_header_test = session.run(x_test)

#embedding_words 没有run
    # embedding_words = session.run(embedding_words)
    # x_header_train = tf.nn.embedding_lookup(embedding_words, x_header_train)
    # x_header_test = tf.nn.embedding_lookup(embedding_words, x_header_test)
    print('======')
    # print(x_header_test.shape)
    # print(x_header_train.shape)

if  __name__ == '__main__':
    test()