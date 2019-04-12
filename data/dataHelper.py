#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : StructureFuncRecognition 
@File     : dataHelper.py
@Time     : 2017/12/26 10:38
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import numpy as np
import re
import codecs
import tensorflow as tf
import os
import nltk
import jieba
from sklearn.model_selection import train_test_split


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string:
    :return: string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index: end_index]


def get_para_label(file_path):
    """
    load data from files, splits the data into words and labels
    :return: paras, labels
    """
    paras = []
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("Data loaded successfully!")
                paras = [clean_str(paragraph) for paragraph in paras]
                return [paras, np.array(labels)]
            tmp = line.strip().split('\t')[-2:]
            label, para = int(tmp[0]), tmp[1]
            if label == 1:
                labels.append([1, 0, 0, 0, 0])
            elif label == 2:
                labels.append([0, 1, 0, 0, 0])
            elif label == 3:
                labels.append([0, 0, 1, 0, 0])
            elif label == 4:
                labels.append([0, 0, 0, 1, 0])
            else:
                labels.append([0, 0, 0, 0, 1])
            paras.append(para)


def get_header(file_path):
    """
    load header data from files, splits the data into words and labels
    :param file_path:
    :return: headers, labels
    """
    headers = []
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("Data loaded successfully!")
                headers = [clean_str(str(header)) for header in headers]
                return [headers, np.array(labels)]
            tmp = line.strip().split('\t')[-2:]
            header, label = tmp[0], int(tmp[1])
            if label == 1:
                labels.append([1, 0, 0, 0, 0])
            elif label == 2:
                labels.append([0, 1, 0, 0, 0])
            elif label == 3:
                labels.append([0, 0, 1, 0, 0])
            elif label == 4:
                labels.append([0, 0, 0, 1, 0])
            else:
                labels.append([0, 0, 0, 0, 1])
            headers.append(header)


def get_section(file_path):
    """
    load header data from files, splits the data into words and labels
    :param file_path:
    :return: headers, labels
    """
    sections = []
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("Data loaded successfully!")
                sections = [clean_str(str(section)) for section in sections]
                return [sections, np.array(labels)]
            tmp = line.strip().split('\t')[-2:]
            label, section = int(tmp[0]), tmp[1]
            if label == 1:
                labels.append([1, 0, 0, 0, 0])
            elif label == 2:
                labels.append([0, 1, 0, 0, 0])
            elif label == 3:
                labels.append([0, 0, 1, 0, 0])
            elif label == 4:
                labels.append([0, 0, 0, 1, 0])
            else:
                labels.append([0, 0, 0, 0, 1])
            sections.append(section)


def batch_iter_eval(x, y, batch_size=32):
    """生成随机批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def load_header_data(data_file, dev_sample_percentage, save_vocab_dir, max_length):
    x_text, y = get_header(data_file)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_length)
    # 神器，填充到最大长度
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Write vocabulary
    vocab_processor.save(os.path.join(save_vocab_dir, "vocab"))

    # Randomly shuffle data

    # Split train/test set
    # TODO: use k-fold cross validation
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=dev_sample_percentage, random_state=22)

    del x, y

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def load_section_data(data_file, dev_sample_percentage, save_vocab_dir, max_length):
    x_text, y = get_section(data_file)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_length)
    # 神器，填充到最大长度
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Write vocabulary
    vocab_processor.save(os.path.join(save_vocab_dir, "vocab"))

    # Randomly shuffle data

    # Split train/test set
    # TODO: use k-fold cross validation
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=dev_sample_percentage, random_state=22)

    del x, y

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def load_data(data_file, dev_sample_percentage, save_vocab_dir, max_length):
    x_text, y = get_para_label(data_file)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_length)
    # 神器，填充到最大长度
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Write vocabulary
    vocab_processor.save(os.path.join(save_vocab_dir, "vocab"))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: use k-fold cross validation

    # old methods
    # dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=dev_sample_percentage)

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def batch_iter_per_epoch(x, y, batch_size=64):
    """生成批次数据,每个epoch"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def word_iter(datasets, language='En'):
    """对数据集每条数据进行分词，yield迭代每个词"""

    if datasets is not None:
        if language == 'En':
            for sample in datasets:
                for token in nltk.word_tokenize(sample):
                    yield token
        elif language == 'CN':
            for sample in datasets:
                for token in jieba.cut(sample):
                    yield token
        else:
            raise NotImplementedError('Not support language!')
