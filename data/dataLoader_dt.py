import codecs
import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import os
from sklearn.model_selection import train_test_split


# 读入文件
# 预处理
# 建立词典
# 划分数据集：train dev
# batch iterator
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
            label, para = int(tmp[0]), tmp[1]   #####label保存为string
            labels.append(label)
            paras.append(para)

def load_para_data(data_file, save_vocab_dir,dev_sample_percentage, max_length):
    x_text, y = get_para_label(data_file)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_length)
    # 神器，填充到最大长度
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Write vocabulary
    vocab_processor.save(os.path.join(save_vocab_dir, "para_vocab"))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: use k-fold cross validation

    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=dev_sample_percentage)

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def get_headerFile(file_path):
    x_headers = []
    y_labels = []
    with codecs.open(filename=file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print('Data loaded successful!')
                x_headers = [clean_str(str(header)) for header in x_headers]
                y_labels = np.array(y_labels)
                return [x_headers, y_labels]
            # line = '10.1016/j.compgeo.2012.01.008	 Introduction	1'
            tmp = line.strip().split('\t')[-2:]
            header,label = tmp[0], int(tmp[1])
            x_headers.append(header)
            y_labels.append(label)

def load_header_training_data(data_file_dir, save_vocab_dir, dev_percentage, sequence_length):
    x_header, y = get_headerFile(data_file_dir)

    # 构造词典
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor(max_document_length=sequence_length)
    #  fit-词典 transform-padding fit_transform词典+padding
    x = np.array(list(vocab_processor.fit_transform(x_header)))

    # 保存词典
    vocab_processor.save(os.path.join(save_vocab_dir, 'vocab_train'))

    #randomly shuffle data ??????????????x[shuffle_indices]
    np.random.seed(1)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split dev/train set
    # old methods  可以用pickle存储数据
    # dev_sample_index = int(dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    # 使用包sklearn
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=dev_percentage, random_state=1)

    del x, y
    print('training vocabulary size is {:d}'.format(len(vocab_processor.vocabulary_)))
    print('Train/Dev split: {:d}/{:d}'.format(len(y_train),len(y_dev)))

    return x_train, y_train, x_dev, y_dev

def load_header_testing_data(data_file_dir, save_vocab_dir, sequence_length):
    x_header, y_test = get_headerFile(data_file_dir)

    # 构造词典
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor(max_document_length=sequence_length)
    #  fit-词典 transform-padding fit_transform词典+padding
    x_test = np.array(vocab_processor.fit_transform(x_header))

    # 保存词典
    vocab_processor.save(os.path.join(save_vocab_dir, 'vocab_test'))
    print('testing vocabulary size is {:d}'.format(len(vocab_processor.vocabulary_)))

    return x_test, y_test

# generate batch data
def batch_iterator(x, y, batch_size=64, isShuffle=True):
    data_lenth = len(y)
    num_batch = int((data_lenth-1) /batch_size) + 1

    # train 需要shuffle  test不需要
    if isShuffle:
        shuffle_indices = np.random.permutation(np.arange(data_lenth))
        x = x[shuffle_indices]
        y = y[shuffle_indices]

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i+1) * batch_size, data_lenth)
        yield x[start_index: end_index], y[start_index: end_index]