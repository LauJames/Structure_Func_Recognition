#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
deal with the imbalance of dataset

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition
@File     : oversample.py
@Time     : 2018/2/26 19:22
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import codecs
import numpy as np
from collections import Counter


def load_rawdata(file_path):
    """
    load the data
    :param file_path:
    :return: lines, labels
    """
    lines = []
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("processing successfully!")
                return [lines, np.array(labels)]
            tmp = line.strip().split('\t')
            label = int(tmp[2])
            labels.append(label)
            lines.append(tmp)


def oversample(file_path):
    """
    oversample the dataset
    :param file_path:
    :return: x_resampled
    """
    x, y = load_rawdata(file_path)
    x = np.array(x)
    labels_num = np.unique(y)
    # print(labels_num)
    stats_class = {}
    maj_n = 0
    for i in labels_num:
        num_class = sum(y == i)
        stats_class[i] = num_class
        if num_class > maj_n:
            maj_n = num_class
            maj_class = i
    # print(stats_class.keys())
    # keep the majority class
    x_resampled = x[y == maj_class]

    # loop over the other classes over picking at random
    for key in stats_class.keys():
        # If this is the majority class, skip it
        if key == maj_class:
            continue
        # Define the number of sample to create
        num_samples = int(stats_class[maj_class]-stats_class[key])

        indx = np.random.randint(low=0, high=stats_class[key], size=num_samples)
        # print(indx)
        x_resampled = np.vstack([x_resampled, x[y == key], x[y == key][indx]])

    label_list = x_resampled[:, 2]
    # print(np.unique(label_list))
    c = Counter()
    for label in label_list:
        c[label] = c[label] + 1
    print('各类频次统计：' + str(c))
    return x_resampled


if __name__ == "__main__":
    line_resampled = oversample('../cs.para.labelled')
    f = codecs.open("cs.para.labelled.resampled", 'w+', 'utf-8')
    for row in line_resampled:
        for col in row:
            f.write(str(col)+'\t')
        f.write('\n')
    f.close()
