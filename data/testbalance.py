#! /user/bin/evn python
# -*- coding:utf8 -*-

import codecs
import numpy as np
from collections import Counter


def load_rawdata(file_path):
    """
    load the data
    :param file_path:
    :return: lines, labels
    """
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("processing successfully!")
                return np.array(labels)
            tmp = line.strip().split('\t')
            label = int(tmp[2])
            labels.append(label)


if __name__ == "__main__":
    labels = load_rawdata('cs.para.labelled.resampled')    
    c = Counter()
    print(labels)
    for label in labels:
        c[label] = c[label] + 1
    print("label frequencies:" + str(c))
