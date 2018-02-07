#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : wordCount.py
@Time     : 2018/2/7 13:38
@Software : PyCharm
@Copyright: "Copyright (c) 2017 Lau James. All Rights Reserved"
"""

import numpy as np
import pandas as pd
import codecs
import re
from collections import Counter


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


def contentlength():
    with open("section3500", 'r', encoding='utf-8') as f:
        lenList = []
        for line in f:
            # do something with the line
            dframe = np.array(line.split('\t'))
            content = dframe[2]
            content = clean_str(content)
            content = content.split()
            # et.al
            lenList.append(len(content))
        pdstata(lenList=lenList)


def pdstata(lenList):
    c = Counter()
    for num in lenList:
        c[num] = c[num] + 1
    len_pd = pd.Series(lenList)
    meanList = len_pd.mean()
    maxList = len_pd.max()
    minList = len_pd.min()
    medianList = len_pd.median()
    countList = len_pd.count()
    quantileList = len_pd.quantile([0.25, 0.75])
    try:
        f = codecs.open('stats', 'w+', 'utf-8')
        f.write('共有' + str(countList) + '条数据 \r\n')
        f.write('长度频次统计列表：\r\n')
        f.write(str(c))
        f.write('长度均值:' + str(meanList) + '\r\n')
        f.write('长度最大值:' + str(maxList) + '\r\n')
        f.write('长度最小值:' + str(minList) + '\r\n')
        f.write('长度中位数:' + str(medianList) + '\r\n')
        f.write('1/4分位数、3/4分位数：')
        f.write(str(quantileList) + '\r\n')
    finally:
        f.close()


if __name__=="__main__":
    contentlength()
