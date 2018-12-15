#! /user/bin/evn python
# -*- coding:utf8 -*-
import codecs
import re
from collections import Counter
import numpy as np

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

    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", "", string)
    # string = re.sub(' ','',string)
    return string.strip().lower()

def get_stopwords():
    stopwords_dir = '/Users/j_yee/PycharmProjects/Structure_Func_Recognition/data/stopwords.txt'
    stopwords = []
    with codecs.open(stopwords_dir, encoding='iso-8859-1') as fp:
        while True:
            line = fp.readline().strip()
            if not line:
                # print('停用词：=======',stopwords)
                return stopwords
            stopwords.append(line)


def wordcount(str):
    # 文章字符串前期处理
    # str_list = str.replace('\n', '').lower().split(' ')
    str_list = str.replace('\n', '').split(' ')
    # print(str_list)
    str_list = [clean_str(str) for str in str_list]
    # print(str_list)
    count_dict = {}
    stopwords = get_stopwords()
    # 如果字典里有该单词则加1，否则添加入字典
    for str in str_list:
        str = clean_str(str)
        if str in stopwords:
            # print(str)
            continue
        elif str in count_dict.keys():
            count_dict[str] = count_dict[str] + 1
        else:
            count_dict[str] = 1
    #按照词频从高到低排列
    count_list=sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
    return count_list


# 直接使用counter（）
# def get_word_list(file_path):
#     paras = []
#     labels = []
#     with codecs.open(file_path, encoding='utf-8') as fp:
#         while True:
#             line = fp.readline()
#             if not line:
#                 paragraph = ''
#                 for para in paras:
#                     paragraph = paragraph + clean_str(para)
#                     # print(paras.__len__())
#                 print(paragraph)
#                 print("vacabulary dictionary builded succesfully!")
#                 c = Counter(paragraph)
#                 topN = c.most_common(5)
#                 print(topN)
#                 dictionary = []
#                 for i in range(topN.__len__()):
#                     dictionary.append(topN[i][0])
#                 return dictionary
#                 # return paragraph
#             tmp = line.strip().split('\t')[-1:]
#             para = tmp[0]
#             paras.append(para)

def get_word_list(file_path, vacab_size):
    paras = []
    labels = []
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                paragraph = ''
                for para in paras:
                    paragraph = paragraph + clean_str(para)
                # print(paragraph)
                print("vacabulary dictionary builded succesfully!")

                count = wordcount(paragraph)
                dictionary = []
                for i in range(0,vacab_size):
                    dictionary.append(count[i][0])
                    # print(count[i][0],count[i][1])
                return dictionary     # 返回词典
            tmp = line.strip().split('\t')[-1:]
            para = tmp[0]
            paras.append(para)


def para2id(file_path, vacab_size):
    para2ids = []
    labels = []
    dictionary = get_word_list(file_path, vacab_size)
    with codecs.open(file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print("Data loaded successfully!")
                return [np.array(para2ids), np.array(labels)]

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
            para2id = np.zeros(vacab_size)
            para_list = para.split(' ')
            for word in [clean_str(word) for word in para_list]:
                if word in dictionary:
                    index = dictionary.index(word)
                    para2id[index] = para2id[index] + 1
            para2ids.append(para2id)


if __name__ == '__main__':
    a = ['you','are','a', 'kind','kind','a','girl']
    print(Counter(a).most_common(3))
    # c = Counter('you  ARE a a kind girl I')
    # print(wordcount('you are a a kind girl I'))
    # print(c.most_common(3))
    # dic = get_word_list('/Users/j_yee/PycharmProjects/Structure_Func_Recognition/data/test_para',10)
    # print(dic)
    # print('beam 的 index： ', dic.index('beam'))
    # para2ids, labels = para2id('/Users/j_yee/PycharmProjects/Structure_Func_Recognition/data/test_para',10)
    # print(para2ids.shape)
    # print(para2ids)
