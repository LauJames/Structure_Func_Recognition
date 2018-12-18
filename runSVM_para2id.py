#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Structure_Func_Recognition 
@File     : runSVM.py
@Time     : 18-12-14 上午11:19
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import  numpy as np
import os
import sys
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from data import dataLoader_dt
from data import wordList
base_dir = os.getcwd()
header_data_file_dir = base_dir+'/data/header3500'
header_data_test_file_dir = base_dir+'/data/header500'
header_all_data_path = base_dir + '/data/header'

section_data_file_dir = base_dir+'/data/section3500'
section_data_test_file_dir = base_dir+'/data/section500'
section_all_data_path =base_dir+ '/data/section'

paragraph_data_file_dir = base_dir+'/data/paragraph3500'
paragraph_data_test_file_dir = base_dir+'/data/paragraph500'
paragraph_all_data_path =base_dir+ '/data/paragraph'
save_train_vacab_dir = base_dir+'/SVM/save/train'
save_test_vacab_dir = base_dir+'/SVM/save/test'

if not os.path.exists(save_train_vacab_dir):
    os.makedirs(save_train_vacab_dir)
if not os.path.exists(save_test_vacab_dir):
    os.makedirs(save_test_vacab_dir)


def svm_header():
    print('SVM decision for header...')

    x_header_train, y_header_train = wordList.para2id_header(header_data_file_dir, header_all_data_path, 4000)
    x_header_test, y_header_test = wordList.para2id_header(header_data_test_file_dir, header_all_data_path,4000)

    # 处理y
    y_header_train = np.argmax(y_header_train, 1)
    y_header_test = np.argmax(y_header_test, 1)

    clf = svm.SVC(C=10, kernel='rbf', gamma=0.001, decision_function_shape='ovo', probability=True)
    # clf = svm.NuSVC(nu=0.5, gamma=0.001, decision_function_shape='ovo')
    # clf = svm.SVC()
    print(clf)
    clf.fit(x_header_train, y_header_train)
    print("fitted...")

    y_pred = clf.predict(x_header_test)
    y_train_pred = clf.predict(x_header_train)

    print('for trainning data: ')
    # clf.score(y_header_train, y_train_pred)
    print(np.mean(y_train_pred == y_header_train))
    print(classification_report(y_header_train, y_train_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_header_train, y_train_pred))

    print('for testing data: ')
    # clf.score(y_header_test, y_pred)
    print(np.mean(y_pred == y_header_test))


    print(classification_report(y_header_test, y_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion'] ))
    print(confusion_matrix(y_header_test, y_pred))
    print(clf.predict_proba(x_header_test))


def svm_section():
    print('SVM decision for section...')
    x_section_train, y_section_train = wordList.para2id(section_data_file_dir, section_all_data_path,7500)
    x_section_test, y_section_test = wordList.para2id(section_data_test_file_dir, section_all_data_path,7500)
    # x_header_test, y_header_test, x_header_train, y_header_train = dataLoader_dt.load_header_data(header_data_file_dir, save_test_vacab_dir,
    #                                                                     0.9, 15)
    # 处理y
    y_section_train = np.argmax(y_section_train, 1)
    y_section_test = np.argmax(y_section_test, 1)

    clf = svm.SVC(C=10, kernel='rbf', gamma=0.001, decision_function_shape='ovo', probability=True)
    # clf = svm.NuSVC()
    # clf = svm.SVC()
    print(clf)
    clf.fit(x_section_train, y_section_train)
    print("fitted...")

    y_pred = clf.predict(x_section_test)
    y_train_pred = clf.predict(x_section_train)

    print('for trainning data: ')
    # clf.score(y_header_train, y_train_pred)
    print(np.mean(y_train_pred == y_section_train))
    print(classification_report(y_section_train, y_train_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_section_train, y_train_pred))

    print('for testing data: ')
    # clf.score(y_header_test, y_pred)
    print(np.mean(y_pred == y_section_test))


    print(classification_report(y_section_test, y_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion'] ))
    print(confusion_matrix(y_section_test, y_pred))

    print(clf.predict_proba(x_section_test))


def svm_paragraph():
    print('SVM decision for paragraph...')
    x_para_train, y_para_train = wordList.para2id(paragraph_data_file_dir, paragraph_all_data_path, 7500)
    x_para_test, y_para_test = wordList.para2id(paragraph_data_test_file_dir, paragraph_all_data_path, 7500)
    # x_para_test, y_para_test, x_para_train, y_para_train = dataLoader_dt.load_paragraph_data(paragraph_data_test_file_dir,
    #                                                                                       save_test_vacab_dir, 0.9, 600)
    # 处理y
    y_para_train = np.argmax(y_para_train, 1)
    y_para_test = np.argmax(y_para_test, 1)

    clf = svm.SVC(C=10, kernel='rbf', gamma=0.001, decision_function_shape='ovo', probability=True)
    # clf = svm.NuSVC()
    # clf = svm.SVC()
    print(clf)
    clf.fit(x_para_train, y_para_train)
    print("fitted...")

    y_pred = clf.predict(x_para_test)
    y_train_pred = clf.predict(x_para_train)

    print('for trainning data: ')
    # clf.score(y_header_train, y_train_pred)
    print(np.mean(y_train_pred == y_para_train))
    print(classification_report(y_para_train, y_train_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_para_train, y_train_pred))

    print('for testing data: ')
    # clf.score(y_header_test, y_pred)
    print(np.mean(y_pred == y_para_test))


    print(classification_report(y_para_test, y_pred, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion'] ))
    print(confusion_matrix(y_para_test, y_pred))

    print(clf.predict_proba(x_para_test))

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['header', 'section','paragraph']:
        raise ValueError("Please input: python3 runLSTM.py [header/section/paragraph]")
    if sys.argv[1] == 'header':
        svm_header()
    elif sys.argv[1] == 'section':
        svm_section()
    else:
        svm_paragraph()

    # print(os.getcwd()+'/data/header3500')
    # print(os.path.abspath(__file__))
    # svm_header()
    # svm_paragraph()
