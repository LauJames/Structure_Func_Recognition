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
from sklearn import svm
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  classification_report, confusion_matrix
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


def svm_header():
    print('SVM decision for header...')

    x_header_train, y_header_train, _, _ = dataLoader_dt.load_header_data(header_data_file_dir, save_train_vacab_dir, 0,
                                                                          15)
    x_header_test, y_header_test, _, _ = dataLoader_dt.load_header_data(header_data_test_file_dir, save_test_vacab_dir,
                                                                        0, 15)

    # x_header_test, y_header_test, x_header_train, y_header_train = dataLoader_dt.load_header_data(header_data_file_dir, save_test_vacab_dir,
    #                                                                     0.9, 15)

    # 词向量
    # with tf.device('/cpu:0'), tf.name_scope('embedding'):
    #     embedding_words = tf.Variable(tf.random_uniform([8000, 256], -1.0, 1.0), name='embedding_words')
    #
    #     x_train = tf.nn.embedding_lookup(embedding_words, x_header_train)
    #     x_test = tf.nn.embedding_lookup(embedding_words, x_header_test)
    #
    #     session = tf.Session()
    #     session.run(tf.global_variables_initializer())
    #
    #     x_header_train = session.run(x_train)
    #     x_header_test = session.run(x_test)
    #
    #     session.close()

    # 处理y
    y_header_train = np.argmax(y_header_train, 1)
    y_header_test = np.argmax(y_header_test, 1)

    # parameters = [
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         # 'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'C': [1, 3, 5, 7, 9],
    #         'gamma': [0.0001, 0.001, 0.1, 1],
    #         'kernel': ['rbf']
    #     },
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'C': [1, 3, 5, 7, 9],
    #         'kernel': ['linear']
    #     }]
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    # print('finish GridSearchCV...')
    # clf.fit(x_header_train, y_header_train)
    # print("fitted...")
    # print(clf.best_params_)
    # best_model = clf.best_estimator_
    #
    # y_pred = best_model.predict(x_header_test)
    # y_train_pred = best_model.predict(x_header_train)

    # clf = svm.SVC(C=0.5, kernel='rbf', gamma=0.001, decision_function_shape='ovo')
    clf = svm.NuSVC(nu=0.5, gamma=0.001, decision_function_shape='ovo')
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


def svm_section():
    print('SVM decision for section...')
    x_section_train, y_section_train, _, _ = dataLoader_dt.load_paragraph_data(section_data_file_dir, save_train_vacab_dir, 0,
                                                                          2500)
    x_section_test, y_section_test, _, _ = dataLoader_dt.load_paragraph_data(section_data_test_file_dir, save_test_vacab_dir, 0,
                                                                          2500)
    # x_header_test, y_header_test, x_header_train, y_header_train = dataLoader_dt.load_header_data(header_data_file_dir, save_test_vacab_dir,
    #                                                                     0.9, 15)
    # 处理y
    y_section_train = np.argmax(y_section_train, 1)
    y_section_test = np.argmax(y_section_test, 1)

    # parameters = [
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         # 'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'C': [1, 3, 5, 7, 9],
    #         'gamma': [0.0001, 0.001, 0.1, 1],
    #         'kernel': ['rbf']
    #     },
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'C': [1, 3, 5, 7, 9],
    #         'kernel': ['linear']
    #     }]
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    # print('finish GridSearchCV...')
    # clf.fit(x_header_train, y_header_train)
    # print("fitted...")
    # print(clf.best_params_)
    # best_model = clf.best_estimator_
    #
    # y_pred = best_model.predict(x_header_test)
    # y_train_pred = best_model.predict(x_header_train)

    # clf = svm.SVC(C=0.5, kernel='rbf', gamma=0.001, decision_function_shape='ovo')
    # clf = svm.NuSVC()
    clf = svm.SVC()
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


def svm_paragraph():
    print('SVM decision for paragraph...')
    x_para_train, y_para_train, _, _ = dataLoader_dt.load_paragraph_data(paragraph_data_file_dir, save_train_vacab_dir, 0,
                                                                          600)
    x_para_test, y_para_test, _, _ = dataLoader_dt.load_paragraph_data(paragraph_data_test_file_dir, save_test_vacab_dir, 0,
                                                                          600)
    # x_para_test, y_para_test, x_para_train, y_para_train = dataLoader_dt.load_paragraph_data(paragraph_data_test_file_dir,
    #                                                                                       save_test_vacab_dir, 0.9, 600)
    # 处理y
    y_para_train = np.argmax(y_para_train, 1)
    y_para_test = np.argmax(y_para_test, 1)

    # parameters = [
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         # 'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'C': [1, 3, 5, 7, 9],
    #         'gamma': [0.0001, 0.001, 0.1, 1],
    #         'kernel': ['rbf']
    #     },
    #     {
    #         # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'C': [1, 3, 5, 7, 9],
    #         'kernel': ['linear']
    #     }]
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    # print('finish GridSearchCV...')
    # clf.fit(x_header_train, y_header_train)
    # print("fitted...")
    # print(clf.best_params_)
    # best_model = clf.best_estimator_
    #
    # y_pred = best_model.predict(x_header_test)
    # y_train_pred = best_model.predict(x_header_train)

    # clf = svm.SVC(C=0.5, kernel='rbf', gamma=0.001, decision_function_shape='ovo')
    # clf = svm.NuSVC()
    clf = svm.SVC()
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

if __name__ == '__main__':
    # svm_header()
    svm_paragraph()