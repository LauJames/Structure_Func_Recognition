# -*- coding:utf8 -*-

from sklearn import tree
import numpy as np
import os
import sys
import  argparse
from data import dataLoader_dt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
header_data_file_dir = './data/header3500'
header_data_test_file_dir = './data/header500'
section_data_file_dir = './data/section3500'
section_data_test_file_dir = './data/section500'
paragraph_data_file_dir = './data/paragraph3500'
paragraph_data_test_file_dir = './data/paragraph500'
# save_vacab_dir = './model_DT/save'
save_tree_dir = './model_DT/save/tree'

save_train_vacab_dir = './model_DT/save/train'
save_test_vacab_dir = './model_DT/save/test'


# parse = argparse.ArgumentParser()
# parse.add_argument(dest='data_type', help='header/section/paragraph')
# args = parse.parse_args()

if not os.path.exists(save_train_vacab_dir):
    os.makedirs(save_train_vacab_dir)
if not os.path.exists(save_test_vacab_dir):
    os.makedirs(save_test_vacab_dir)
if not os.path.exists(save_tree_dir):
    os.makedirs(save_tree_dir)


def dt_header():
    print('decision for header...')
    x_header_train, y_header_train, _, _ = dataLoader_dt.load_header_data(header_data_file_dir, save_train_vacab_dir, 0,
                                                                          15)
    x_header_test, y_header_test, _, _ = dataLoader_dt.load_header_data(header_data_test_file_dir, save_test_vacab_dir, 0,
                                                                        15)


    # param_grid = {"max_depth": [3, 15],
    #               "min_samples_split": [3, 5, 10],
    #               "min_samples_leaf": [3, 5, 10],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"],
    #               "n_estimators": range(10, 50, 10)}
    #
    # class_weights = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2, }
    # criterion = "gini",
    # splitter = "best",
    # max_depth = None,
    # min_samples_split = 2,
    # min_samples_leaf = 1,
    # min_weight_fraction_leaf = 0.,
    # max_features = None,
    # random_state = None,
    # max_leaf_nodes = None,
    # min_impurity_decrease = 0.,
    # min_impurity_split = None,
    # class_weight = None,
    # presort = False

    # clf =  tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
    #             max_features=None, max_leaf_nodes=None,
    #             min_impurity_split=1e-07, min_samples_leaf=1,
    #             min_samples_split=2, min_weight_fraction_leaf=0.0,
    #             presort=False, random_state=None, splitter='best')

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, min_samples_split=5,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0., max_features='sqrt',
                                      random_state=5, max_leaf_nodes=None, min_impurity_decrease=0,
                                      min_impurity_split=None)
    # clf = tree.DecisionTreeClassifier()

    print(clf)
    clf.fit(x_header_train, y_header_train)



    '''''测试结果的打印'''
    # x_test, y_test = dataLoader_dt.load_header_testing_data(data_test_file_dir, save_vacab_dir, squence_lenth)
    answer = clf.predict(x_header_test)
    # print(answer)
    # print(y_dev)
    # print(np.mean(answer == y_header_test))

    '''''准确率与召回率'''
    # data_train results:
    print('use training data to predict:')
    answer_train = np.argmax(clf.predict(x_header_train), 1)
    y_train = np.argmax(y_header_train, 1)
    print(np.mean(answer_train == y_train))
    print(classification_report(y_train, answer_train, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_train, answer_train))

    print('use testing data to predict:')
    answer = np.argmax(answer, 1)
    y_header_test = np.argmax(y_header_test, 1)
    print(np.mean(answer == y_header_test))

    # precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    print(classification_report(y_header_test, answer,  target_names=['Introduction', 'Relaterd work',
                                                                            'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_header_test, answer))

    # print(clf.score(y_header_train, answer_train))
    # print(clf.score(y_header_test, answer))


def dt_section():
    print('decision for section...')
    # train_vacab = os.path.join(save_train_vacab_dir, 'section_vocab')
    # test_vsacab = os.path.join(save_test_vacab_dir, 'section_vocab')
    # print(os.path.join(save_train_vacab_dir, 'section_vocab'))
    # print(os.path.join(save_test_vacab_dir, 'section_vocab'))

    x_section_train, y_section_train, _, _ = dataLoader_dt.load_section_data(section_data_file_dir, save_train_vacab_dir, 0,
                                                                             100)
    x_section_test, y_section_test, _, _ = dataLoader_dt.load_section_data(section_data_test_file_dir, save_test_vacab_dir, 0,
                                                                           100)

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, min_samples_split=5,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0., max_features='sqrt',
                                      random_state=5, max_leaf_nodes=None, min_impurity_decrease=0,
                                      min_impurity_split=None)
    print(clf)
    clf.fit(x_section_train, y_section_train)

    '''''测试结果的打印'''

    answer = clf.predict(x_section_test)
    # print(answer)
    # print(y_dev)
    # print(np.mean(answer == y_section_test))

    '''''准确率与召回率'''
    print('use training data to predict:')
    answer_train = np.argmax(clf.predict(x_section_train), 1)
    y_train = np.argmax(y_section_train, 1)
    print(np.mean(answer_train == y_train))
    print(classification_report(y_train, answer_train, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_train, answer_train))

    print('use testing data to predict:')
    answer = np.argmax(answer, 1)
    y_section_test = np.argmax(y_section_test, 1)
    print(np.mean(answer == y_section_test))

    # precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    print(classification_report(y_section_test, answer, labels=['1', '2', '3', '4', '5'],
                                target_names=['Introduction', 'Relaterd work',
                                              'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_section_test, answer))


def dt_paragraph():
    print('decision for paragraph...')

    x_paragraph_train, y_paragraph_train, _, _ = dataLoader_dt.load_paragraph_data(paragraph_data_file_dir,
                                                                                   save_train_vacab_dir, 0,
                                                                                   600)
    x_paragraph_test, y_paragraph_test, _, _ = dataLoader_dt.load_paragraph_data(paragraph_data_test_file_dir,
                                                                                 save_test_vacab_dir, 0, 600)

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',  min_samples_split=5,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0., max_features='sqrt',
                                      random_state=5, max_leaf_nodes=None, min_impurity_decrease=0,
                                      min_impurity_split=None)
    print(clf)
    clf.fit(x_paragraph_train, y_paragraph_train)

    '''''测试结果的打印'''
    answer = clf.predict(x_paragraph_test)
    # print(answer)
    # print(np.mean(answer == y_paragraph_test))

    '''''准确率与召回率'''
    print('use training data to predict:')
    answer_train = np.argmax(clf.predict(x_paragraph_train), 1)
    y_train = np.argmax(y_paragraph_train, 1)
    print(np.mean(answer_train == y_train))
    print(classification_report(y_train, answer_train, target_names=['Introduction', 'Relaterd work',
                                                                     'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_train, answer_train))

    print('use testing data to predict:')
    # precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))

    answer = np.argmax(answer, 1)
    y_paragraph_test = np.argmax(y_paragraph_test, 1)
    print(np.mean(answer == y_paragraph_test))

    print(classification_report(y_paragraph_test, answer,
                                target_names=['Introduction', 'Relaterd work', 'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_paragraph_test, answer))


if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['header', 'section', 'paragraph']:
    #     raise ValueError("Please input: python3 DecisionTree.py [header/section/paragraph]")
    #
    # if sys.argv[1] == 'header':
    #     dt_header()
    # elif sys.argv[1] == 'section':
    #     dt_section()
    # else:
    #     dt_paragraph()
    dt_section()
