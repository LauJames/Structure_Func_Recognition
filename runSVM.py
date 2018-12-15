# -*- coding:utf8 -*-
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from data import dataLoader_dt

data_file_dir = './data/header3500'
data_test_file_dir = './data/header500'
save_vacab_dir = './svm/save'

def SVM():
    if not os.path.exists(save_vacab_dir):
        os.makedirs(save_vacab_dir)
    x_train, y_train, _, _ = dataLoader_dt.load_header_training_data(data_file_dir, save_vacab_dir,
                                                                             0, 15)
    x_test, y_test, _, _ = dataLoader_dt.load_header_training_data(data_test_file_dir, save_vacab_dir,
                                                                   0, 15)
    # parameters = [
    #     {
    #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'kernel': ['rbf']
    #     },
    #     {
    #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #         'kernel': ['linear']
    #     }]
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
    # clf.fit(x_train, y_train)
    # print(clf.best_params_)
    # y_pred = best_model = clf.best_estimator_
    # y_pred_train = best_model.predict(x_test)

    clf = svm.SVC(C=1, gamma='scale', decision_function_shape='ovo')
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    print('train_predict:')
    print(np.mean(y_pred_train == y_train))
    print('test_predict:')
    print(np.mean(y_pred == y_test))
    print(classification_report(y_test, y_pred, labels=[1, 2, 3, 4, 5],
                                target_names=['Introduction', 'Relaterd work',
                                              'Methods', 'Experiment', 'Conclusion']))
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    SVM()