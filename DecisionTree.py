from sklearn import  tree
import numpy as np
from data import dataLoader
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
data_file_dir = './data/header3500'
data_test_file_dir = './data/header500'
save_vacab_dir = './model_DT/save'
save_tree_dir = './model_DT/save/tree'

squence_lenth = 15

def decision_tree():
    x_train, y_train, x_dev, y_dev= dataLoader.load_header_training_data(data_file_dir, save_vacab_dir,
                                                                         0, squence_lenth)
    x_test, y_test, _, _ = dataLoader.load_header_training_data(data_test_file_dir, save_vacab_dir,
                                                                         0, squence_lenth)
    # param_grid = {"max_depth": [3, 15],
    #               "min_samples_split": [3, 5, 10],
    #               "min_samples_leaf": [3, 5, 10],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"],
    #               "n_estimators": range(10, 50, 10)}
    #
    class_weights = {"1":0.2, "2":0.2, "3":0.2, "4":0.2, "5":0.2, }
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
    clf = tree.DecisionTreeClassifier(criterion='entropy',splitter= 'best', max_depth=3,min_samples_split=5,
                                      min_samples_leaf=1, min_weight_fraction_leaf = 0., max_features='sqrt',
                                      random_state=5, max_leaf_nodes=None, min_impurity_decrease=0,
                                      min_impurity_split=None)
    print(clf)
    clf.fit(x_train, y_train)

    '''''测试结果的打印'''
    # x_test, y_test = dataLoader.load_header_testing_data(data_test_file_dir, save_vacab_dir, squence_lenth)
    answer = clf.predict(x_test)
    print(answer)
    # print(y_dev)
    print(np.mean(answer == y_test))

    '''''准确率与召回率'''
    # precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    print(classification_report(y_test, answer,labels=['1', '2', '3', '4', '5'], target_names=['Introduction', 'Relaterd work',
                                                                                               'Methods', 'Experiment', 'Conclusion']))


if __name__ == '__main__':
    decision_tree()