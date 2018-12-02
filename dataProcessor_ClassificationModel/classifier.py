# -*- coding: utf-8 -*-

from time import time
import json
import pickle
import numpy as np
from scipy import sparse, io
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import classification_report


# 感知机
def perceptron(type, x, y):
    from sklearn.linear_model import Perceptron
    model = Perceptron(max_iter=50)
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(x, y):
        t = time()
        train_x = x[train_index]
        train_y = np.array(y)[train_index]
        test_x = x[test_index]
        test_y = np.array(y)[test_index]
        clone_model = clone(model)
        clone_model.fit(train_x, train_y)
        y_true = test_y
        y_pred = clone_model.predict(test_x)
        ans = classification_report(y_true, y_pred)
        print(ans)
        with open('report/' + type + '/perceptron', 'a+') as fp:
            fp.write(ans)
            fp.write(str(time()-t) + '\n\n')
    with open('model/' + type + 'Model/perceptron', 'wb') as fp:
        pickle.dump(clone_model, fp)


# 逻辑回归
def logistic_regression(type, x, y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(x, y):
        t = time()
        train_x = x[train_index]
        train_y = np.array(y)[train_index]
        test_x = x[test_index]
        test_y = np.array(y)[test_index]
        clone_model = clone(model)
        clone_model.fit(train_x, train_y)
        y_true = test_y
        y_pred = clone_model.predict(test_x)
        ans = classification_report(y_true, y_pred)
        print(ans)
        with open('report/' + type + '/logistic_regression', 'a+') as fp:
            fp.write(ans)
            fp.write(str(time() - t) + '\n\n')
    with open('model/' + type + 'Model/logistic_regression', 'wb') as fp:
        pickle.dump(clone_model, fp)


# 决策树
def decision_tree(type, x, y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    kf =StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(x, y):
        t = time()
        train_x = x[train_index]
        train_y = np.array(y)[train_index]
        test_x = x[test_index]
        test_y = np.array(y)[test_index]
        clone_model = clone(model)
        clone_model.fit(train_x, train_y)
        y_true = test_y
        y_pred = clone_model.predict(test_x)
        ans = classification_report(y_true, y_pred)
        print(ans)
        with open('report/' + type + '/decision_tree', 'a+') as fp:
            fp.write(ans)
            fp.write(str(time() - t) + '\n\n')
    with open('model/' + type + 'Model/decision_tree', 'wb') as fp:
        pickle.dump(clone_model, fp)


# 高斯朴素贝叶斯
def gaussianNB(type, x, y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(x, y):
        t = time()
        train_x = x[train_index]
        train_y = np.array(y)[train_index]
        test_x = x[test_index]
        test_y = np.array(y)[test_index]
        clone_model = clone(model)
        clone_model.fit(train_x, train_y)
        y_true = test_y
        y_pred = clone_model.predict(test_x)
        ans = classification_report(y_true, y_pred)
        print(ans)
        with open('report/' + type + '/gaussianNB', 'a+') as fp:
            fp.write(ans)
            fp.write(str(time() - t) + '\n\n')
    with open('model/' + type + 'Model/gaussianNB', 'wb') as fp:
        pickle.dump(clone_model, fp)


# # GBDT（Gradient Boosting Tree)梯度提升树
# def gradient_boosting(train_x, lable_y):
#     from sklearn.ensemble import GradientBoostingClassifier
#     model = GradientBoostingClassifier(n_estimators=266)
#     scoring = ['accuracy', 'precision', 'recall', 'f1']
#     scores = cross_validate(model, train_x, lable_y, scoring=scoring, cv=5)
#     with open('model/gbdt', 'wb') as fp:
#         pickle.dump(model, fp)
#     print('GBDT:', scores)

# SVM
def svm(type, x, y):
    from sklearn import svm
    model = svm.SVC(kernel='rbf')
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(x, y):
        t = time()
        train_x = x[train_index]
        train_y = np.array(y)[train_index]
        test_x = x[test_index]
        test_y = np.array(y)[test_index]
        clone_model = clone(model)
        clone_model.fit(train_x, train_y)
        y_true = test_y
        y_pred = clone_model.predict(test_x)
        ans = classification_report(y_true, y_pred)
        print(ans)
        with open('report/' + type + '/svm', 'a+') as fp:
            fp.write(ans)
            fp.write(str(time() - t) + '\n\n')
    with open('model/' + type + 'Model/svm', 'wb') as fp:
        pickle.dump(clone_model, fp)


if '__main__' == __name__:
    # raw
    with open('data/raw', 'rb') as fp:
        coo_matrix = io.mmread(fp)
    x = sparse.coo_matrix((coo_matrix.data, (coo_matrix.row, coo_matrix.col))).todense()
    with open('data/lable', 'r') as fp:
        y = json.load(fp)
    perceptron('raw', x, y)
    logistic_regression('raw', x, y)
    decision_tree('raw', x, y)
    gaussianNB('raw', x, y)
    # svm('raw', x, y)

    # pca
    with open('data/pca', 'rb') as fp:
        x = io.mmread(fp)
    with open('data/lable', 'r') as fp:
        y = json.load(fp)
    perceptron('pca', x, y)
    logistic_regression('pca', x, y)
    decision_tree('pca', x, y)
    gaussianNB('pca', x, y)
    # svm('pca', x, y)

    # nmf
    with open('data/nmf', 'rb') as fp:
        x = io.mmread(fp)
    with open('data/lable', 'r') as fp:
        y = json.load(fp)
    perceptron('nmf', x, y)
    logistic_regression('nmf', x, y)
    decision_tree('nmf', x, y)
    gaussianNB('nmf', x, y)
    # svm('nmf', x, y)


