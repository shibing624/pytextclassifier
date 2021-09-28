# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.svm import SVC


def search_cv(x_train, y_train, x_test, y_test, model=GradientBoostingClassifier(n_estimators=30)):
    # grid search找到最好的参数
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 4], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
    clf = GridSearchCV(model, param_grid=parameters)
    grid_search = clf.fit(x_train, y_train)
    # 对结果打分
    print("Best score: %0.3f" % grid_search.best_score_)
    print(grid_search.best_estimator_)

    # best prarams
    print('best prarams:', clf.best_params_)

    print('-----grid search end------------')
    print('on all train set')
    scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
    print(scores.mean(), scores)
    print('on test set')
    scores = cross_val_score(grid_search.best_estimator_, x_test, y_test, cv=3, scoring='accuracy')
    print(scores.mean(), scores)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5), n_jobs=1, figure_path=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(figure_path)
    return plt


if __name__ == "__main__":
    from models.feature import Feature
    from models.reader import data_reader

    data_content, data_lbl = data_reader('../data/training_new_seg_10k.txt', '\t')
    # init feature
    feature = Feature(data_content, feature_type='tfidf_word', feature_vec_path='../output/temp')
    # get data feature
    data_feature = feature.get_feature()
    # label
    data_label = feature.label_encoder(data_lbl)
    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.2)
    search_cv(X_train, y_train, X_val, y_val, model=SVC())

    # test plot_learning_curve
    title = "Learning Curves (Random Forest, n_estimators = 30)"
    estimator = SVC()
    plot_learning_curve(estimator, title, X_train, y_train, cv=5, n_jobs=4,
                        figure_path='../output/curve.png')
    plt.show()
