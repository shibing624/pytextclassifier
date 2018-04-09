# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_model(model_type):
    if model_type == "logistic_regression":
        model = LogisticRegression  # 快，准确率一般。val mean acc:0.769
    elif model_type == "random_forest":
        model = RandomForestClassifier  # 耗时，准确率高， val mean acc:0.962
    elif model_type == "gbdt":
        model = GradientBoostingClassifier  # 耗时，准确率低。val mean acc:0.803
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier
    elif model_type == "knn":
        model = KNeighborsClassifier
    elif model_type == "bayes":
        model = MultinomialNB
    elif model_type == "xgboost":
        model = XGBClassifier  # 快，准确率高。
    elif model_type == "svm":
        model = SVC  # 慢，准确率高。

    return model()
