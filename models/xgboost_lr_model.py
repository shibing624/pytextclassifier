# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import pickle

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix


class XGBLR(object):
    """
    xgboost as feature transform
    xgboost's output as the input feature of LR
    """

    def __init__(self, xgb_model_name, lr_model_name,
                 one_hot_model_name, xgb_eval_metric='mlogloss'):
        self.lr_clf = LogisticRegression()
        self.one_hot_encoder = OneHotEncoder()
        self.xgb_clf = xgb.XGBClassifier()
        self.xgb_model_name = xgb_model_name
        self.lr_model_name = lr_model_name
        self.one_hot_model_name = one_hot_model_name
        self.xgb_eval_metric = xgb_eval_metric
        self.init = False

    def train_model(self, train_x, train_y):
        """
        train a xgboost_lr model
        :param train_x:
        :param train_y:
        :return:
        """
        self.xgb_clf.fit(train_x, train_y, eval_metric=self.xgb_eval_metric,
                         eval_set=[(train_x, train_y)])
        xgb_eval_result = self.xgb_clf.evals_result()
        print('Xgb train eval result:', xgb_eval_result)

        train_x_mat = DMatrix(train_x)
        # get boost tree leaf info
        train_xgb_pred_mat = self.xgb_clf.get_booster().predict(train_x_mat, pred_leaf=True)
        print(train_xgb_pred_mat)

        # begin one-hot encoding
        train_lr_feature_mat = self.one_hot_encoder.fit_transform(train_xgb_pred_mat)
        print('train_mat:', train_lr_feature_mat.shape)
        print('train_mat array:', train_lr_feature_mat.toarray())

        # lr
        self.lr_clf.fit(train_lr_feature_mat, train_y)
        self.init = True

        # dump xgboost+lr model
        with open(self.xgb_model_name, 'wb') as f1, open(self.lr_model_name, 'wb') as f2, \
                open(self.one_hot_model_name, 'wb') as f3:
            pickle.dump(self.xgb_clf, f1, True)
            pickle.dump(self.lr_clf, f2, True)
            pickle.dump(self.one_hot_encoder, f3, True)

    def load_model(self):
        with open(self.xgb_model_name, 'rb') as f1, open(self.lr_model_name, 'rb') as f2, \
                open(self.one_hot_model_name, 'rb') as f3:
            self.xgb_clf = pickle.load(f1)
            self.lr_clf = pickle.load(f2)
            self.one_hot_encoder = pickle.load(f3)

    def predict(self, test_x):
        if not self.init:
            self.load_model()
        test_x_mat = DMatrix(test_x)
        xgb_pred_mat = self.xgb_clf.get_booster().predict(test_x_mat, pred_leaf=True)

        lr_feature = self.one_hot_encoder.transform(xgb_pred_mat)
        lr_pred_res = self.lr_clf.predict(lr_feature)
        return lr_pred_res
