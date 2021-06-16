# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class XGBLR(object):
    """
    xgboost as feature transform
    xgboost's output as the input feature of LR
    """
    def __init__(self, model_save_path=''):
        import xgboost as xgb
        self.lr_clf = LogisticRegression()
        self.one_hot_encoder = OneHotEncoder()
        self.xgb_clf = xgb.XGBClassifier()
        self.xgb_eval_metric = 'mlogloss'
        self.model_save_path = model_save_path
        self.init = False

    def fit(self, train_x, train_y):
        """
        train a xgboost_lr model
        :param train_x:
        :param train_y:
        :return:
        """
        from xgboost import DMatrix
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

        model = [self.xgb_clf, self.lr_clf, self.one_hot_encoder]
        # dump xgboost+lr model
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model, f, True)

    def load_model(self):
        if not self.model_save_path:
            print("model save path must be not null. please fit model first.")
        with open(self.model_save_path, 'rb') as f:
            model = pickle.load(f)
            self.xgb_clf = model[0]
            self.lr_clf = model[1]
            self.one_hot_encoder = model[2]

    def predict(self, test_x):
        """
        返回标签
        :param test_x:
        :return:
        """
        from xgboost import DMatrix
        if not self.init:
            self.load_model()
        test_x_mat = DMatrix(test_x)
        xgb_pred_mat = self.xgb_clf.get_booster().predict(test_x_mat, pred_leaf=True)

        lr_feature = self.one_hot_encoder.transform(xgb_pred_mat)
        return self.lr_clf.predict(lr_feature)

    def predict_proba(self, test_x):
        """
        返回标签及其概率值
        :param test_x: X : array-like, shape = [n_samples, n_features]
        :return: T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model
        """
        from xgboost import DMatrix
        if not self.init:
            self.load_model()
        test_x_mat = DMatrix(test_x)
        xgb_pred_mat = self.xgb_clf.get_booster().predict(test_x_mat, pred_leaf=True)

        lr_feature = self.one_hot_encoder.transform(xgb_pred_mat)
        return self.lr_clf.predict_proba(lr_feature)

    def evaluate(self, right_labels, pred_labels, ignore_label=None):
        """
        simple evaluate
        :param right_labels: right labels
        :param pred_labels: predict labels
        :return: pre, rec, f
        """
        assert len(pred_labels) == len(right_labels)
        pre_pro_labels, pre_right_labels = [], []
        rec_pro_labels, rec_right_labels = [], []
        labels_len = len(pred_labels)
        for i in range(labels_len):
            pro_label = pred_labels[i]
            if pro_label != ignore_label:  #
                pre_pro_labels.append(pro_label)
                pre_right_labels.append(right_labels[i])
            if right_labels[i] != ignore_label:
                rec_pro_labels.append(pro_label)
                rec_right_labels.append(right_labels[i])
        pre_pro_labels, pre_right_labels = np.array(pre_pro_labels, dtype='int32'), \
                                           np.array(pre_right_labels, dtype='int32')
        rec_pro_labels, rec_right_labels = np.array(rec_pro_labels, dtype='int32'), \
                                           np.array(rec_right_labels, dtype='int32')
        pre = 0. if len(pre_pro_labels) == 0 \
            else len(np.where(pre_pro_labels == pre_right_labels)[0]) / float(len(pre_pro_labels))
        rec = len(np.where(rec_pro_labels == rec_right_labels)[0]) / float(len(rec_right_labels))
        f = 0. if (pre + rec) == 0. \
            else (pre * rec * 2.) / (pre + rec)
        print('P:', pre, '\tR:', rec, '\tF:', f)
        return pre, rec, f

    def score(self, test_data, test_labels):
        pred_labels = self.predict(test_data)
        pre, rec, f = self.evaluate(test_labels, pred_labels)
        return pre
