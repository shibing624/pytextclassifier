# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from sklearn.model_selection import train_test_split

import config
from evaluate import eval
from feature import label_encoder
from feature import tfidf
from model import get_model
from reader import data_reader
from utils.io_utils import dump_pkl


def train(model_type, data_path=None, pr_figure_path=None,
          model_save_path=None, vectorizer_path=None, col_sep=','):
    data_content, data_lbl = data_reader(data_path, col_sep)
    # data feature
    data_tfidf = tfidf(data_content, vectorizer_path=vectorizer_path)
    # label
    data_label = label_encoder(data_lbl)
    X_train, X_val, y_train, y_val = train_test_split(
        data_tfidf, data_label, test_size=0.1, random_state=42)
    model = get_model(model_type)
    # fit
    model.fit(X_train, y_train)
    # save model
    dump_pkl(model, model_save_path)
    # evaluate
    eval(model, X_val, y_val, 0.5, pr_figure_path)


if __name__ == '__main__':
    train(config.model_type,
          config.train_seg_path,
          config.pr_figure_path,
          config.model_save_path,
          config.vectorizer_path,
          config.col_sep)
