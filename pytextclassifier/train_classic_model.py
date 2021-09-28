# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import time

from sklearn.model_selection import train_test_split

from pytextclassifier import config
from pytextclassifier.models.classic_model import get_model
from pytextclassifier.utils.evaluate import eval, plt_history
from pytextclassifier.feature import Feature
from pytextclassifier.reader import data_reader
from pytextclassifier.models.xgboost_lr_model import XGBLR
from pytextclassifier.utils.data_utils import save_pkl, write_vocab, build_vocab, load_vocab, save_dict
from pytextclassifier.utils.log import logger


def train_classic(model_type='logistic_regression',
                  data_path='',
                  model_save_path='',
                  feature_vec_path='',
                  col_sep='\t',
                  feature_type='tfidf_word',
                  min_count=1,
                  word_vocab_path='',
                  label_vocab_path='',
                  pr_figure_path=''):
    logger.info("train classic model, model_type:{}, feature_type:{}".format(model_type, feature_type))
    # load data
    data_content, data_lbl = data_reader(data_path, col_sep)
    word_lst = []
    for i in data_content:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_vocab(word_lst, min_count=min_count, sort=True, lower=True)
    # save word vocab
    write_vocab(word_vocab, word_vocab_path)
    word_id = load_vocab(word_vocab_path)
    # label
    label_vocab = build_vocab(data_lbl)
    # save label vocab
    write_vocab(label_vocab, label_vocab_path)
    label_id = load_vocab(label_vocab_path)
    print(label_id)
    data_label = [label_id[i] for i in data_lbl]
    num_classes = len(set(data_label))
    logger.info('num_classes:%d' % num_classes)
    logger.info('data size:%d' % len(data_content))
    logger.info('label size:%d' % len(data_lbl))

    # init feature
    if feature_type in ['doc_vectorize', 'vectorize']:
        logger.error('feature type error. use tfidf_word replace.')
        feature_type = 'tfidf_word'
    feature = Feature(data=data_content, feature_type=feature_type,
                      feature_vec_path=feature_vec_path, word_vocab=word_vocab, is_infer=False)
    # get data feature
    data_feature = feature.get_feature()

    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.1, random_state=0)
    if model_type == 'xgboost_lr':
        model = XGBLR(model_save_path=model_save_path)
    else:
        model = get_model(model_type)
    # fit
    model.fit(X_train, y_train)
    # save model
    if model_type != 'xgboost_lr':
        save_pkl(model, model_save_path, overwrite=True)
    # evaluate
    eval(model, X_val, y_val, num_classes=num_classes, pr_figure_path=pr_figure_path)

    # analysis lr model
    if config.debug and model_type == "logistic_regression":
        feature_weight = {}
        word_dict_rev = sorted(word_id.items(), key=lambda x: x[1])
        for feature, index in word_dict_rev:
            feature_weight[feature] = list(map(float, model.coef_[:, index]))
        save_dict(feature_weight, config.lr_feature_weight_path)


if __name__ == '__main__':
    start_time = time.time()
    if config.model_type in ['logistic_regression', 'random_forest', 'bayes', 'decision_tree',
                             'svm', 'knn', 'xgboost', 'xgboost_lr']:
        train_classic(model_type=config.model_type,
                      data_path=config.train_seg_path,
                      model_save_path=config.model_save_path,
                      feature_vec_path=config.feature_vec_path,
                      col_sep=config.col_sep,
                      feature_type=config.feature_type,
                      min_count=config.min_count,
                      word_vocab_path=config.word_vocab_path,
                      label_vocab_path=config.label_vocab_path,
                      pr_figure_path=config.pr_figure_path)
    logger.info("spend time %s s." % (time.time() - start_time))
    logger.info("finish train.")
