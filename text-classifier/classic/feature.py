# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from utils.io_utils import dump_pkl


def tfidf(data_set, vectorizer_path=None):
    """
    Get TFIDF value
    :param data_set:
    :param vectorizer_path:
    :return:
    """
    vectorizer = TfidfVectorizer()
    data_feature = vectorizer.fit_transform(data_set)
    dump_pkl(vectorizer, vectorizer_path)
    print('data_feature shape:', data_feature.shape)
    return data_feature


def label_encoder(labels):
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    return corpus_encode_label


def select_best_feature(data_set, data_lbl):
    ch2 = SelectKBest(chi2, k=10000)
    return ch2.fit_transform(data_set, data_lbl), ch2
