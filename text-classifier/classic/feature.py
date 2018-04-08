# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
from utils.io_utils import load_pkl, dump_pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tfidf(data_set, space_path):
    if os.path.exists(space_path):
        tfidf_space = load_pkl(space_path)
    else:
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf_space = transformer.fit_transform(vectorizer.fit_transform(data_set))
        dump_pkl(tfidf_space, space_path)
    print('tfidf shape:', tfidf_space.shape)
    return tfidf_space


def label_encoder(labels):
    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    return corpus_encode_label
