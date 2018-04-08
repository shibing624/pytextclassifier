# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
from utils.io_utils import load_pkl, dump_pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing


def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = CountVectorizer()
    return count_vector.fit_transform(files_data)


def tfidf(data_set, space_path):
    """
    Get TFIDF value
    :param data_set:
    :param space_path:
    :return:
    """
    if os.path.exists(space_path):
        tfidf_space = load_pkl(space_path)
    else:
        bow = bagOfWords(data_set)
        transformer = TfidfTransformer(use_idf=True)
        tfidf_space = transformer.fit_transform(bow)
        dump_pkl(tfidf_space, space_path)
    print('tfidf shape:', tfidf_space.shape)
    return tfidf_space


def label_encoder(labels):
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    return corpus_encode_label
