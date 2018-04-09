# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2

from utils.io_utils import load_pkl, dump_pkl


def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """
    count_vector = CountVectorizer()
    return count_vector.fit_transform(files_data)


def tfidf(data_set, space_path, overwrite=False):
    """
    Get TFIDF value
    :param data_set:
    :param space_path:
    :return:
    """
    if os.path.exists(space_path) and not overwrite:
        tfidf_space = load_pkl(space_path)
    else:
        bow = bagOfWords(data_set)
        transformer = TfidfTransformer()
        tfidf_space = transformer.fit_transform(bow)
        dump_pkl(tfidf_space, space_path)
    print('tfidf shape:', tfidf_space.shape)
    return tfidf_space


def label_encoder(labels):
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    return corpus_encode_label


def select_best_feature(data_set, data_lbl):
    ch2 = SelectKBest(chi2, k=10000)
    return ch2.fit_transform(data_set, data_lbl), ch2
