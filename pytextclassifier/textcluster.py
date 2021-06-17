# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from codecs import open

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from pytextclassifier import config
from pytextclassifier.cluster import read_words, show_plt
from pytextclassifier.utils.data_utils import save_pkl, load_pkl
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.tokenizer import Tokenizer


class TextCluster(object):
    def __init__(self, model_name='kmeans', tokenizer=None, stopwords_path=None):
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.stopwords = read_words(stopwords_path) if stopwords_path else read_words(config.stop_words_path)
        self.model = None
        self.vectorizer = None

    def __repr__(self):
        return 'TextCluster instance ({}, {})'.format(self.model_name, self.tokenizer)

    @staticmethod
    def load_file_data(file_path):
        """
        Load text file, format(txt): text
        :param file_path: str
        :return: list, text list
        """
        contents = []
        if not os.path.exists(file_path):
            raise ValueError('file not found. path: {}'.format(file_path))
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    contents.append(line)
        logger.info('load file done. path: {}, size: {}'.format(file_path, len(contents)))
        return contents

    def _encode_data(self, X):
        """
        Encoding input text
        :param X: list of text, eg: [text1, text2, ...]
        :return: X, X_tokens
        """
        # tokenize text
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in X]
        logger.debug('data tokens top 1: {}'.format(X_tokens[:1]))
        return X, X_tokens

    def train(self, X, n_clusters=3):
        """
        Train model
        :param X: list of text, eg: [text1, text2, ...]
        :param n_clusters: int
        :return: model
        """
        logger.debug('train model')
        X, X_tokens = self._encode_data(X)
        vectorizer = TfidfVectorizer(analyzer='word', max_df=0.9, min_df=0.1,
                                     ngram_range=(1, 2), smooth_idf=True, sublinear_tf=True)
        X_vec = vectorizer.fit_transform(X_tokens)
        self.vectorizer = vectorizer
        # build model
        if self.model_name in ['kmeans', 'k-means', 'k-means++']:
            model = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100,
                                    n_init=10, max_no_improvement=10, verbose=0)
            # fit cluster
            model.fit(X_vec)
            # logger.debug(model.cluster_centers_)
            labels = model.labels_
            logger.debug('cluster labels:{}'.format(labels))
        else:
            raise ValueError("model name set wrong.")
        self.model = model
        return model, X_vec, labels

    def predict(self, X):
        """
        Predict label
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if self.model is None:
            raise ValueError('model is None, run train first.')
        # tokenize text
        X, X_tokens = self._encode_data(X)
        # transform
        X_vec = self.vectorizer.transform(X_tokens)
        return self.model.predict(X_vec)

    def show_clusters(self, feature_matrix, labels, image_file='cluster.png'):
        """
        Show cluster plt image
        :param feature_matrix:
        :param labels:
        :param image_file:
        :return:
        """
        if self.model is None:
            raise ValueError('model is None, run train first.')
        show_plt(feature_matrix, labels, image_file)

    def save(self, model_dir=''):
        """
        Save model to model_dir
        :param model_dir: path
        :return: None
        """
        if self.model is None:
            raise ValueError('model is None, run train first.')
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        vectorizer_path = os.path.join(model_dir, 'cluster_vectorizer.pkl')
        save_pkl(self.vectorizer, vectorizer_path)
        model_path = os.path.join(model_dir, 'cluster_model.pkl')
        save_pkl(self.model, model_path)
        logger.info('save done. vec path: {}, model path: {}'.format(vectorizer_path, model_path))

    def load(self, model_dir=''):
        """
        Load model from model_dir
        :param model_dir: path
        :return: None
        """
        model_path = os.path.join(model_dir, 'cluster_model.pkl')
        if not os.path.exists(model_path):
            raise ValueError("model is not found. please train and save model first.")
        self.model = load_pkl(model_path)
        vectorizer_path = os.path.join(model_dir, 'cluster_vectorizer.pkl')
        self.vectorizer = load_pkl(vectorizer_path)
        logger.info('model loaded from {}'.format(model_dir))
