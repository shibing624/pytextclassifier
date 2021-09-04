# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from codecs import open

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from pytextclassifier.cluster import show_plt
from pytextclassifier.utils.data_utils import save_pkl, load_pkl
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.tokenizer import Tokenizer
from pytextclassifier.preprocess import read_stopwords

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')


class TextCluster(object):
    def __init__(self, model=None, tokenizer=None, vectorizer=None, stopwords_path=None,
                 n_clusters=3, n_init=10, ngram_range=(1, 2),  **kwargs):
        self.model = model if model else MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(ngram_range=ngram_range, **kwargs)
        self.stopwords = read_stopwords(stopwords_path) if stopwords_path else read_stopwords(default_stopwords_path)
        self.is_trained = False

    def __repr__(self):
        return 'TextCluster instance ({}, {}, {})'.format(self.model, self.tokenizer, self.vectorizer)

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

    def encode_data(self, X):
        """
        Encoding input text
        :param X: list of text, eg: [text1, text2, ...]
        :return: X, X_tokens
        """
        # tokenize text
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in X]
        logger.debug('data tokens top 1: {}'.format(X_tokens[:1]))
        return X_tokens

    def train(self, X):
        """
        Train model
        :param X: list of text, eg: [text1, text2, ...]
        :param n_clusters: int
        :return: model
        """
        logger.debug('train model')
        X_tokens = self.encode_data(X)
        X_vec = self.vectorizer.fit_transform(X_tokens)
        # fit cluster
        self.model.fit(X_vec)
        labels = self.model.labels_
        logger.debug('cluster labels:{}'.format(labels))
        self.is_trained = True
        return X_vec, labels

    def predict(self, X):
        """
        Predict label
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if not self.is_trained:
            raise ValueError('model is None, run train first.')
        # tokenize text
        X_tokens = self.encode_data(X)
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
        if not self.is_trained:
            raise ValueError('model is None, run train first.')
        show_plt(feature_matrix, labels, image_file)

    def save(self, model_dir=''):
        """
        Save model to model_dir
        :param model_dir: path
        :return: None
        """
        if not self.is_trained:
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
        self.is_trained = True
        logger.info('model loaded from {}'.format(model_dir))
