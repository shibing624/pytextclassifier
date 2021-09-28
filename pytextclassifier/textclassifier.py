# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pytextclassifier.utils.data_utils import save_pkl, load_pkl
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.tokenizer import Tokenizer
from pytextclassifier.preprocess import read_stopwords

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')


class TextClassifier:
    def __init__(self, model=None, tokenizer=None, vectorizer=None, stopwords_path=None,
                 solver='lbfgs', fit_intercept=False, ngram_range=(1, 2), **kwargs):
        """
        Init instance
        :param model: sklearn model
        :param tokenizer: word segmentation
        :param vectorizer: sklearn vectorizer
        """
        self.model = model if model else LogisticRegression(solver=solver, fit_intercept=fit_intercept)
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(ngram_range=ngram_range, **kwargs)
        self.stopwords = read_stopwords(stopwords_path) if stopwords_path else read_stopwords(default_stopwords_path)
        self.is_trained = False

    def __repr__(self):
        return 'TextClassifier instance ({}, {}, {})'.format(self.model, self.tokenizer, self.vectorizer)

    def encode_data(self, data_list_or_filepath, header=None, names=None, delimiter=',', **kwargs):
        """
        Encoding data_list text
        data_list_or_filepath: list of (label, text), eg: [(label, text), (label, text) ...]
        return: X, X_tokens, Y
        """
        if names is None:
            names = ['label', 'text']
        if isinstance(data_list_or_filepath, list):
            data_df = pd.DataFrame(data_list_or_filepath, columns=names)
        elif isinstance(data_list_or_filepath, str) and os.path.exists(data_list_or_filepath):
            data_df = pd.read_csv(data_list_or_filepath, header=header, delimiter=delimiter, names=names, **kwargs)
        else:
            raise TypeError('should be list or file path, eg: [(label, text), ... ]')
        X, Y = data_df['text'], data_df['label']
        logger.debug('loaded data list, X size: {}, y size: {}'.format(len(X), len(Y)))
        assert len(X) == len(Y)
        logger.debug('num_classes:%d' % len(set(Y)))
        # tokenize text
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in X]
        logger.debug('data tokens top 1: {}'.format(X_tokens[:1]))
        return X, X_tokens, Y

    def train(self, data_list, **kwargs):
        """
        Train model
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: model
        """
        logger.debug('train model...')
        X_train, X_train_tokens, Y_train = self.encode_data(data_list, **kwargs)
        X_train_vec = self.vectorizer.fit_transform(X_train_tokens)
        # fit
        self.model.fit(X_train_vec, Y_train)
        self.is_trained = True
        logger.debug('train model done')

    def test(self, data_list, **kwargs):
        """
        Test model with data
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: acc score
        """
        logger.debug('test model...')
        if not self.is_trained:
            raise ValueError('please train model first.')
        X_test, X_test_tokens, Y_test = self.encode_data(data_list, **kwargs)
        X_test_vec = self.vectorizer.transform(X_test_tokens)
        Y_predict = self.model.predict(X_test_vec)
        acc_score = simple_evaluate(Y_test, Y_predict)
        logger.debug('test model done')
        return acc_score

    def predict_proba(self, X):
        """
        Predict proba
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, accuracy score
        """
        if isinstance(X, str) or not hasattr(X, '__len__'):
            raise ValueError('input X should be list, eg: [text1, text2, ...]')
        if self.is_trained is None:
            raise ValueError('please train model first.')
        # tokenize text
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in X]
        # transform
        X_vec = self.vectorizer.transform(X_tokens)
        return self.model.predict_proba(X_vec)

    def predict(self, X):
        """
        Predict label
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if isinstance(X, str) or not hasattr(X, '__len__'):
            raise ValueError('input X should be list, eg: [text1, text2, ...]')
        if not self.is_trained:
            raise ValueError('please train model first.')
        # tokenize text
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in X]
        # transform
        X_vec = self.vectorizer.transform(X_tokens)
        return self.model.predict(X_vec)

    def save(self, model_dir=''):
        """
        Save model to model_dir
        :param model_dir: path
        :return: None
        """
        if not self.is_trained:
            raise ValueError('please train model first.')
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
        save_pkl(self.vectorizer, vectorizer_path)
        model_path = os.path.join(model_dir, 'classifier_model.pkl')
        save_pkl(self.model, model_path)
        logger.info('save done. vec path: {}, model path: {}'.format(vectorizer_path, model_path))

    def load(self, model_dir=''):
        """
        Load model from model_dir
        :param model_dir: path
        :return: None
        """
        model_path = os.path.join(model_dir, 'classifier_model.pkl')
        if not os.path.exists(model_path):
            raise ValueError("model is not found. please train and save model first.")
        self.model = load_pkl(model_path)
        vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
        self.vectorizer = load_pkl(vectorizer_path)
        self.is_trained = True
        logger.info('model loaded {}'.format(model_dir))
