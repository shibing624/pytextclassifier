# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pytextclassifier.models.classic_model import get_model
from pytextclassifier.models.evaluate import simple_evaluate
from pytextclassifier.utils.data_utils import save_pkl, load_pkl
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.tokenizer import Tokenizer


class TextClassifier(object):
    def __init__(self, model_name='lr', tokenizer=None):
        """
        Init instance
        :param model_name: str, support lr, random_forest, xgboost, svm, mlp, ensemble, stack
        :param tokenizer: word segmentation
        """
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.model = None
        self.vectorizer = None

    def __repr__(self):
        return 'TextClassifier instance ({}, {})'.format(self.model_name, self.tokenizer)

    @staticmethod
    def load_file_data(file_path, delimiter=','):
        """
        Load text file, format(csv): label, text
        :param file_path: str
        :param delimiter: ,
        :return: pd.DataFrame
        """
        if not os.path.exists(file_path):
            raise ValueError('file not found. path: {}'.format(file_path))
        data = pd.read_csv(file_path, header=None, encoding='utf-8', names=['label', 'text'], delimiter=delimiter)
        logger.info('load file done. path: {}, size: {}'.format(file_path, len(data)))
        return data

    def _encode_data(self, data_list):
        """
        Encoding data_list text
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: X, X_tokens, Y
        """
        try:
            data_list = pd.DataFrame(data_list, columns=['label', 'text'])
        except Exception as e:
            logger.error(e)
            raise TypeError('data_list should be list')

        X, Y = data_list['text'], data_list['label']
        logger.debug('load data_list, X size: {}, label size: {}'.format(len(X), len(Y)))
        assert len(X) == len(Y)
        logger.debug('num_classes:%d' % len(set(Y)))
        # tokenize text
        X_tokens = [' '.join(self.tokenizer.tokenize(i)) for i in X]
        logger.debug('data tokens top 3: {}'.format(X_tokens[:3]))
        return X, X_tokens, Y

    def train(self, data_list):
        """
        Train model
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: model
        """
        logger.debug('train model')
        X_train, X_train_token, Y_train = self._encode_data(data_list)
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                     smooth_idf=True, sublinear_tf=True)
        X_train_vec = vectorizer.fit_transform(X_train_token)
        self.vectorizer = vectorizer
        # build model
        model = get_model(self.model_name)
        # fit
        model.fit(X_train_vec, Y_train)
        self.model = model
        return model

    def test(self, data_list):
        """
        Test model with data
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: acc score
        """
        logger.debug('test model')
        if self.model is None:
            raise ValueError('model is None, run train first.')
        X_test, X_test_token, Y_test = self._encode_data(data_list)
        X_test_vec = self.vectorizer.transform(X_test_token)
        Y_predict = self.model.predict(X_test_vec)
        acc_score = simple_evaluate(Y_test, Y_predict)
        return acc_score

    def predict_proba(self, X):
        """
        Predict proba
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, accuracy score
        """
        if self.model is None:
            raise ValueError('model is None, run train first.')
        # tokenize text
        X_tokens = [' '.join(self.tokenizer.tokenize(i)) for i in X]
        # transform
        X_vec = self.vectorizer.transform(X_tokens)
        return self.model.predict_proba(X_vec)

    def predict(self, X):
        """
        Predict label
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if self.model is None:
            raise ValueError('model is None, run train first.')
        # tokenize text
        X_tokens = [' '.join(self.tokenizer.tokenize(i)) for i in X]
        # transform
        X_vec = self.vectorizer.transform(X_tokens)
        return self.model.predict(X_vec)

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
        logger.info('model loaded from {}'.format(model_dir))
