# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pytextclassifier.models.classic_model import get_model
from pytextclassifier.utils.data_utils import save_pkl, load_pkl
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.tokenizer import Tokenizer


class TextClassifier(object):
    def __init__(self, model_name='lr', tokenizer=None):
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.model = None

    def __repr__(self):
        return 'TextClassifier instance ({}, {})'.format(self.model_name, self.tokenizer)

    def _load_file_data(self, file_path, delimiter=','):
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, header=None, encoding='utf-8', names=['label', 'text'], delimiter=delimiter)
            logger.info('load file done. path: {}, size: {}'.format(file_path, len(data)))
        else:
            data = None
            logger.error('file not found. path: {}'.format(file_path))
        return data

    def train(self, data, delimiter=','):
        logger.debug('train model')
        if isinstance(data, str):
            data = self._load_file_data(data, delimiter)
        elif isinstance(data, list):
            data = pd.DataFrame(data, columns=['label', 'text'])
        else:
            raise TypeError('data should be list or str(file path)')

        X_train, Y_train = data['text'], data['label']
        logger.info('num_classes:%d' % len(set(Y_train)))
        logger.info('data size:%d' % len(X_train))
        logger.info('label size:%d' % len(Y_train))
        # encode train data text
        X_train_token = [' '.join(self.tokenizer.tokenize(i)) for i in X_train]
        logger.debug('train tokens top 3: {}'.format(X_train_token[:3]))
        vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=True, use_idf=True, norm='l1')
        X_train_vec = vectorizer.fit_transform(X_train_token)
        self.vectorizer = vectorizer
        # build model
        model = get_model(self.model_name)
        # fit
        model.fit(X_train_vec, Y_train)
        self.model = model
        return model

    def test(self, data, delimiter=','):
        if isinstance(data, str):
            data = self._load_file_data(data, delimiter)
        elif isinstance(data, list):
            data = pd.DataFrame(data, columns=['label', 'text'])
        else:
            raise TypeError('data should be list or str(file path)')

        X_test, Y_test = data['text'], data['label']
        logger.debug('load test data, X size: {}, Y_test size: {}'.format(len(X_test), len(Y_test)))
        assert len(X_test) == len(Y_test)

        X_test_token = [' '.join(self.tokenizer.tokenize(i)) for i in X_test]
        X_test_vec = self.vectorizer.transform(X_test_token)
        return self.model.score(X_test_vec, Y_test)

    def predict(self, X):
        pd.Series(X)
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def save(self, model_dir=''):
        if self.model is None:
            raise ValueError('model is None, run train first.')
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        save_pkl(self.vectorizer, vectorizer_path)
        model_path = os.path.join(model_dir, 'model.pkl')
        save_pkl(self.model, model_path)

    def load(self, model_dir=''):
        model_path = os.path.join(model_dir, 'model.pkl')
        if not os.path.exists(model_path):
            raise ValueError("model is not found. please train and save model first.")
        self.model = load_pkl(model_path)
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        self.vectorizer = load_pkl(vectorizer_path)
        logger.info('model loaded from {}'.format(model_dir))
