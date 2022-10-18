# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pandas as pd
from loguru import logger


def load_data(data_list_or_path, header=None, names=('labels', 'text'), delimiter='\t'):
    """
    Encoding data_list text
    @param data_list_or_path: list of (label, text), eg: [(label, text), (label, text) ...]
    @param header: read_csv header
    @param names: read_csv names
    @param delimiter: read_csv sep
    @return: X, y, data_df
    """
    if isinstance(data_list_or_path, list):
        data_df = pd.DataFrame(data_list_or_path, columns=names)
    elif isinstance(data_list_or_path, str) and os.path.exists(data_list_or_path):
        data_df = pd.read_csv(data_list_or_path, header=header, delimiter=delimiter, names=names)
    elif isinstance(data_list_or_path, pd.DataFrame):
        data_df = data_list_or_path
    else:
        raise TypeError('should be list or file path, eg: [(label, text), ... ]')
    X, y = data_df['text'], data_df['labels']
    if y.size:
        num_classes = len(y[0]) if isinstance(y[0], list) else len(set(y))
        logger.debug(f'loaded data list, X size: {len(X)}, y size: {len(y)}, num_classes: {num_classes}')
    assert len(X) == len(y)
    return X, y, data_df


class ClassifierABC:
    """
    Abstract class for classifier
    """
    def train(self, data_list_or_path, model_dir: str, **kwargs):
        raise NotImplementedError('train method not implemented.')

    def predict(self, sentences: list):
        raise NotImplementedError('predict method not implemented.')

    def evaluate_model(self, **kwargs):
        raise NotImplementedError('evaluate_model method not implemented.')

    def evaluate(self, **kwargs):
        raise NotImplementedError('evaluate method not implemented.')

    def load_model(self):
        raise NotImplementedError('load method not implemented.')

    def save_model(self):
        raise NotImplementedError('save method not implemented.')
