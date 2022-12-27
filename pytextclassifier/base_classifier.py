# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pandas as pd
from loguru import logger


def load_data(data_list_or_path, header=None, names=('labels', 'text'), delimiter='\t',
              labels_sep=',', is_train=False):
    """
    Encoding data_list text
    @param data_list_or_path: list of (label, text), eg: [(label, text), (label, text) ...]
    @param header: read_csv header
    @param names: read_csv names
    @param delimiter: read_csv sep
    @param labels_sep: multi label split
    @param is_train: is train data
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
    labels = set()
    if y.size:
        for label in y.tolist():
            if isinstance(label, str):
                labels.update(label.split(labels_sep))
            elif isinstance(label, list):
                labels.update(range(len(label)))
            else:
                labels.add(label)
        num_classes = len(labels)
        labels = sorted(list(labels))
        logger.debug(f'loaded data list, X size: {len(X)}, y size: {len(y)}')
        if is_train:
            logger.debug('num_classes: %d, labels: %s' % (num_classes, labels))
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
