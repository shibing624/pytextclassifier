# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import torch
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.append('..')
from pytextclassifier.log import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_list_or_filepath, header=None, names=None, delimiter='\t', **kwargs):
    """
    Encoding data_list text
    data_list_or_filepath: list of (label, text), eg: [(label, text), (label, text) ...]
    return: X, X_tokens, Y
    """
    if names is None:
        names = ['labels', 'text']
    if isinstance(data_list_or_filepath, list):
        data_df = pd.DataFrame(data_list_or_filepath, columns=names)
    elif isinstance(data_list_or_filepath, str) and os.path.exists(data_list_or_filepath):
        data_df = pd.read_csv(data_list_or_filepath, header=header, delimiter=delimiter, names=names, **kwargs)
    elif isinstance(data_list_or_filepath, pd.DataFrame):
        data_df = data_list_or_filepath
    else:
        raise TypeError('should be list or file path, eg: [(label, text), ... ]')
    X, y = data_df['text'], data_df['label']
    logger.debug('loaded data list, X size: {}, y size: {}'.format(len(X), len(y)))
    assert len(X) == len(y)
    logger.debug('num_classes:%d' % len(set(y)))
    return X, y, data_df


class TextClassifier:
    def __init__(self, model_name='lr', model_dir=''):
        """
        Init instance
        :param model: sklearn model
        :param tokenizer: word segmentation
        :param vectorizer: sklearn vectorizer
        """
        self.model_name = model_name
        self.is_trained = False
        self.model_dir = model_dir
        self.model = None
        self.word_vocab_path = os.path.join(self.model_dir, 'word_vocab.pkl')
        self.label_vocab_path = os.path.join(self.model_dir, 'label_vocab.pkl')
        self.save_model_path = os.path.join(self.model_dir, 'model.pth')

    def __repr__(self):
        return 'TextClassifier instance ({})'.format(self.model_name)

    def train(self, data_list, **kwargs):
        """
        Train model
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: model
        """
        logger.debug('train model...')
        logger.debug(f'device: {device}')
        os.makedirs(self.model_dir, exist_ok=True)
        print(f'device: {device}')
        # load data
        X, y, data_df = load_data(data_list)

        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import train, evaluate, get_model
            model = get_model(self.model_name)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
            # train model
            model, vectorizer = train(X_train, y_train, self.model_dir, model=model, **kwargs)
            # evaluate the model
            evaluate(X_test, y_test, model=model, vectorizer=vectorizer)
            self.model = model
            self.vectorizer = vectorizer
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import (
                build_dataset, build_iterator, batch_size, FastTextModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                            **kwargs)
            train_data, dev_data = train_test_split(data, test_size=0.1, random_state=1)
            train_iter = build_iterator(train_data, batch_size, device)
            dev_iter = build_iterator(dev_data, batch_size, device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = FastTextModel(vocab_size, num_classes).to(device)
            init_network(model)
            logger.debug(model.parameters)
            # train model
            train(model, train_iter, dev_iter, save_path=self.save_model_path, **kwargs)
            self.model = model
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import (
                build_dataset, build_iterator, batch_size, TextCNNModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                            **kwargs)
            train_data, dev_data = train_test_split(data, test_size=0.1, random_state=1)
            train_iter = build_iterator(train_data, batch_size, device)
            dev_iter = build_iterator(dev_data, batch_size, device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = TextCNNModel(vocab_size, num_classes).to(device)
            init_network(model)
            print(model.parameters)
            # train model
            train(model, train_iter, dev_iter, save_path=self.save_model_path, **kwargs)
            self.model = model
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import (
                build_dataset, build_iterator, batch_size, TextRNNAttModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                            **kwargs)
            train_data, dev_data = train_test_split(data, test_size=0.1, random_state=1)
            train_iter = build_iterator(train_data, batch_size, device)
            dev_iter = build_iterator(dev_data, batch_size, device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = TextRNNAttModel(vocab_size, num_classes).to(device)
            init_network(model)
            print(model.parameters)
            # train model
            train(model, train_iter, dev_iter, save_path=self.save_model_path, **kwargs)
            self.model = model
        elif self.model_name == 'bert':
            from pytextclassifier.tools.bert_classification import (
                build_dataset, predict, BertClassificationModel)
            data_df, label_id_map = build_dataset(data_df, self.label_vocab_path)
            print(data_df.head())
            train_df, dev_df = train_test_split(data_df, test_size=0.1, random_state=1)
            # create model
            use_cuda = False if device == torch.device('cpu') else True
            num_classes = len(set(y.tolist()))
            model = BertClassificationModel(use_cuda=use_cuda, model_dir=self.model_dir, num_classes=num_classes,
                                            **kwargs)
            # train model
            # Train and Evaluation data needs to be in a Pandas Dataframe,
            # it should contain a 'text' and a 'labels' column. text with type str, the label with type int.
            model.train_model(train_df)
            # evaluate the model
            result, model_outputs, wrong_predictions = model.eval_model(dev_df)
            print('evaluate: ', result, model_outputs, wrong_predictions)
            self.model = model
        else:
            raise ValueError('model_name not found.')
        self.is_trained = True
        logger.debug('train model done')

    def evaluate(self, data_list, **kwargs):
        """
        Evaluate model with data
        :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
        :return: acc score
        """
        logger.debug('evaluate model...')
        if not self.is_trained:
            raise ValueError('please train model first.')
        # load data
        X, y, data_df = load_data(data_list)
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import train, evaluate, get_model
            # evaluate the model
            acc = evaluate(X, y, model=self.model, vectorizer=self.vectorizer)
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                                **kwargs)
            dev_iter = build_iterator(dev_data, device, **kwargs)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                                **kwargs)
            dev_iter = build_iterator(dev_data, device, **kwargs)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path, self.label_vocab_path,
                                                                **kwargs)
            dev_iter = build_iterator(dev_data, device, **kwargs)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name == 'bert':
            from pytextclassifier.tools.bert_classification import (
                build_dataset, predict, BertClassificationModel)
            dev_df, label_id_map = build_dataset(data_df, self.label_vocab_path)
            # evaluate the model
            result, model_outputs, wrong_predictions = self.model.eval_model(dev_df)
            print('evaluate: ', result, model_outputs, wrong_predictions)
            acc = result['mcc']
        else:
            raise ValueError('model_name not found.')
        logger.debug('evaluate model done, accuracy_score: {}'.format(acc))
        return acc

    def predict(self, input_text_list):
        """
        Predict label
        :param input_text_list: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if isinstance(input_text_list, str) or not hasattr(input_text_list, '__len__'):
            raise ValueError('input X should be list, eg: [text1, text2, ...]')
        if not self.is_trained:
            raise ValueError('please train model first.')
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import predict, get_model
            predict_label, predict_proba = predict(input_text_list, model=self.model, vectorizer=self.vectorizer)
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import predict
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, label_id_map)
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import predict
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, label_id_map)
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import predict
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, label_id_map)
        elif self.model_name == 'bert':
            from pytextclassifier.tools.bert_classification import predict
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, label_id_map)
        else:
            raise ValueError('model_name not found.')
        return predict_label, predict_proba

    def load_model(self, **kwargs):
        """
        Load model from self.model_dir
        :return: None
        """
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import load_model
            self.model, self.vectorizer = load_model(model_dir=self.model_dir)
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import load_model, FastTextModel
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            model = FastTextModel(len(word_id_map), len(label_id_map)).to(device)
            self.model = load_model(model, self.save_model_path)
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import load_model, TextCNNModel
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            model = TextCNNModel(len(word_id_map), len(label_id_map)).to(device)
            self.model = load_model(model, self.save_model_path)
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import load_model, TextRNNAttModel
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            model = TextRNNAttModel(len(word_id_map), len(label_id_map)).to(device)
            self.model = load_model(model, self.save_model_path)
        elif self.model_name == 'bert':
            from pytextclassifier.tools.bert_classification import BertClassificationModel
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            use_cuda = False if device == torch.device('cpu') else True
            self.model = BertClassificationModel(model_name=self.model_dir, num_classes=len(label_id_map),
                                                 use_cuda=use_cuda, **kwargs)
        else:
            raise ValueError('model_name not found.')
        self.is_trained = True
        logger.info('model loaded {}'.format(self.model_dir))
