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


def load_data(data_list_or_filepath, header=None, names=None, delimiter='\t'):
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
        data_df = pd.read_csv(data_list_or_filepath, header=header, delimiter=delimiter, names=names)
    elif isinstance(data_list_or_filepath, pd.DataFrame):
        data_df = data_list_or_filepath
    else:
        raise TypeError('should be list or file path, eg: [(label, text), ... ]')
    X, y = data_df['text'], data_df['labels']
    logger.debug(f'loaded data list, X size: {len(X)}, y size: {len(y)}, num_classes: {len(set(y))}')
    assert len(X) == len(y)
    return X, y, data_df


class TextClassifier:
    def __init__(self, model_name='lr', model_dir=None):
        """
        Init classification instance
        @param model_name: 模型名称，可以是 lr, random_forest, textcnn, fasttext, textrnn_att, bert
        @param model_dir: 模型保存路径，默认跟model_name同名
        """
        self.model_name = model_name
        self.is_trained = False
        self.model_dir = model_dir if model_dir else model_name
        self.model = None
        self.word_vocab_path = os.path.join(self.model_dir, 'word_vocab.pkl')
        self.label_vocab_path = os.path.join(self.model_dir, 'label_vocab.pkl')
        self.save_model_path = os.path.join(self.model_dir, f'{model_name}_model.pth')

    def __repr__(self):
        return 'TextClassifier instance ({})'.format(self.model_name)

    def train(self, data_list_or_filepath, header=None, names=None, delimiter='\t', vectorizer=None,
              pad_size=128, test_size=0.1, batch_size=64, num_epochs=10, learning_rate=1e-3,
              require_improvement=1000, hf_model_type='bert', hf_model_name='bert-base-chinese'):
        """
        Train model
        @param data_list_or_filepath: list of (label, text) or filepath, eg: [(label, text), (label, text) ...]
        @param header: 读文件的header
        @param names: 读文件的names，默认为[labels, text]
        @param delimiter: 读文件的字段分隔
        @param vectorizer: 自定义sklearn vectorizer，如TfidfVectorizer
        @param pad_size: max_seq_length
        @param test_size: 训练模型的dev data占比
        @param batch_size: mini-batch大小
        @param num_epochs: 训练多少轮
        @param learning_rate: 学习率
        @param require_improvement: 默认1000，若超过1000batch效果还没提升，则提前结束训练
        @param hf_model_type: 默认bert，simpletransformers的model_type
        @param hf_model_name: 默认bert-base-chinese，simpletransformers的model_name
        @return: None, and set self.is_trained = True
        """
        logger.info(f'model_name: {self.model_name}')
        logger.debug(f'train model...')
        logger.debug(f'device: {device}')
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        # load data
        X, y, data_df = load_data(data_list_or_filepath, header=header, names=names, delimiter=delimiter)
        logger.debug(data_df.head())
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import train, evaluate, get_model
            model = get_model(self.model_name)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
            # train model
            model, vectorizer = train(X_train, y_train, self.model_dir, model=model, vectorizer=vectorizer)
            # evaluate the model
            test_acc = evaluate(X_test, y_test, model, vectorizer)
            logger.debug(
                f'[train] evaluate, X_test size: {len(X_test)}, y_test size: {len(y_test)}, test acc: {test_acc} ')
            self.model = model
            self.vectorizer = vectorizer
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import (
                build_dataset, build_iterator, FastTextModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                            self.label_vocab_path, pad_size=pad_size)
            train_data, dev_data = train_test_split(data, test_size=test_size, random_state=1)
            train_iter = build_iterator(train_data, batch_size=batch_size, device=device)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = FastTextModel(vocab_size, num_classes).to(device)
            init_network(model)
            logger.debug(model.parameters)
            # train model
            train(model, train_iter, dev_iter, num_epochs=num_epochs, learning_rate=learning_rate,
                  require_improvement=1000, save_path=self.save_model_path)
            self.model = model
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import (
                build_dataset, build_iterator, TextCNNModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                            self.label_vocab_path, pad_size=pad_size)
            train_data, dev_data = train_test_split(data, test_size=test_size, random_state=1)
            train_iter = build_iterator(train_data, batch_size=batch_size, device=device)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = TextCNNModel(vocab_size, num_classes).to(device)
            init_network(model)
            logger.debug(model.parameters)
            # train model
            train(model, train_iter, dev_iter, num_epochs=num_epochs, learning_rate=learning_rate,
                  require_improvement=require_improvement, save_path=self.save_model_path)
            self.model = model
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import (
                build_dataset, build_iterator, TextRNNAttModel, init_network, train)
            data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                            self.label_vocab_path, pad_size=pad_size)
            train_data, dev_data = train_test_split(data, test_size=test_size, random_state=1)
            train_iter = build_iterator(train_data, batch_size=batch_size, device=device)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            # create model
            vocab_size = len(word_id_map)
            num_classes = len(set(y.tolist()))
            model = TextRNNAttModel(vocab_size, num_classes).to(device)
            init_network(model)
            logger.debug(model.parameters)
            # train model
            train(model, train_iter, dev_iter, num_epochs=num_epochs, learning_rate=learning_rate,
                  require_improvement=require_improvement, save_path=self.save_model_path)
            self.model = model
        elif self.model_name in ['bert', 'albert', 'roberta', 'xlnet']:
            from pytextclassifier.tools.bert_classification import (
                build_dataset, predict, BertClassificationModel)
            data_df, label_id_map = build_dataset(data_df, self.label_vocab_path)
            train_df, dev_df = train_test_split(data_df, test_size=test_size, random_state=1)
            # create model
            use_cuda = False if device == torch.device('cpu') else True
            model = BertClassificationModel(model_type=hf_model_type,
                                            model_name=hf_model_name,
                                            num_classes=len(label_id_map),
                                            num_epochs=num_epochs,
                                            batch_size=batch_size,
                                            max_seq_length=pad_size,
                                            model_dir=self.model_dir,
                                            use_cuda=use_cuda)
            # train model
            # Train and Evaluation data needs to be in a Pandas Dataframe,
            # it should contain a 'text' and a 'labels' column. text with type str, the label with type int.
            model.train_model(train_df)
            # evaluate the model
            result, model_outputs, wrong_predictions = model.eval_model(dev_df)
            logger.debug(f'evaluate, {result}, wrong_predictions: {wrong_predictions}')
            if wrong_predictions and dev_df:
                acc = (len(dev_df) - len(wrong_predictions[0])) / len(dev_df)
                logger.debug(f'evaluate, dev data size: {len(dev_df)}, wrong_predictions size: '
                             f'{len(wrong_predictions[0])}, acc :{acc}')
            self.model = model
        else:
            raise ValueError('model_name not found.')
        self.is_trained = True
        logger.debug('train model done')

    def evaluate(self, data_list_or_filepath, header=None, names=None, delimiter='\t',
                 pad_size=128, batch_size=64):
        """
        Evaluate model
        @param data_list_or_filepath:
        @param header:
        @param names:
        @param delimiter:
        @param pad_size:
        @param batch_size:
        @return: accuracy_score
        """
        logger.debug('evaluate model...')
        if not self.is_trained:
            raise ValueError('please train model first.')
        # load data
        X, y, data_df = load_data(data_list_or_filepath, header=header, names=names, delimiter=delimiter)
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import train, evaluate, get_model
            # evaluate the model
            acc = evaluate(X, y, self.model, self.vectorizer)
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                                self.label_vocab_path, pad_size=pad_size)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                                self.label_vocab_path, pad_size=pad_size)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import (
                build_dataset, build_iterator, evaluate)
            dev_data, word_id_map, label_id_map = build_dataset(X, y, self.word_vocab_path,
                                                                self.label_vocab_path, pad_size=pad_size)
            dev_iter = build_iterator(dev_data, batch_size=batch_size, device=device)
            acc, dev_loss = evaluate(self.model, dev_iter)
        elif self.model_name in ['bert', 'albert', 'roberta', 'xlnet']:
            from pytextclassifier.tools.bert_classification import (
                build_dataset, predict, BertClassificationModel)
            dev_df, label_id_map = build_dataset(data_df, self.label_vocab_path)
            result, model_outputs, wrong_predictions = self.model.eval_model(dev_df)
            if wrong_predictions:
                acc = (len(dev_df) - len(wrong_predictions[0])) / len(dev_df)
            else:
                acc = 1.0
        else:
            raise ValueError('model_name not found.')
        logger.debug(f'evaluate model done, accuracy_score: {acc}')
        return acc

    def predict(self, input_text_list):
        """
        Predict label and prob
        @param input_text_list: list, eg: [text1, text2, ...]
        @return: (predict_label, predict_proba)
        """
        if isinstance(input_text_list, str) or not hasattr(input_text_list, '__len__'):
            raise ValueError('input X should be list, eg: [text1, text2, ...]')
        if not self.is_trained:
            raise ValueError('please train model first.')
        if self.model_name in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
            from pytextclassifier.tools.lr_classification import predict, get_model
            predict_label, predict_proba = predict(input_text_list, self.model, self.vectorizer)
        elif self.model_name == 'fasttext':
            from pytextclassifier.tools.fasttext_classification import predict
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, word_id_map, label_id_map)
        elif self.model_name == 'textcnn':
            from pytextclassifier.tools.textcnn_classification import predict
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, word_id_map, label_id_map)
        elif self.model_name == 'textrnn_att':
            from pytextclassifier.tools.textrnn_att_classification import predict
            word_id_map = pickle.load(open(self.word_vocab_path, 'rb'))
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, word_id_map, label_id_map)
        elif self.model_name in ['bert', 'albert', 'roberta', 'xlnet']:
            from pytextclassifier.tools.bert_classification import predict
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            predict_label, predict_proba = predict(self.model, input_text_list, label_id_map)
        else:
            raise ValueError('model_name not found.')
        return predict_label, predict_proba

    def load_model(self):
        """
        Load model from self.model_dir
        @return: None, set self.is_trained = True
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
        elif self.model_name in ['bert', 'albert', 'roberta', 'xlnet']:
            from pytextclassifier.tools.bert_classification import BertClassificationModel
            label_id_map = pickle.load(open(self.label_vocab_path, 'rb'))
            use_cuda = False if device == torch.device('cpu') else True
            self.model = BertClassificationModel(model_type=self.model_name,
                                                 model_name=self.model_dir,
                                                 num_classes=len(label_id_map),
                                                 use_cuda=use_cuda)
        else:
            raise ValueError('model_name not found.')
        self.is_trained = True
        logger.info(f'model loaded done. {self.model_dir}')
