# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BERT Classifier, support 'bert', 'albert', 'roberta', 'xlnet' model
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.append('..')
from pytextclassifier.base_classifier import ClassifierABC, load_data
from pytextclassifier.data_helper import set_seed, load_vocab
from pytextclassifier.bert_classification_model import BertClassificationModel, BertClassificationArgs

pwd_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertClassifier(ClassifierABC):
    def __init__(
            self,
            model_dir,
            num_classes,
            model_type='bert',
            model_name='bert-base-chinese',
            num_epochs=3,
            batch_size=64,
            max_seq_length=128,
            multi_label=False,
            labels_sep=',',
            args=None,
    ):
        """
        Init classification model
        @param model_dir: str, model dir
        @param model_type: support 'bert', 'albert', 'roberta', 'xlnet'
        @param model_name:
        @param num_classes:
        @param num_epochs:
        @param batch_size:
        @param max_seq_length:
        @param multi_label:
        @param labels_sep:
        @param args:
        """
        default_args = {
            "output_dir": model_dir,
            "max_seq_length": max_seq_length,
            "num_train_epochs": num_epochs,
            "train_batch_size": batch_size,
            "best_model_dir": os.path.join(model_dir, 'best_model'),
            "labels_sep": labels_sep,
        }
        train_args = BertClassificationArgs()
        if args and isinstance(args, dict):
            train_args.update_from_dict(args)
        train_args.update_from_dict(default_args)

        self.model = BertClassificationModel(
            model_type=model_type,
            model_name=model_name,
            num_labels=num_classes,
            multi_label=multi_label,
            args=train_args,
            use_cuda=use_cuda,
        )
        self.model_dir = model_dir
        self.model_type = model_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.train_args = train_args
        self.use_cuda = use_cuda
        self.multi_label = multi_label
        self.labels_sep = labels_sep
        self.label_vocab_path = os.path.join(self.model_dir, 'label_vocab.json')
        self.is_trained = False

    def __str__(self):
        return f'BertClassifier instance ({self.model})'

    def train(
            self,
            data_list_or_path,
            dev_data_list_or_path=None,
            header=None,
            names=('labels', 'text'),
            delimiter='\t',
            test_size=0.1,
    ):
        """
        Train model with data_list_or_path and save model to model_dir
        @param data_list_or_path:
        @param dev_data_list_or_path:
        @param header:
        @param names:
        @param delimiter:
        @param test_size:
        @return:
        """
        logger.debug('train model ...')
        SEED = 1
        set_seed(SEED)
        # load data
        X, y, data_df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter,
                                  labels_sep=self.labels_sep, is_train=True)
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        labels_map = self.build_labels_map(y, self.label_vocab_path, self.multi_label, self.labels_sep)
        labels_list = sorted(list(labels_map.keys()))
        if dev_data_list_or_path is not None:
            dev_X, dev_y, dev_df = load_data(dev_data_list_or_path, header=header, names=names,
                                             delimiter=delimiter, labels_sep=self.labels_sep,
                                             is_train=False)
            train_data = data_df
            dev_data = dev_df
        else:
            if test_size > 0:
                train_data, dev_data = train_test_split(data_df, test_size=test_size, random_state=SEED)
            else:
                train_data = data_df
                dev_data = None
        logger.debug(f"train_data size: {len(train_data)}")
        logger.debug(f'train_data sample:\n{train_data[:3]}')
        if dev_data is not None and dev_data.size:
            logger.debug(f"dev_data size: {len(dev_data)}")
            logger.debug(f'dev_data sample:\n{dev_data[:3]}')
        # train model
        if self.train_args.lazy_loading:
            train_data = data_list_or_path
            dev_data = dev_data_list_or_path
        if dev_data is not None:
            self.model.train_model(
                train_data, eval_df=dev_data,
                args={'labels_map': labels_map, 'labels_list': labels_list}
            )
        else:
            self.model.train_model(
                train_data,
                args={'labels_map': labels_map, 'labels_list': labels_list}
            )
        self.is_trained = True
        logger.debug('train model done')

    def predict(self, sentences: list):
        """
        Predict labels and label probability for sentences.
        @param sentences: list, input text list, eg: [text1, text2, ...]
        @return: predict_labels, predict_probs
        """
        if not self.is_trained:
            raise ValueError('model not trained.')
        # predict
        predictions, raw_outputs = self.model.predict(sentences)
        if self.multi_label:
            return predictions, raw_outputs
        else:
            # predict probability
            predict_probs = [1 - np.exp(-np.max(raw_output)) for raw_output, prediction in
                             zip(raw_outputs, predictions)]
            return predictions, predict_probs

    def evaluate_model(self, data_list_or_path, header=None,
                       names=('labels', 'text'), delimiter='\t', **kwargs):
        if self.train_args.lazy_loading:
            eval_df = data_list_or_path
        else:
            X_test, y_test, eval_df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter,
                                            labels_sep=self.labels_sep)
        if not self.is_trained:
            self.load_model()
        result, model_outputs, wrong_predictions = self.model.eval_model(
            eval_df,
            output_dir=self.model_dir,
            **kwargs,
        )
        return result

    def load_model(self):
        """
        Load model from model_dir
        @return:
        """
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        if os.path.exists(model_path):
            labels_map = json.load(open(self.label_vocab_path, 'r', encoding='utf-8'))
            labels_list = sorted(list(labels_map.keys()))
            num_classes = len(labels_map)
            assert num_classes == self.num_classes, f'num_classes not match, {num_classes} != {self.num_classes}'
            self.train_args.update_from_dict({'labels_map': labels_map, 'labels_list': labels_list})
            self.model = BertClassificationModel(
                model_type=self.model_type,
                model_name=self.model_dir,
                num_labels=self.num_classes,
                multi_label=self.multi_label,
                args=self.train_args,
                use_cuda=self.use_cuda,
            )
            self.is_trained = True
        else:
            logger.error(f'{model_path} not exists.')
            self.is_trained = False
        return self.is_trained

    @staticmethod
    def build_labels_map(y, label_vocab_path, multi_label=False, labels_sep=','):
        """
        Build labels map
        @param y:
        @param label_vocab_path:
        @param multi_label:
        @param labels_sep:
        @return:
        """
        if multi_label:
            labels = set()
            for label in y.tolist():
                if isinstance(label, str):
                    labels.update(label.split(labels_sep))
                elif isinstance(label, list):
                    labels.update(range(len(label)))
                else:
                    labels.add(label)
        else:
            labels = set(y.tolist())
        labels = sorted(list(labels))
        id_label_map = {id: v for id, v in enumerate(labels)}
        label_id_map = {v: k for k, v in id_label_map.items()}
        json.dump(label_id_map, open(label_vocab_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        logger.debug(f"label vocab size: {len(label_id_map)}, label_vocab_path: {label_vocab_path}")
        return label_id_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bert Text Classification')
    parser.add_argument('--pretrain_model_type', default='bert', type=str,
                        help='pretrained huggingface model type')
    parser.add_argument('--pretrain_model_name', default='bert-base-chinese', type=str,
                        help='pretrained huggingface model name')
    parser.add_argument('--model_dir', default='models/bert', type=str, help='save model dir')
    parser.add_argument('--data_path', default=os.path.join(pwd_path, '../examples/thucnews_train_1w.txt'),
                        type=str, help='sample data file path')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--num_epochs', default=3, type=int, help='train epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--max_seq_length', default=128, type=int, help='max seq length, trim longer sentence.')
    args = parser.parse_args()
    print(args)
    # create model
    m = BertClassifier(
        model_dir=args.model_dir,
        num_classes=args.num_classes,
        model_type=args.pretrain_model_type,
        model_name=args.pretrain_model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        multi_label=False,
    )
    # train model
    m.train(data_list_or_path=args.data_path)
    # load trained best model and predict
    m.load_model()
    print('best model loaded from file, and predict')
    X, y, _ = load_data(args.data_path)
    X = X[:5]
    y = y[:5]
    predict_labels, predict_probs = m.predict(X)
    for text, pred_label, pred_prob, y_truth in zip(X, predict_labels, predict_probs, y):
        print(text, 'pred:', pred_label, pred_prob, ' truth:', y_truth)
