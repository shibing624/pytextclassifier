# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BERT Classifier, support 'bert', 'albert', 'roberta', 'xlnet' model
"""
import argparse
import json
import os

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from pytextclassifier.base_classifier import ClassifierABC, load_data
from pytextclassifier.data_helper import set_seed, load_vocab

pwd_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_dataset(data_df, label_vocab_path):
    y = data_df['labels']
    if os.path.exists(label_vocab_path):
        label_id_map = json.load(open(label_vocab_path, 'r', encoding='utf-8'))
    else:
        id_label_map = {id: v for id, v in enumerate(set(y.tolist()))}
        label_id_map = {v: k for k, v in id_label_map.items()}
        json.dump(label_id_map, open(label_vocab_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    logger.debug(f"label vocab size: {len(label_id_map)}, label_vocab_path: {label_vocab_path}")

    df = data_df.copy()
    df.loc[:, 'labels'] = df.loc[:, 'labels'].map(lambda x: label_id_map.get(x))
    data_df = df
    return data_df, label_id_map


class BertClassifier(ClassifierABC):
    def __init__(
            self,
            model_dir,
            model_type='bert',
            model_name='bert-base-chinese',
            num_classes=10,
            num_epochs=3,
            batch_size=64,
            max_seq_length=128,
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
        @param use_cuda:
        """
        train_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "output_dir": model_dir,
            "max_seq_length": max_seq_length,
            "num_train_epochs": num_epochs,
            "train_batch_size": batch_size,
        }
        use_cuda = torch.cuda.is_available()
        try:
            from simpletransformers.classification import ClassificationModel
        except ImportError:
            raise ImportError("Please install simpletransformers with `pip install simpletransformers`")
        self.model = ClassificationModel(
            model_type=model_type,
            model_name=model_name,
            num_labels=num_classes,
            args=train_args,
            use_cuda=use_cuda
        )
        self.model_dir = model_dir
        self.model_type = model_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.is_trained = False

    def __str__(self):
        return f'BertClassifier instance ({self.model})'

    def train(
            self,
            data_list_or_path,
            header=None, names=('labels', 'text'), delimiter='\t', test_size=0.1,
    ):
        """
        Train model with data_list_or_path and save model to model_dir
        @param data_list_or_path:
        @param header:
        @param names:
        @param delimiter:
        @param test_size:
        @return:
        """
        logger.debug('train model...')
        SEED = 1
        set_seed(SEED)
        # load data
        X, y, data_df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter)
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        label_vocab_path = os.path.join(self.model_dir, 'label_vocab.json')
        data_df, self.label_id_map = build_dataset(data_df, label_vocab_path)
        train_data, dev_data = train_test_split(data_df, test_size=test_size, random_state=SEED)
        logger.debug(f"train_data size: {len(train_data)}, dev_data size: {len(dev_data)}")
        logger.debug(f'train_data sample:\n{train_data[:3]}\ndev_data sample:\n{dev_data[:3]}')
        # train model
        self.model.train_model(train_data)
        # evaluate model
        result, model_outputs, wrong_predictions = self.model.eval_model(dev_data)
        logger.debug(f'evaluate, result:{result} model_outputs:{model_outputs} wrong_predictions:{wrong_predictions}')
        self.is_trained = True
        logger.debug('train model done')
        return result

    def predict(self, sentences: list):
        """
        Predict labels and label probability for sentences.
        @param sentences: list, input text list, eg: [text1, text2, ...]
        @return: predict_label, predict_prob
        """
        if not self.is_trained:
            raise ValueError('model not trained.')
        # predict
        predictions, raw_outputs = self.model.predict(sentences)
        # predict probability
        id_label_map = {v: k for k, v in self.label_id_map.items()}
        predict_labels = [id_label_map.get(i) for i in predictions]
        predict_probs = [1 - np.exp(-raw_output[prediction]) for raw_output, prediction in
                         zip(raw_outputs, predictions)]
        return predict_labels, predict_probs

    def evaluate_model(self, data_list_or_path, header=None,
                       names=('labels', 'text'), delimiter='\t'):
        X_test, y_test, df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter)
        self.load_model()
        result, model_outputs, wrong_predictions = self.model.eval_model(df)
        return result

    def load_model(self):
        """
        Load model from model_dir
        @param model_dir:
        @return:
        """
        model_path = os.path.join(self.model_dir, 'pytorch_model.pth')
        if os.path.exists(model_path):
            self.label_vocab_path = os.path.join(self.model_dir, 'label_vocab.json')
            self.label_id_map = load_vocab(self.label_vocab_path)
            num_classes = len(self.label_id_map)
            self.model = BertClassifier(
                model_dir=self.model_dir,
                model_type=self.model_type,
                model_name=self.model_dir,
                num_classes=num_classes,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
            )
            self.is_trained = True
        else:
            logger.error(f'{model_path} not exists.')
            self.is_trained = False
        return self.is_trained


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bert Text Classification')
    parser.add_argument('--pretrain_model_type', default='bert', type=str,
                        help='pretrained huggingface model type')
    parser.add_argument('--pretrain_model_name', default='bert-base-chinese', type=str,
                        help='pretrained huggingface model name')
    parser.add_argument('--model_dir', default='bert', type=str, help='save model dir')
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
        model_type=args.pretrain_model_type,
        model_name=args.pretrain_model_name,
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
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
