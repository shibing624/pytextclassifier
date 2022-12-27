# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Fasttext Classifier
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn import metrics
from sklearn.model_selection import train_test_split

sys.path.append('..')
from pytextclassifier.base_classifier import ClassifierABC, load_data
from pytextclassifier.data_helper import set_seed, build_vocab, load_vocab
from pytextclassifier.time_util import get_time_spend

pwd_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_dataset(
        tokenizer, X, y, word_vocab_path, label_vocab_path, max_vocab_size=10000,
        max_seq_length=128, unk_token='[UNK]', pad_token='[PAD]',
        n_gram_vocab=250499, enable_ngram=True
):
    if os.path.exists(word_vocab_path):
        word_id_map = load_vocab(word_vocab_path)
    else:
        word_id_map = build_vocab(X, tokenizer=tokenizer, max_size=max_vocab_size, min_freq=1,
                                  unk_token=unk_token, pad_token=pad_token)
        json.dump(word_id_map, open(word_vocab_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        logger.debug('save vocab_path: {}'.format(word_vocab_path))
    logger.debug(f"word vocab size: {len(word_id_map)}, word_vocab_path: {word_vocab_path}")

    if os.path.exists(label_vocab_path):
        label_id_map = load_vocab(label_vocab_path)
    else:
        id_label_map = {id: v for id, v in enumerate(set(y.tolist()))}
        label_id_map = {v: k for k, v in id_label_map.items()}
        json.dump(label_id_map, open(label_vocab_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        logger.debug('save label_vocab_path: {}'.format(label_vocab_path))
    logger.debug(f"label vocab size: {len(label_id_map)}, label_vocab_path: {label_vocab_path}")

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(X, y, max_seq_length=128):
        contents = []
        for content, label in zip(X, y):
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if max_seq_length:
                if len(token) < max_seq_length:
                    token.extend([pad_token] * (max_seq_length - len(token)))
                else:
                    token = token[:max_seq_length]
                    seq_len = max_seq_length
            # word to id
            for word in token:
                words_line.append(word_id_map.get(word, word_id_map.get(unk_token)))
            label_id = label_id_map.get(label)
            # fasttext ngram
            bigram = []
            trigram = []
            if enable_ngram:
                buckets = n_gram_vocab
                # ------ngram------
                for i in range(max_seq_length):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
            else:
                bigram = [0] * max_seq_length
                trigram = [0] * max_seq_length
            contents.append((words_line, label_id, seq_len, bigram, trigram))
        return contents

    dataset = load_dataset(X, y, max_seq_length)
    return dataset, word_id_map, label_id_map


class DatasetIterater:
    def __init__(self, dataset, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size if len(dataset) > batch_size else 1
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad_token前的长度(超过max_seq_length的设为max_seq_length)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, device, batch_size=32):
    return DatasetIterater(dataset, device, batch_size)


class FastTextModel(nn.Module):
    """Bag of Tricks for Efficient Text Classification"""

    def __init__(
            self, vocab_size, num_classes, embed_size=200, n_gram_vocab=250499, hidden_size=256, dropout_rate=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size - 1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed_size)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class FastTextClassifier(ClassifierABC):
    def __init__(
            self,
            model_dir,
            dropout_rate=0.5, batch_size=64, max_seq_length=128,
            embed_size=200, hidden_size=256, n_gram_vocab=250499,
            max_vocab_size=10000, unk_token='[UNK]', pad_token='[PAD]',
            tokenizer=None,
            enable_ngram=True,
    ):
        """
        初始化
        @param model_dir: 模型的保存路径
        @param dropout_rate: 随机失活
        @param batch_size: mini-batch大小
        @param max_seq_length: 每句话处理成的长度(短填长切)
        @param embed_size: 字向量维度
        @param hidden_size: 隐藏层大小
        @param n_gram_vocab: ngram 词表大小
        @param max_vocab_size: 词表长度限制
        @param unk_token: 未知字
        @param pad_token: padding符号
        @param tokenizer: 切词器
        @param enable_ngram: 是否使用ngram
        """
        self.model_dir = model_dir
        self.is_trained = False
        self.model = None
        logger.debug(f'device: {device}')
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_gram_vocab = n_gram_vocab
        self.max_vocab_size = max_vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.tokenizer = tokenizer if tokenizer else lambda x: [y for y in x]  # char-level
        self.enable_ngram = enable_ngram

    def __str__(self):
        return f'FasttextClassifier instance ({self.model})'

    def train(
            self,
            data_list_or_path,
            header=None, names=('labels', 'text'), delimiter='\t', test_size=0.1,
            num_epochs=20, learning_rate=1e-3,
            require_improvement=1000, evaluate_during_training_steps=100
    ):
        """
        Train model with data_list_or_path and save model to model_dir
        @param data_list_or_path:
        @param header:
        @param names:
        @param delimiter:
        @param test_size:
        @param num_epochs: epoch数
        @param learning_rate: 学习率
        @param require_improvement: 若超过1000batch效果还没提升，则提前结束训练
        @param evaluate_during_training_steps: 每隔多少step评估一次模型
        @return:
        """
        logger.debug('train model...')
        SEED = 1
        set_seed(SEED)
        # load data
        X, y, data_df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter, is_train=True)
        model_dir = self.model_dir
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        word_vocab_path = os.path.join(model_dir, 'word_vocab.json')
        label_vocab_path = os.path.join(model_dir, 'label_vocab.json')
        save_model_path = os.path.join(model_dir, 'model.pth')

        dataset, self.word_id_map, self.label_id_map = build_dataset(
            self.tokenizer, X, y, word_vocab_path,
            label_vocab_path,
            max_vocab_size=self.max_vocab_size,
            max_seq_length=self.max_seq_length,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            n_gram_vocab=self.n_gram_vocab,
            enable_ngram=self.enable_ngram
        )
        train_data, dev_data = train_test_split(dataset, test_size=test_size, random_state=SEED)
        logger.debug(f"train_data size: {len(train_data)}, dev_data size: {len(dev_data)}")
        logger.debug(f'train_data sample:\n{train_data[:3]}\ndev_data sample:\n{dev_data[:3]}')
        train_iter = build_iterator(train_data, device, self.batch_size)
        dev_iter = build_iterator(dev_data, device, self.batch_size)
        # create model
        vocab_size = len(self.word_id_map)
        num_classes = len(self.label_id_map)
        logger.debug(f'vocab_size:{vocab_size}', 'num_classes:', num_classes)
        self.model = FastTextModel(
            vocab_size, num_classes, self.embed_size, self.n_gram_vocab, self.hidden_size,
            self.dropout_rate
        )
        self.model.to(device)
        # init_network(self.model)
        logger.info(self.model.parameters)
        # train model
        history = self.train_model_from_data_iterator(
            save_model_path, train_iter, dev_iter, num_epochs, learning_rate,
            require_improvement, evaluate_during_training_steps
        )
        self.is_trained = True
        logger.debug('train model done')
        return history

    def train_model_from_data_iterator(
            self, save_model_path, train_iter, dev_iter,
            num_epochs=10, learning_rate=1e-3,
            require_improvement=1000, evaluate_during_training_steps=100
    ):
        history = []
        # train
        start_time = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = 1e10
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        for epoch in range(num_epochs):
            logger.debug('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                self.model.train()
                outputs = self.model(trains)
                loss = F.cross_entropy(outputs, labels)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if total_batch % evaluate_during_training_steps == 0:
                    # 输出在训练集和验证集上的效果
                    y_true = labels.cpu()
                    y_pred = torch.max(outputs, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(y_true, y_pred)
                    if dev_iter is not None:
                        dev_acc, dev_loss = self.evaluate(dev_iter)
                        if dev_loss < dev_best_loss:
                            dev_best_loss = dev_loss
                            torch.save(self.model.state_dict(), save_model_path)
                            logger.debug(f'Saved model: {save_model_path}')
                            improve = '*'
                            last_improve = total_batch
                        else:
                            improve = ''
                        time_dif = get_time_spend(start_time)
                        msg = 'Iter:{0:>6},Train Loss:{1:>5.2},Train Acc:{2:>6.2%},' \
                              'Val Loss:{3:>5.2},Val Acc:{4:>6.2%},Time:{5} {6}'.format(
                            total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)
                    else:
                        time_dif = get_time_spend(start_time)
                        msg = 'Iter:{0:>6},Train Loss:{1:>5.2},Train Acc:{2:>6.2%},Time:{3}'.format(
                            total_batch, loss.item(), train_acc, time_dif)
                    logger.debug(msg)
                    history.append(msg)
                    self.model.train()
                total_batch += 1
                if total_batch - last_improve > require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    logger.debug("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        return history

    def predict(self, sentences: list):
        """
        Predict labels and label probability for sentences.
        @param sentences: list, input text list, eg: [text1, text2, ...]
        @return: predict_label, predict_prob
        """
        if not self.is_trained:
            raise ValueError('model not trained.')
        self.model.eval()

        def biGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            return (t1 * 14918087) % buckets

        def triGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            t2 = sequence[t - 2] if t - 2 >= 0 else 0
            return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

        def load_dataset(X, max_seq_length=128):
            contents = []
            for content in X:
                words_line = []
                token = self.tokenizer(content)
                seq_len = len(token)
                if max_seq_length:
                    if len(token) < max_seq_length:
                        token.extend([self.pad_token] * (max_seq_length - len(token)))
                    else:
                        token = token[:max_seq_length]
                        seq_len = max_seq_length
                # word to id
                for word in token:
                    words_line.append(self.word_id_map.get(word, self.word_id_map.get(self.unk_token)))
                # fasttext ngram
                bigram = []
                trigram = []
                if self.enable_ngram:
                    buckets = self.n_gram_vocab
                    # ------ngram------
                    for i in range(max_seq_length):
                        bigram.append(biGramHash(words_line, i, buckets))
                        trigram.append(triGramHash(words_line, i, buckets))
                    # -----------------
                else:
                    bigram = [0] * max_seq_length
                    trigram = [0] * max_seq_length
                contents.append((words_line, 0, seq_len, bigram, trigram))
            return contents

        data = load_dataset(sentences, self.max_seq_length)
        data_iter = build_iterator(data, device, self.batch_size)
        # predict probs
        predict_all = np.array([], dtype=int)
        proba_all = np.array([], dtype=float)
        with torch.no_grad():
            for texts, _ in data_iter:
                outputs = self.model(texts)
                logit = F.softmax(outputs, dim=1).detach().cpu().numpy()
                pred = np.argmax(logit, axis=1)
                proba = np.max(logit, axis=1)

                predict_all = np.append(predict_all, pred)
                proba_all = np.append(proba_all, proba)
        id_label_map = {v: k for k, v in self.label_id_map.items()}
        predict_labels = [id_label_map.get(i) for i in predict_all]
        predict_probs = proba_all.tolist()
        return predict_labels, predict_probs

    def evaluate_model(self, data_list_or_path, header=None,
                       names=('labels', 'text'), delimiter='\t'):
        X_test, y_test, df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter)
        self.load_model()
        data, word_id_map, label_id_map = build_dataset(
            self.tokenizer, X_test, y_test,
            self.word_vocab_path,
            self.label_vocab_path,
            max_vocab_size=self.max_vocab_size,
            max_seq_length=self.max_seq_length,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            n_gram_vocab=self.n_gram_vocab,
            enable_ngram=self.enable_ngram
        )
        data_iter = build_iterator(data, device, self.batch_size)
        return self.evaluate(data_iter)[0]

    def evaluate(self, data_iter):
        """
        Evaluate model.
        @param data_iter:
        @return: accuracy score, loss
        """
        if not self.model:
            raise ValueError('model not trained.')
        self.model.eval()
        loss_total = 0.0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = self.model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.cpu().numpy()
                predic = torch.max(outputs, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
            logger.debug(f"evaluate, last batch, y_true: {labels}, y_pred: {predic}")
        acc = metrics.accuracy_score(labels_all, predict_all)
        return acc, loss_total / len(data_iter)

    def load_model(self):
        """
        Load model from model_dir
        @return:
        """
        model_path = os.path.join(self.model_dir, 'model.pth')
        if os.path.exists(model_path):
            self.word_vocab_path = os.path.join(self.model_dir, 'word_vocab.json')
            self.label_vocab_path = os.path.join(self.model_dir, 'label_vocab.json')
            self.word_id_map = load_vocab(self.word_vocab_path)
            self.label_id_map = load_vocab(self.label_vocab_path)
            vocab_size = len(self.word_id_map)
            num_classes = len(self.label_id_map)
            self.model = FastTextModel(
                vocab_size, num_classes, self.embed_size, self.n_gram_vocab, self.hidden_size,
                self.dropout_rate
            )
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.is_trained = True
        else:
            logger.error(f'{model_path} not exists.')
            self.is_trained = False
        return self.is_trained


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--model_dir', default='models/fasttext', type=str, help='save model dir')
    parser.add_argument('--data_path', default=os.path.join(pwd_path, '../examples/thucnews_train_1w.txt'),
                        type=str, help='sample data file path')
    args = parser.parse_args()
    print(args)
    # create model
    m = FastTextClassifier(args.model_dir)
    # train model
    m.train(data_list_or_path=args.data_path, num_epochs=3)
    # load trained best model and predict
    m.load_model()
    print('best model loaded from file, and predict')
    X, y, _ = load_data(args.data_path)
    X = X[:5]
    y = y[:5]
    predict_labels, predict_probs = m.predict(X)
    for text, pred_label, pred_prob, y_truth in zip(X, predict_labels, predict_probs, y):
        print(text, 'pred:', pred_label, pred_prob, ' truth:', y_truth)
