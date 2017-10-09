# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 情感分析

import os
import sys
import jieba
from algorithm import bayes
from normal import filter_stop_words

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.marshal")


class Sentiment:
    def __init__(self):
        self.classifier = bayes.Bayes()

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip)

    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip)

    def handle(self, doc):
        words = jieba.lcut(doc)
        words = filter_stop_words(words)
        return words

    def train(self, neg_docs, pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.handle(sent), "neg"])
        for sent in pos_docs:
            data.append([self.handle(sent), "pos"])
        self.classifier.train(data)

    def classify(self, sent):
        ret, prob = self.classifier.classify(self.handle(sent))
        if ret == 'pos':
            return prob
        return 1 - prob


def train(neg_file, pos_file):
    neg_docs = []
    pos_docs = []
    with open(neg_file, mode="r", encoding="utf-8") as f:
        for line in f:
            neg_docs.append(line.strip())
    with open(pos_file, mode="r", encoding="utf-8") as f:
        for line in f:
            pos_docs.append(line.strip())
    global classifier
    classifier = Sentiment()
    classifier.train(neg_docs, pos_docs)


def save(fname, iszip=True):
    classifier.save(fname, iszip)


def load(fname, iszip=True):
    classifier.load(fname, iszip)


def classify(sent):
    classifier = Sentiment()
    classifier.load()
    return classifier.classify(sent)

# classifier = Sentiment()
# classifier.load()
