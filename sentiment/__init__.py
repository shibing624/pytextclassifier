# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 
import jieba
import pypinyin
from analysis import sentiment


class NLP:
    def __init__(self, doc):
        self.doc = doc

    @property
    def words(self):
        return jieba.lcut(self.doc)

    @property
    def pinyin(self):
        return pypinyin.lazy_pinyin(self.doc)

    @property
    def sentiments(self):
        return sentiment.classify(self.doc)
