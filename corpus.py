#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:Deal with corpus data:load 
"""

import os
import re


class Corpus:
    def __init__(self, file_path):
        root_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.normpath(os.path.join(root_path, file_path))

        re_split = re.compile("\s+")
        self.pos_doc_list = []
        self.neg_doc_list = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                splits = re_split.split(line.strip())
                if splits[0] == "pos":
                    self.pos_doc_list.append(splits[1:])
                elif splits[0] == "neg":
                    self.neg_doc_list.append(splits[1:])
                else:
                    raise ValueError("Corpus Error")

        self.doc_length = len(self.pos_doc_list)
        assert len(self.neg_doc_list) == self.doc_length
        self.train_num = 0
        self.test_num = 0
        out_content = "corpus:%s.\n" % file_path
        out_content += "positive:%d ; negative:%d" % (self.doc_length, self.doc_length)
        print(out_content)

    def get_corpus(self, start=0, end=-1):
        """
        Get corpus data
        :param start: 
        :param end: 
        :return: data,data_labels
        """
        assert self.doc_length >= self.test_num + self.train_num

        if end == -1:
            end = self.doc_length

        data = self.pos_doc_list[start:end] + self.neg_doc_list[start:end]
        data_labels = [1] * (end - start) + [0] * (end - start)
        return data, data_labels

    def get_train_corpus(self, num):
        self.train_num = num
        return self.get_corpus(end=num)

    def get_test_corpus(self, num):
        self.test_num = num
        return self.get_corpus(start=self.train_num, end=self.train_num + num)

    def get_all_corpus(self):
        data = self.pos_doc_list[:] + self.neg_doc_list[:]
        data_labels = [1] * self.doc_length + [0] * self.doc_length
        return data, data_labels


class WaimaiCorpus(Corpus):
    def __init__(self):
        Corpus.__init__(self, "data/corpus/ch_waimai_corpus.txt")


class MovieCorpus(Corpus):
    def __init__(self):
        Corpus.__init__(self, "data/corpus/en_movie_corpus.txt")

class HotelCorpus(Corpus):
    def __init__(self):
        Corpus.__init__(self,"data/corpus/ch_hotel_corpus.txt")