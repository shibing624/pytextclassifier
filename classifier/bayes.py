#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:Classifier based on Naive bayes
"""

import numpy as np


class BayesClassifier:
    def __init__(self, train_data, train_labels, best_words):
        self.__pos_word = {}
        self.__neg_word = {}
        self.__pos_p = 0.
        self.__neg_p = 1.
        self.__train(train_data, train_labels, best_words)

    def __train(self, train_data, train_labels, best_words=None):
        """
        Train data, select feature
        :param train_data: 
        :param train_labels: 
        :param best_words: 
        :return: 
        """
        print("training BayesClassifier...")
        total_pos_data, total_neg_data = {}, {}
        total_pos_length, total_neg_length = 0, 0
        total_word = set()
        for i, doc in enumerate(train_data):
            if train_labels[i] == 1:
                for word in doc:
                    if best_words is None or word in best_words:
                        total_pos_data[word] = total_pos_data.get(word, 0) + 1
                        total_pos_length += 1
                        total_word.add(word)
            else:
                for word in doc:
                    if best_words is None or word in best_words:
                        total_neg_data[word] = total_neg_data.get(word, 0) + 1
                        total_neg_length += 1
                        total_word.add(word)
        self.__pos_p = total_pos_length / (total_pos_length + total_neg_length)
        self.__neg_p = total_neg_length / (total_pos_length + total_neg_length)

        # get each word probability
        for word in total_word:
            self.__pos_word[word] = np.log(total_pos_data.get(word, 1e-100) / total_pos_length)
            self.__neg_word[word] = np.log(total_neg_data.get(word, 1e-100) / total_neg_length)
        print("train BayesClassifier done.")

    def classify(self, input_data):
        """
        Calculate the probability of each class by input data
        :param input_data: 
        :return: 
        """
        pos_score = 0.
        for word in input_data:
            pos_score += self.__pos_word.get(word, 0.)
        pos_score += np.log(self.__pos_p)

        neg_score = 0.
        for word in input_data:
            neg_score += self.__neg_word.get(word, 0.)
        neg_score += np.log(self.__neg_p)

        if pos_score > neg_score:
            return 1
        else:
            return 0
