#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:classifier based on K-Nearest Neighbours
"""

import numpy as np


class KNNClassifier:
    """
    KNN classifier 
    """
    def __init__(self, train_data, train_data_labels, k=3, best_words=None, stopwords=None):
        self.__train_data_labels = []
        self.__total_words = []
        self.__k = k
        self.__stopwords = stopwords
        self.__train_data_vectors = None
        self.__total_words_length = 0
        self.train_num = 0
        if train_data is not None:
            self.__train(train_data, train_data_labels, best_words)

    def set_k(self, k):
        self.__k = k

    def __doc2vector(self, doc):
        vector = [0] * self.__total_words_length
        for i in range(self.__total_words_length):
            vector[i] = doc.count(self.__total_words[i])
        length = sum(vector)
        if length == 0:
            return [0 for i in vector]
        # return vector
        return [i / length for i in vector]

    def __get_total_words(self, train_data, best_words):
        if best_words is not None:
            total_words = best_words[:]
        else:
            total_words = set()
            for doc in train_data:
                total_words |= set(doc)
        if self.__stopwords:
            with open(self.__stopwords, encoding="utf-8") as f:
                for line in f:
                    if line.strip() in total_words:
                        total_words.remove(line.strip())
        return list(total_words)

    @staticmethod
    def __normalize(vectors):
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        ranges = maxs - mins
        m = vectors.shape[0]
        norm_vectors = vectors - np.tile(mins, (m, 1))
        norm_vectors = norm_vectors / np.tile(ranges, (m, 1))
        return norm_vectors

    def __train(self, train_data, train_data_labels, best_words=None):
        print("training KNNClassifier...")
        self.__train_data_labels = train_data_labels[:]
        self.__total_words = self.__get_total_words(train_data, best_words)
        self.__total_words_length = len(self.__total_words)
        vectors = []
        for doc in train_data:
            vectors.append(self.__doc2vector(doc))
            self.train_num += 1

        self.__train_data_vectors = np.array(vectors)
        print("train KNNClassifier done.")

    def __get_sorted_distances(self, input_data):
        size = self.__train_data_vectors.shape
        vector = self.__doc2vector(input_data)
        input_data_vector = np.array(vector)
        diff_matrix = np.tile(input_data_vector, (size[0], 1)) - self.__train_data_vectors
        sq_diff_matrix = diff_matrix ** 2
        sq_distances = sq_diff_matrix.sum(axis=1)
        distances = sq_distances ** 0.5
        sorted_distances = distances.argsort()
        return sorted_distances

    def classify(self, input_data):
        if isinstance(self.__k, int):
            return self.single_k_classify(input_data)
        elif isinstance(self.__k, list):
            return self.multiple_k_classify(input_data)
        else:
            print("wrong k.")

    def single_k_classify(self, input_data):
        """
        single center of the data set
        :param input_data: 
        :return: 
        """
        sorted_distances = self.__get_sorted_distances(input_data)
        i = 0
        # class_count[0] records the number of label 0
        class_count = [0, 0]
        while i < self.__k:
            label = self.__train_data_labels[sorted_distances[i]]
            class_count[label] += 1
            i += 1

        if class_count[0] > class_count[1]:
            return 0
        else:
            return 1

    def multiple_k_classify(self, input_data):
        """
        multiple k center of the data set
        :param input_data: 
        :return: 
        """
        sorted_distances = self.__get_sorted_distances(input_data)
        i = 0
        class_count = [0, 0]
        final_record = [0, 0]
        assert type(self.__k) is list

        for k in sorted(self.__k):
            while i < k:
                label = self.__train_data_labels[sorted_distances[i]]
                class_count[label] += 1
                i += 1
            if class_count[0] > class_count[1]:
                final_record[0] += 1
            else:
                final_record[1] += 1
        if final_record[0] > final_record[1]:
            return 0
        else:
            return 1
