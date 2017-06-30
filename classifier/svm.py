#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:Classifier based on Support Vector Machine
"""

from sklearn.svm import SVC
import numpy as np


class SVMClassifier:
    def __init__(self, train_data, train_labels, best_words, C):
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        self.best_wrods = best_words
        self.clf = SVC(C=C)
        self.__train(train_data, train_labels)

    def word2v(self, all_data):
        vectors = []
        for data in all_data:
            vector = []
            for feature in self.best_wrods:
                vector.append(data.count(feature))
            vectors.append(vector)
        vectors = np.array(vectors)
        return vectors

    def classify(self, data):
        vector = self.word2v([data])
        prediction = self.clf.predict(vector)
        return prediction[0]

    def __train(self, train_data, train_labels):
        print("training SVMClassifier...")
        train_vectors = self.word2v(train_data)
        self.clf.fit(train_vectors, np.array(train_labels))
        print("train SVMClassifier done.")

    def test(self, test_data):
        classify_labels = []
        print("testing SVMClassifier...")
        for data in test_data:
            classify_labels.append(self.classify(data))
        print("test SVMClassifier done.")
        return classify_labels
