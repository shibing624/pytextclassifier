#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:Classifier based on Maximum Entropy
"""
import numpy as np
from collections import defaultdict


class MaxEntClassifier:
    def __init__(self, max_iter=500):
        self.feats = defaultdict(int)
        self.labels = {0, 1}
        self.weight = []
        self.max_iter = max_iter

    def prob_weight(self, features, label):
        weight = 0.0
        for feature in features:
            if (label, feature) in self.feats:
                weight += self.weight[self.feats[(label, feature)]]
        return np.exp(weight)

    def calculate_probability(self, features):
        weights = [(self.prob_weight(features, label), label) for label in self.labels]
        try:
            z = sum([weight for weight, label in weights])
            prob = [(weight / z, label) for weight, label in weights]
        except ZeroDivisionError:
            return "collapse"
        return prob

    def convergence(self, last_weight):
        for i, j in zip(last_weight, self.weight):
            if abs(i - j) >= 0.001:
                return False
            return True

    def classify(self, input_features):
        prob = self.calculate_probability(input_features)
        prob.sort(reverse=True)
        if prob[0][0] > prob[1][0]:
            return prob[0][1]
        else:
            return prob[1][1]

    def train(self, train_data, train_labels, best_words=None):
        print("training MaxEntClassifier...")
        length = len(train_labels)
        if best_words is None:
            for i in range(length):
                for word in set(train_data[i]):
                    self.feats[(train_labels[i], word)] += 1
        else:
            for i in range(length):
                for word in set(train_data[i]):
                    if word in best_words:
                        self.feats[(train_labels[i], word)] += 1
        max_num = max([len(record) - 1 for record in train_data])
        # weight for each feature
        self.weight = [0.] * len(self.feats)
        # init the feature expectation on empirical distribution
        ep_empirical = [0.] * len(self.feats)
        for i, f in enumerate(self.feats):
            ep_empirical[i] = self.feats[f] / length
            # each feature correspond to id
            self.feats[f] = i
        for i in range(self.max_iter):
            ep_model = [0.] * len(self.feats)
            for doc in train_data:
                prob = self.calculate_probability(doc)
                if prob == "collapse":
                    print("program collapse. the iter number : %d." % (i + 1))
                    return
                for feature in doc:
                    for weight, label in prob:
                        if (label, feature) in self.feats:
                            # get feature id
                            idx = self.feats[(label, feature)]
                            # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                            ep_model[idx] += weight * (1. / length)

            last_weight = self.weight[:]
            for j, win in enumerate(self.weight):
                delta = 1. / max_num * np.log(ep_empirical[j] / ep_model[j])
                # update weight
                self.weight[j] += delta

            # test the algorithm is convergence
            if self.convergence(last_weight):
                print("program convergence. the iter number : %d." % (i + 1))
                break
        print("train MaxEntClassifier done.")

    def test(self, test_data):
        classify_results = []
        print("testing MaxEntClassifier...")
        for i in range(self.max_iter):
            classify_labels = []
            for data in test_data:
                classify_labels.append(self.classify(data))
            classify_results.append(classify_labels)
        print("test MaxEntClassifier done.")
        return classify_results
