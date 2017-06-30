#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:evaluator of classification algorithm
"""

import datetime
import time
import os
from statistic_test import ChiSquare


class Evaluator:
    def __init__(self, evaluate_type, train_num, test_num, feature_num, max_iter, C, k, corpus):
        self.type = evaluate_type
        self.train_num = train_num
        self.test_num = test_num
        self.feature_num = feature_num
        self.max_iter = max_iter
        self.C = C
        self.k = k
        self.parameters = [train_num, test_num, feature_num]

        # get the corpus
        self.train_data, self.train_labels = corpus.get_train_corpus(train_num)
        self.test_data, self.test_labels = corpus.get_test_corpus(test_num)
        # out folder
        self.out_folder_path = "data/out/"
        # feature extraction test
        statistic_test = ChiSquare(self.train_data, self.train_labels)
        self.best_words = statistic_test.get_best_words(feature_num)

        # is got single k
        self.single_classifiers_got = False
        self.precisions = [[0, 0], [0, 0], [0, 0]]

    def set_precisions(self, precisions):
        self.precisions = precisions

    def write(self, file_path, classify_labels, i=-1):
        result = self.get_accuracy(self.test_labels, classify_labels, self.parameters)
        if i > 0:
            self.precisions[i][0] = result[10][1] / 100
            self.precisions[i][1] = result[7][1] / 100
        self.write_content(file_path, result)

    @staticmethod
    def get_accuracy(origin_labels, classify_labels, parameters):
        assert len(origin_labels) == len(classify_labels)

        contents = []

        contents.extend([("train num", parameters[0]), ("test num", parameters[1])])
        contents.append(("feature num", parameters[2]))

        pos_right, pos_false = 0, 0
        neg_right, neg_false = 0, 0
        for i in range(len(origin_labels)):
            if origin_labels[i] == 1:
                if classify_labels[i] == 1:
                    pos_right += 1
                else:
                    neg_false += 1
            else:
                if classify_labels[i] == 0:
                    neg_right += 1
                else:
                    pos_false += 1
        contents.extend([("neg-right", neg_right), ("neg-false", neg_false)])
        contents.extend([("pos-right", pos_right), ("pos-false", pos_false)])

        pos_precision = pos_right / (pos_right + pos_false) * 100
        pos_recall = pos_right / (pos_right + neg_false) * 100
        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
        contents.extend([("pos-precision", pos_precision), ("pos-recall", pos_recall), ("pos-f1", pos_f1)])

        neg_precision = neg_right / (neg_right + neg_false) * 100
        neg_recall = neg_right / (neg_right + pos_false) * 100
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        contents.extend([("neg-precision", neg_precision), ("neg-recall", neg_recall), ("neg-f1", neg_f1)])

        total_recall = (neg_right + pos_right) / (neg_right + neg_false + pos_right + pos_false) * 100
        contents.append(("total-recall", total_recall))

        print("pos-right\tpos-false\tneg-right\tneg-false\tpos-precision\tpos-recall\t"
              "pos-f1\tneg-precision\tneg-recall\tneg-f1\ttotal-recall")
        print("---" * 45)
        print("%8d\t%8d\t%8d\t%8d\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f" %
              (pos_right, pos_false, neg_right, neg_false, pos_precision, pos_recall,
               pos_f1, neg_precision, neg_recall, neg_f1, total_recall))

        return contents

    @staticmethod
    def write_content(file_path, contents):
        if os.path.exists(file_path):
            os.remove(file_path)
        if isinstance(contents, list) and isinstance(contents[0], tuple):
            with open(file_path, "w")as f:
                for i, (head, content) in enumerate(contents):
                    f.write(str(head) + ",")
                f.write("\n")
                for i, (head, content) in enumerate(contents):
                    f.writelines(str(content) + ",")

        elif isinstance(contents, list) and isinstance(contents[0], list) and \
                isinstance(contents[0][0], tuple):
            with open(file_path, "w") as f:
                i = 0
                # write the head
                for j, (head, k) in enumerate(contents[0]):
                    f.writelines(str(head) + ",")
                # write the content
                for temp_content in contents:
                    i += 1
                    for j, (k, content) in enumerate(temp_content):
                        f.writelines(str(content) + ",")
        else:
            print("out put file error.")

    def test_knn(self):
        from classifier.knn import KNNClassifier
        if isinstance(self.k, int):
            k = "%s" % self.k
        else:
            k = "-".join([str(i) for i in self.k])
        print("KNNClassifier")
        print("---" * 40)
        print("Train num: %s" % self.train_num)
        print("Test num: %s" % self.test_num)
        print("K = %s" % k)
        knn = KNNClassifier(self.train_data, self.train_labels, k=self.k, best_words=self.best_words)
        classify_labels = []
        print("testing KNNClassifier...")
        start = time.time()
        print("time: %s " % datetime.datetime.now())
        for data in self.test_data:
            classify_labels.append(knn.classify(data))
        print("test KNNClassifier done.")
        print("time: %s ; cost time: %s s" % (datetime.datetime.now(), (time.time() - start)))

        # Set file name with time string
        file_path = "KNN_%s_train_%d_test_%d_f_%d_k_%s_%s.out" % \
                    (self.type,
                     self.train_num, self.test_num,
                     self.feature_num, k,
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.write(self.out_folder_path + file_path, classify_labels)

    def test_bayes(self):
        print("BayesClassifier")
        print("---" * 40)
        print("Train num: %s" % self.train_num)
        print("Test num: %s" % self.test_num)
        from classifier.bayes import BayesClassifier
        bayes = BayesClassifier(self.train_data, self.train_labels, self.best_words)
        classify_labels = []
        print("testing BayesClassifier...")

        for data in self.test_data:
            classify_labels.append(bayes.classify(data))
        print("test BayesClassifier done.")
        file_path = "Bayes_%s_train_%d_test_%d_f_%d_%s.out" % \
                    (self.type, self.train_num, self.test_num, self.feature_num,
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.write(self.out_folder_path + file_path, classify_labels, 0)

    def test_maxent(self):
        print("MaxEntClassifier")
        print("---" * 40)
        print("Train num: %s" % self.train_num)
        print("Test num: %s" % self.test_num)
        print("max iter: %s" % self.max_iter)
        from classifier.maxent import MaxEntClassifier
        maxent = MaxEntClassifier(self.max_iter)
        maxent.train(self.train_data, self.train_labels, self.best_words)
        classify_results = maxent.test(self.test_data)
        file_path = "MaxEnt_%s_train_%d_test_%d_f_%d_maxiter_%d_%s.out" % \
                    (self.type,
                     self.train_num, self.test_num,
                     self.feature_num, self.max_iter,
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.write(self.out_folder_path + file_path, classify_results, 1)

    def test_svm(self):
        print("SVMClassifier")
        print("---" * 40)
        print("Train num: %s" % self.train_num)
        print("Test num: %s" % self.test_num)
        print("C: %s" % self.C)
        from classifier.svm import SVMClassifier
        svm = SVMClassifier(self.train_data, self.train_labels, self.best_words, self.C)

        classify_labels = svm.test(self.test_data)
        file_path = "SVM_%s_train_%d_test_%d_f_%d_C_%d_%s.out" % \
                    (self.type,
                     self.train_num, self.test_num,
                     self.feature_num, self.C,
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.write(self.out_folder_path + file_path, classify_labels, 2)
