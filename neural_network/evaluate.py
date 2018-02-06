# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: evaluate model precision and recall rate
import numpy as np
from sklearn import metrics


def evaluate(y_true, y_pred):
    """
    evaluate precision, recall, f1
    :param y_true:
    :param y_pred:
    :return:score
    """
    assert len(y_true) == len(y_pred), \
        "the count of pred label should be same with true label"

    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))

    return score


def demo():
    pred_labels = [2, 2, 1, 1, 2, 3, 1, 2, 2, 2]
    true_labels = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    # pred_labels = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    # true_labels = [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    score = evaluate(true_labels, pred_labels)
    print("score: %f" % score)


if __name__ == "__main__":
    demo()
