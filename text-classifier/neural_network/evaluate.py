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


def simple_evaluate(right_labels, pred_labels, ignore_label=None):
    """
    simple evaluate
    :param right_labels: right labels
    :param pred_labels: predict labels
    :param ignore_label: the label should be ignored
    :return: pre, rec, f
    """
    assert len(pred_labels) == len(right_labels)
    pre_pro_labels, pre_right_labels = [], []
    rec_pro_labels, rec_right_labels = [], []
    labels_len = len(pred_labels)
    for i in range(labels_len):
        pro_label = pred_labels[i]
        if pro_label != ignore_label:  #
            pre_pro_labels.append(pro_label)
            pre_right_labels.append(right_labels[i])
        if right_labels[i] != ignore_label:
            rec_pro_labels.append(pro_label)
            rec_right_labels.append(right_labels[i])
    pre_pro_labels, pre_right_labels = np.array(pre_pro_labels, dtype='int32'), \
                                       np.array(pre_right_labels, dtype='int32')
    rec_pro_labels, rec_right_labels = np.array(rec_pro_labels, dtype='int32'), \
                                       np.array(rec_right_labels, dtype='int32')
    pre = 0. if len(pre_pro_labels) == 0 \
        else len(np.where(pre_pro_labels == pre_right_labels)[0]) / float(len(pre_pro_labels))
    rec = len(np.where(rec_pro_labels == rec_right_labels)[0]) / float(len(rec_right_labels))
    f = 0. if (pre + rec) == 0. \
        else (pre * rec * 2.) / (pre + rec)
    return pre, rec, f


def demo():
    # pred_labels = [1, 2, 1, 1, 1, 3, 1, 2, 2, 2]
    # true_labels = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    pred_labels = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    true_labels = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    p, r, f = simple_evaluate(true_labels, pred_labels)
    print("p: %f,r:%f,f:%f" % (p, r, f))


if __name__ == "__main__":
    demo()
