# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import numpy as np
from matplotlib import pylab
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


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
    print('P:', pre, '\tR:', rec, '\tF:', f)
    print(classification_report(right_labels, pred_labels))
    return pre, rec, f


def eval(model, test_data, test_label, thresholds=0.5, num_classes=2, pr_figure_path=None, pred_save_path=None):
    print('{0}, val mean acc:{1}'.format(model.__str__(), model.score(test_data, test_label)))
    if num_classes == 2:
        # binary classification
        label_pred_probas = model.predict_proba(test_data)[:, 1]
        label_pred = label_pred_probas > thresholds
        precision, recall, threshold = precision_recall_curve(test_label, label_pred)
        plot_pr(thresholds, precision, recall, figure_path=pr_figure_path)
    else:
        # multi
        label_pred = model.predict(test_data)
    print(classification_report(test_label, label_pred))
    save(label_pred, pred_save_path)
    return label_pred


def plot_pr(auc_score, precision, recall, label=None, figure_path=None):
    """绘制R/P曲线"""
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.savefig(figure_path)


def save(label_pred, pred_save_path=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in label_pred:
                f.write(str(i) + '\n')
