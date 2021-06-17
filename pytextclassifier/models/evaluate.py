# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import math

import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


def simple_evaluate(y_true, y_pred):
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
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:.4f}'.format(average_accuracy))
    print('overall_accuracy: {0:.4f}'.format(overall_accuracy))
    print('accuracy_score: {0:.4f}'.format(accuracy_score))
    return accuracy_score


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
        # precision_recall_curve: multiclass format is not supported
    print(classification_report(test_label, label_pred))
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in label_pred:
                f.write(str(i) + '\n')
    return label_pred


def plot_pr(auc_score, precision, recall, label=None, figure_path=None):
    """绘制R/P曲线"""
    try:
        from matplotlib import pylab
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
    except Exception as e:
        print("save image error with matplotlib")
        pass


def plt_history(history, output_dir='output/', model_name='cnn'):
    try:
        from matplotlib import pyplot
        model_name = model_name.upper()
        fig1 = pyplot.figure()
        pyplot.plot(history.history['loss'], 'r', linewidth=3.0)
        pyplot.plot(history.history['val_loss'], 'b', linewidth=3.0)
        pyplot.legend(['Training loss', 'Validation Loss'], fontsize=18)
        pyplot.xlabel('Epochs ', fontsize=16)
        pyplot.ylabel('Loss', fontsize=16)
        pyplot.title('Loss Curves :' + model_name, fontsize=16)
        loss_path = output_dir + model_name + '_loss.png'
        fig1.savefig(loss_path)
        print('save to:', loss_path)
        # pyplot.show()

        fig2 = pyplot.figure()
        pyplot.plot(history.history['acc'], 'r', linewidth=3.0)
        pyplot.plot(history.history['val_acc'], 'b', linewidth=3.0)
        pyplot.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        pyplot.xlabel('Epochs ', fontsize=16)
        pyplot.ylabel('Accuracy', fontsize=16)
        pyplot.title('Accuracy Curves : ' + model_name, fontsize=16)
        acc_path = output_dir + model_name + '_accuracy.png'
        fig2.savefig(acc_path)
        print('save to:', acc_path)
    except Exception as e:
        print("save image error with matplotlib")
        pass


def cal_multiclass_lr_predict(data_set, feature_weight_dict, id_label):
    """计算多分类预测结果, 返回预测值大于0.03的结果list, 每个元素[id, name, prob]
    [in]  data_set: 输入数据
          feature_weight_dict: 特征
          id_label: label分类
    [out] label_list: 预测结果
    """
    result = []
    for line in data_set:
        features = line.split(" ")
        label_pred = {}
        total = 0.0
        for id in id_label.keys():
            value = 0.0
            for word in features:
                if word not in feature_weight_dict:
                    continue
                cur_weight = float(feature_weight_dict[word].split(" ")[id])
                value += cur_weight
            label_prob = 1.0 / (1.0 + math.exp(-value))
            total += label_prob
            label_pred[id] = label_prob

        label_list = []
        for k, v in sorted(label_pred.items(), key=lambda x: x[1], reverse=True):
            val = "%.4f" % (v / total)
            if float(val) < 0.03:
                continue
            label_list.append((k, id_label[k], float(val)))
        result.append(label_list)
    return result
