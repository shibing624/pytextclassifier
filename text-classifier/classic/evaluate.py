# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from matplotlib import pylab
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import config


def eval(model, test_data, test_label, thresholds=0.5, pr_figure_path=None, pred_save_path=None):
    print('{0}, val mean acc:{1}'.format(model.__str__(), model.score(test_data, test_label)))
    if (len(set(test_label))) == 2 and config.model_type != 'svm':
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
