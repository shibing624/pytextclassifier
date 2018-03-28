# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
from codecs import open

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from util import dump_pkl
from util import load_pkl


def data_reader(path):
    label_list, content_list = [], []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            if parts and len(parts) > 1:
                content_list.append(parts[1].strip())
                label_list.append(parts[0].strip())
    return content_list, label_list


def tfidf(data_set, space_path):
    if os.path.exists(space_path):
        tfidf_space = load_pkl(space_path)
    else:
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf_space = transformer.fit_transform(vectorizer.fit_transform(data_set))
        dump_pkl(tfidf_space, space_path)
    print('tfidf shape:', tfidf_space.shape)
    return tfidf_space


def label_encoder(labels):
    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    return corpus_encode_label


def lr(data_set, data_label):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(data_set, data_label)
    return model


def randomForest(data_set, data_label):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(data_set, data_label)
    return model


def gbdt(data_set, data_label):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(data_set, data_label)
    return model


def eval(model, test_data, test_label, thresholds=0.5, pr_figure_path=None):
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_curve
    print('{0}, val mean acc:{1}'.format(model.__str__(), model.score(test_data, test_label)))
    label_pred = model.predict(test_data)
    print(classification_report(test_label, label_pred, target_names=['approve', 'disapprove']))
    label_pred_prob = model.predict_proba(test_data)[:, 1]
    with open("pred.txt", 'w', 'utf-8') as f:
        for i in label_pred_prob:
            f.write(str(i))
            f.write('\n')
    label_pred = label_pred_prob > thresholds
    print(classification_report(test_label, label_pred, target_names=['approve', 'disapprove']))
    precision, recall, threshold = precision_recall_curve(test_label, label_pred_prob)
    plot_pr(thresholds, precision, recall, label='disapprove', figure_path=pr_figure_path)


def plot_pr(auc_score, precision, recall, label=None, figure_path=None):
    """绘制R/P曲线"""
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


if __name__ == '__main__':
    # data
    data_path = "../data/risk/data_seg.txt"  # 输入的文件
    space_path = "../data/risk/tfidf.dat"  # 输出的文件
    pr_figure_path = "../data/risk/R_P.png"  # 保存P_R曲线图

    data_content, data_lbl = data_reader(data_path)
    # data feature
    data_tfidf = tfidf(data_content, space_path)
    # label
    data_label = label_encoder(data_lbl)
    X_train, X_test, y_train, y_test = train_test_split(
        data_tfidf, data_label, test_size=0.01, random_state=42)

    # fit and eval
    model = lr(X_train, y_train)
    eval(model, X_test, y_test, 0.65, pr_figure_path)  # 快，准确率一般。val mean acc:0.912

    # # fit and eval
    # model = randomForest(train_feature, train_label_encode)
    # eval(model, test_feature, test_label_encode)  # 耗时，准确率高， val mean acc:0.962
    #
    # # fit and eval
    # model = gbdt(train_feature, train_label_encode)
    # eval(model, test_feature, test_label_encode)  # 耗时，准确率低。val mean acc:0.803
