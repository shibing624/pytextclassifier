# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import FastTextClassifier, load_data

if __name__ == '__main__':
    m = FastTextClassifier(output_dir='models/fasttext-toy', enable_ngram=False)
    data = [
        ('education', '名师指导托福语法技巧：名词的复数形式'),
        ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事？'),
        ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
        ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与'),
        ('sports', '米兰客场8战不败保持连胜'),
    ]
    m.train(data, num_epochs=3)
    print(m)
    # load trained best model
    m.load_model()
    predict_label, predict_proba = m.predict(['福建春季公务员考试报名18日截止 2月6日考试',
                                              '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
    test_data = [
        ('education', '福建春季公务员考试报名18日截止 2月6日考试'),
        ('sports', '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'),
    ]
    acc_score = m.evaluate_model(test_data)
    print(f'acc_score: {acc_score}')  # 1.0

    #### train model with 1w data
    print('-' * 42)
    data_file = 'thucnews_train_1w.txt'
    m = FastTextClassifier(output_dir='models/fasttext')
    m.train(data_file, names=('labels', 'text'), num_epochs=3)
    # load best trained model from output_dir
    m.load_model()
    predict_label, predict_proba = m.predict(
        ['顺义北京苏活88平米起精装房在售',
         '美EB-5项目“15日快速移民”将推迟']
    )
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
    x, y, df = load_data(data_file)
    test_data = df[:100]
    acc_score = m.evaluate_model(test_data)
    print(f'acc_score: {acc_score}')
