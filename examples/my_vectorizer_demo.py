# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import ClassicClassifier
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    vec = CountVectorizer(ngram_range=(1, 3))
    m = ClassicClassifier(model_dir='models/lr-vec', model_name_or_model='lr', feature_name_or_feature=vec)

    data = [
        ('education', '名师指导托福语法技巧：名词的复数形式'),
        ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事？'),
        ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
        ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与'),
        ('sports', '米兰客场8战不败国米10年连胜'),
    ]
    m.train(data)
    m.load_model()
    predict_label, predict_label_prob = m.predict(['福建春季公务员考试报名18日截止 2月6日考试'])
    print(predict_label, predict_label_prob)
    print('classes_: ', m.model.classes_)  # the classes ordered as prob

    test_data = [
        ('education', '福建春季公务员考试报名18日截止 2月6日考试'),
        ('sports', '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'),
    ]
    acc_score = m.evaluate_model(test_data)
    print(acc_score)  # 1.0
