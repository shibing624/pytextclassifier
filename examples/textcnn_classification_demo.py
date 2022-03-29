# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import TextClassifier

if __name__ == '__main__':
    m = TextClassifier(model_name='textcnn', model_dir='textcnn')
    # model_name is choose classifier, default lr, support lr, random_forest, textcnn, fasttext, textrnn_att, bert
    data = [
        ('education', '名师指导托福语法技巧：名词的复数形式'),
        ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事？'),
        ('education', '北京公务员报名人数创历史新高'),
        ('education', '小学教育资源共享网络平台正式上线'),
        ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
        ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与'),
        ('sports', '米兰客场8战不败国米10年连胜')
    ]
    m.train(data, num_epochs=3)
    print(m)

    new_m = TextClassifier(model_name='textcnn', model_dir='textcnn')
    new_m.load_model()
    print(new_m)
    predict_label, predict_proba = new_m.predict(['福建春季公务员考试报名18日截止 2月6日考试',
                                                  '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')

    test_data = [
        ('education', '福建春季公务员考试报名18日截止 2月6日考试'),
        ('sports', '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'),
    ]
    acc_score = new_m.evaluate(test_data)
    print(f'acc_score: {acc_score}')  # 1.0
