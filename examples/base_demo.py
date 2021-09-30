# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import TextClassifier

if __name__ == '__main__':
    m = TextClassifier(model_name='bert')
    # model_name is choose classifier, support lr, random_forest, textcnn, fasttext, textrnn_att, bert
    print(m)
    data = [
        ('education', 'Student debt to cost Britain billions within decades'),
        ('education', 'Chinese education for TV experiment'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
        ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
    ]
    m.train(data)

    r = m.predict(['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports'])
    print(r)
    del m

    new_m = TextClassifier(model_name='textcnn')
    new_m.load_model()
    predict_label, predict_proba = new_m.predict(['Abbott government spends $8 million on higher education media blitz'])
    print(predict_label)
    print(predict_proba)  # [[0.53337174 0.46662826]]

    test_data = [
        ('education', 'Abbott government spends $8 million on higher education media blitz'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
    ]
    acc_score = new_m.evaluate(test_data)
    print(acc_score)  # 1.0
