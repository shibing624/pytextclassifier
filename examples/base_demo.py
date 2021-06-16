# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import TextClassifier

if __name__ == '__main__':
    m = TextClassifier()
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
    m.save()
    del m

    new_m = TextClassifier()
    new_m.load()
    predict_label = new_m.predict(['Abbott government spends $8 million on higher education media blitz'])
    print(predict_label)  # ['education']

    predict_label = new_m.predict(['Abbott government spends $8 million on higher education media blitz',
                                   'Middle East and Asia boost investment in top level sports'])
    print(predict_label)  # ['education', 'sports']

    test_data = [
        ('education', 'Abbott government spends $8 million on higher education media blitz'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
    ]
    f1 = new_m.test(test_data)
    print(f1)  # 1.0
