# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import BertClassifier

if __name__ == '__main__':
    m = BertClassifier(model_dir='models/bert-english', model_type='bert', model_name='bert-base-uncased', num_epochs=2)
    data = [
        ('education', 'Student debt to cost Britain billions within decades'),
        ('education', 'Chinese education for TV experiment'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
        ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
    ]
    m.train(data)
    print(m)
    # load trained best model
    m.load_model()
    predict_label, predict_proba = m.predict(['Abbott government spends $8 million on higher education media blitz',
                                              'Middle East and Asia boost investment in top level sports'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')

    test_data = [
        ('education', 'Abbott government spends $8 million on higher education media blitz'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
    ]
    acc_score = m.evaluate_model(test_data)
    print(f'acc_score: {acc_score}')  # 1.0
