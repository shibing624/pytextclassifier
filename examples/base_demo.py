# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier import TextClassifier
from loguru import logger

logger.remove()  # Remove default log handler
logger.add(sys.stderr, level="INFO")  # 设置log级别

if __name__ == '__main__':
    m = TextClassifier(model_name='lr', model_dir='lr')
    # model_name is choose classifier, default lr, support lr, random_forest, textcnn, fasttext, textrnn_att, bert
    print(m)
    data = [
        ('education', 'Student debt to cost Britain billions within decades'),
        ('education', 'Chinese education for TV experiment'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
        ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
    ]
    # train and save best model
    m.train(data)
    new_m = TextClassifier(model_name='lr', model_dir='lr')
    # load best model from model_dir
    new_m.load_model()
    predict_label, predict_proba = new_m.predict([
        'Abbott government spends $8 million on higher education media blitz'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')

    test_data = [
        ('education', 'Abbott government spends $8 million on higher education media blitz'),
        ('sports', 'Middle East and Asia boost investment in top level sports'),
    ]
    acc_score = new_m.evaluate(test_data)
    print(f'acc_score: {acc_score}')
