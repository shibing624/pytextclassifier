# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: load data from file
"""
import sys

sys.path.append('..')
from pytextclassifier import TextClassifier, load_data

if __name__ == '__main__':
    m = TextClassifier(model_name='fasttext', model_dir='fasttext')
    # model_name is choose classifier, default lr, support lr, random_forest, textcnn, fasttext, textrnn_att, bert
    data_file = 'thucnews_train_1k.txt'
    m.train(data_file, names=('labels', 'text'), num_epochs=3)

    new_m = TextClassifier(model_name='fasttext', model_dir='fasttext')
    new_m.load_model() # load model from model_dir
    predict_label, predict_proba = new_m.predict(
        ['顺义北京苏活88平米起精装房在售',
         '美EB-5项目“15日快速移民”将推迟'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
    x, y, df = load_data(data_file)
    test_data = df[:100]
    acc_score = new_m.evaluate(test_data)
    print(f'acc_score: {acc_score}')
