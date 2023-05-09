# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 字节TNEWS新闻分类标准数据集评估模型
"""
import sys
import json

sys.path.append('..')
from pytextclassifier import BertClassifier

def convert_json_to_csv(path):
    lst = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip('\n'))
            lst.append((d['label_desc'], d['sentence']))
    return lst


if __name__ == '__main__':
    # train model with TNEWS data file
    train_file = './TNEWS/train.json'
    dev_file = './TNEWS/dev.json'
    train_data = convert_json_to_csv(train_file)[:None]
    dev_data = convert_json_to_csv(dev_file)[:None]
    print('train_data head:', train_data[:10])

    m = BertClassifier(output_dir='models/bert-tnews', num_classes=15,
                       model_type='bert', model_name='bert-base-chinese',
                       batch_size=32, num_epochs=5)
    m.train(train_data, test_size=0)
    m.load_model()
    # {"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物", "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    # {"label": "110", "label_desc": "news_military", "sentence": "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击", "keywords": "伊朗,圣城军,叙利亚,以色列国防军,以色列"}
    predict_label, predict_proba = m.predict(
        ['江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物',
         '以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击',
         ])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')

    score = m.evaluate_model(dev_data)
    print(f'score: {score}') # acc: 0.5643

