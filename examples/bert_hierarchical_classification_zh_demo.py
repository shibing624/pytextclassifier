# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pandas as pd

sys.path.append('..')
from pytextclassifier import BertClassifier


def load_baidu_data(file_path):
    """
    Load baidu data from file.
    @param file_path:
        format: content,labels
    @return:
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:
                continue
            terms = line.split('\t')
            if len(terms) != 2:
                continue
            data.append([terms[0], terms[1]])
    return data


if __name__ == '__main__':
    # model_type: support 'bert', 'albert', 'roberta', 'xlnet'
    # model_name: support 'bert-base-chinese', 'bert-base-cased', 'bert-base-multilingual-cased' ...
    m = BertClassifier(model_dir='models/hierarchical-bert-zh-model', num_classes=34,
                       model_type='bert', model_name='bert-base-chinese', num_epochs=2, multi_label=True)
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
    train_data = [
        ["马国明承认与黄心颖分手，女方被娱乐圈封杀，现如今朋友关系", "人生,人生##分手"],
        ["RedmiBook14集显版明天首发：酷睿i5+8G内存3799元", "产品行为,产品行为##发布"],
    ]
    data = load_baidu_data('baidu_extract_2020_train.csv')
    train_data.extend(data)
    print(train_data[:5])
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])

    print(train_df.head())
    m.train(train_df)
    print(m)
    # Evaluate the model
    acc_score = m.evaluate_model(train_df[:20])
    print(f'acc_score: {acc_score}')

    # load trained best model from model_dir
    m.load_model()
    predict_label, predict_proba = m.predict([
        '马国明承认与黄心颖分手，女方被娱乐圈封杀', 'RedmiBook14集显版明天首发'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
