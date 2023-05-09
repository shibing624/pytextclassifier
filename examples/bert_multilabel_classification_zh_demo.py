# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pandas as pd

sys.path.append('..')
from pytextclassifier import BertClassifier


def load_jd_data(file_path):
    """
    Load jd data from file.
    @param file_path:
        format: content,其他,互联互通,产品功耗,滑轮提手,声音,APP操控性,呼吸灯,外观,底座,制热范围,遥控器电池,味道,制热效果,衣物烘干,体积大小
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
            terms = line.split(',')
            if len(terms) != 16:
                continue
            val = [int(i) for i in terms[1:]]
            data.append([terms[0], val])
    return data


if __name__ == '__main__':
    # model_type: support 'bert', 'albert', 'roberta', 'xlnet'
    # model_name: support 'bert-base-chinese', 'bert-base-cased', 'bert-base-multilingual-cased' ...
    m = BertClassifier(output_dir='models/multilabel-bert-zh-model', num_classes=15,
                       model_type='bert', model_name='bert-base-chinese', num_epochs=2, multi_label=True)
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
    train_data = [
        ["一个小时房间仍然没暖和", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        ["耗电情况：这个没有注意", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    ]
    data = load_jd_data('multilabel_jd_comments.csv')
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
    predict_label, predict_proba = m.predict(['一个小时房间仍然没暖和', '耗电情况：这个没有注意'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
