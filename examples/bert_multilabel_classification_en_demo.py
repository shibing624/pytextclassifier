# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pandas as pd

sys.path.append('..')
from pytextclassifier import BertClassifier

if __name__ == '__main__':
    # model_type: support 'bert', 'albert', 'roberta', 'xlnet'
    # model_name: support 'bert-base-chinese', 'bert-base-cased', 'bert-base-multilingual-cased' ...
    m = BertClassifier(output_dir='models/multilabel-bert-en-toy', num_classes=6,
                       model_type='bert', model_name='bert-base-uncased', num_epochs=2, multi_label=True)
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
    train_data = [
        ["Example sentence 1 for multilabel classification.", [1, 1, 1, 1, 0, 1]],
        ["This is another example sentence. ", [0, 1, 1, 0, 0, 0]],
        ["This is the third example sentence. ", [1, 1, 1, 0, 0, 0]],
    ]
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])

    eval_data = [
        ["Example eval sentence for multilabel classification.", [1, 1, 1, 1, 0, 1]],
        ["Example eval senntence belonging to class 2", [0, 1, 1, 0, 0, 0]],
    ]
    eval_df = pd.DataFrame(eval_data, columns=["text", "labels"])
    print(train_df.head())
    # m.train(train_df)
    print(m)
    # Evaluate the model
    acc_score = m.evaluate_model(eval_df)
    print(f'acc_score: {acc_score}')

    # load trained best model from output_dir
    m.load_model()
    predict_label, predict_proba = m.predict(['some new sentence'])
    print(f'predict_label: {predict_label}, predict_proba: {predict_proba}')
