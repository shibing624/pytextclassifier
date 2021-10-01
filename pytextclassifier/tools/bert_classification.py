# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import pandas as pd
import os
import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import sys

sys.path.append('../..')
from pytextclassifier.log import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(data_filepath, header=None, delimiter='\t', names=['labels', 'text'], **kwargs):
    data_df = pd.read_csv(data_filepath, header=header, delimiter=delimiter, names=names, **kwargs)
    X, y = data_df['text'], data_df['labels']
    logger.debug(f'loaded data list, X size: {len(X)}, y size: {len(y)}')
    assert len(X) == len(y)
    logger.debug(f'num_classes:{len(set(y))}')
    return data_df


def build_dataset(data_df, label_vocab_path):
    X, y = data_df['text'], data_df['labels']
    if os.path.exists(label_vocab_path):
        label_id_map = pickle.load(open(label_vocab_path, 'rb'))
    else:
        id_label_map = {id: v for id, v in enumerate(set(y.tolist()))}
        label_id_map = {v: k for k, v in id_label_map.items()}
        pickle.dump(label_id_map, open(label_vocab_path, 'wb'))
    logger.debug(f"label vocab size: {len(label_id_map)}")
    df = data_df.copy()
    df.loc[:, 'labels'] = df.loc[:, 'labels'].map(lambda x: label_id_map.get(x))
    data_df = df
    return data_df, label_id_map


class BertClassificationModel(ClassificationModel):
    """Bert + fc model"""

    def __init__(self, model_type='bert',
                 model_name='bert-base-chinese',
                 num_classes=10,
                 num_epochs=3,
                 batch_size=64,
                 max_seq_length=128,
                 model_dir='bert',
                 use_cuda=False
                 ):
        train_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "output_dir": model_dir,
            "max_seq_length": max_seq_length,
            "num_train_epochs": num_epochs,
            "train_batch_size": batch_size,
        }
        super(BertClassificationModel, self).__init__(model_name=model_name,
                                                      model_type=model_type,
                                                      num_labels=num_classes,
                                                      args=train_args,
                                                      use_cuda=use_cuda)


def predict(model, data_list, label_id_map):
    # predict
    predictions, raw_outputs = model.predict(data_list)
    # predict proba
    id_label_map = {v: k for k, v in label_id_map.items()}
    predict_label = [id_label_map.get(i) for i in predictions]
    predict_proba = [np.exp(-raw_output[prediction]) for raw_output, prediction in zip(raw_outputs, predictions)]
    return predict_label, predict_proba


def get_args():
    parser = argparse.ArgumentParser(description='Bert Text Classification')
    parser.add_argument('--pretrain_model_type', default='bert', type=str,
                        help='pretrained huggingface model type')
    parser.add_argument('--pretrain_model_name', default='bert-base-chinese', type=str,
                        help='pretrained huggingface model name')
    parser.add_argument('--model_dir', default='bert', type=str, help='save model dir')
    parser.add_argument('--data_path', default=os.path.join(pwd_path, '../../examples/thucnews_train_10w.txt'),
                        type=str, help='sample data file path')
    parser.add_argument('--num_epochs', default=3, type=int, help='train epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--max_seq_length', default=128, type=int, help='max seq length, trim longer sentence.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 保持结果一致
    # load data
    label_vocab_path = os.path.join(model_dir, 'label_vocab.pkl')
    data_df = load_data(args.data_path)
    data_df, label_id_map = build_dataset(data_df, label_vocab_path)
    print(data_df.head())
    train_df, dev_df = train_test_split(data_df, test_size=0.1, random_state=SEED)
    # create model
    use_cuda = False if device == torch.device('cpu') else True
    print(f'device: {device}, use_cuda: {use_cuda}')
    model = BertClassificationModel(model_type=args.pretrain_model_type,
                                    model_name=args.pretrain_model_name,
                                    num_classes=len(label_id_map),
                                    num_epochs=args.num_epochs,
                                    batch_size=args.batch_size,
                                    max_seq_length=args.max_seq_length,
                                    model_dir=args.model_dir,
                                    use_cuda=use_cuda)
    print(model)
    # train model
    # Train and Evaluation data needs to be in a Pandas Dataframe,
    # it should contain a 'text' and a 'labels' column. text with type str, the label with type int.
    model.train_model(train_df)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(dev_df)
    print('evaluate: ', result, model_outputs, wrong_predictions)
    # predict
    predictions, raw_outputs = model.predict(["就要性价比 惠普CQ40仅3800元抱回家"])
    print('pred:', predictions, ' raw_output:', raw_outputs)
    predict_label, predict_proba = predict(model, ["就要性价比 惠普CQ40仅3800元抱回家"], label_id_map)
    print(f'predict_label:{predict_label}, predict_proba:{predict_proba}')
    # predict with new model
    new_model = BertClassificationModel(model_type=args.pretrain_model_type,
                                        model_name='./bert/best_model/',
                                        num_classes=len(label_id_map),
                                        num_epochs=args.num_epochs,
                                        batch_size=args.batch_size,
                                        max_seq_length=args.max_seq_length,
                                        model_dir=args.model_dir,
                                        use_cuda=use_cuda)
    predict_label, predict_proba = predict(new_model, ["就要性价比 惠普CQ40仅3800元抱回家"], label_id_map)
    print(f'predict_label:{predict_label}, predict_proba:{predict_proba}')
