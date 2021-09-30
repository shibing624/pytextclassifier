# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn.model_selection import train_test_split
import jieba

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')


def load_list(path):
    return [word for word in open(path, 'r', encoding='utf-8').read().split()]


def load_data(data_filepath, header=None, delimiter='\t', names=['labels', 'text'], **kwargs):
    data_df = pd.read_csv(data_filepath, header=header, delimiter=delimiter, names=names, **kwargs)
    print(data_df.head())
    X, y = data_df['text'], data_df['labels']
    print('loaded data list, X size: {}, y size: {}'.format(len(X), len(y)))
    assert len(X) == len(y)
    print('num_classes:%d' % len(set(y)))
    return X, y


def _encode_data(X):
    """
    Encoding data_list text
    X: list of text
    return: X_tokens
    """
    # tokenize text
    X_tokens = [' '.join([w for w in jieba.lcut(line) if w not in stopwords]) for line in X]
    return X_tokens


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def save_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            # pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(vocab, f, protocol=2)  # 兼容python2和python3
        print("save %s ok." % pkl_path)


def train(X_train, y_train):
    """
    Train model
    """
    print('train model...')
    X_train_tokens = _encode_data(X_train)
    print(f"X_train size: {len(X_train)}, y_train size: {len(y_train)}")
    print(f'data sample: X_tokens:{X_train_tokens[:1]}, y:{y_train[:1]}')
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    # fit
    model.fit(X_train_vec, y_train)
    print('train model done')
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
    save_pkl(vectorizer, vectorizer_path)
    model_path = os.path.join(model_dir, 'classifier_model.pkl')
    save_pkl(model, model_path)
    print('save done. vec path: {}, model path: {}'.format(vectorizer_path, model_path))


def predict(input_text_list):
    """
    Predict label
    :param input_text_list: list, input text list, eg: [text1, text2, ...]
    :return: list, label name
    """
    # tokenize text
    X_tokens = _encode_data(input_text_list)
    # transform
    X_vec = vectorizer.transform(X_tokens)
    return model.predict(X_vec)


def evaluate(X_test, y_test):
    """
    Evaluate model with data
    :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
    :return: acc score
    """
    print('evaluate model...')
    print(f"X_test size: {len(X_test)}")
    y_pred = predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    print('evaluate model done, accuracy_score: {}'.format(acc_score))


def predict_proba(input_text_list):
    """
    Predict proba
    :param input_text_list: list, input text list, eg: [text1, text2, ...]
    :return: list, accuracy score
    """
    # tokenize text
    X_tokens = _encode_data(input_text_list)
    # transform
    X_vec = vectorizer.transform(X_tokens)
    return model.predict_proba(X_vec)


def load():
    """
    Load model from self.model_dir
    :return: None
    """
    model_path = os.path.join(model_dir, 'classifier_model.pkl')
    if not os.path.exists(model_path):
        raise ValueError("model is not found. please train and save model first.")
    model = load_pkl(model_path)
    vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
    vectorizer = load_pkl(vectorizer_path)
    print('model loaded {}'.format(model_dir))
    return model, vectorizer

def get_args():
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--model_dir', default='lr', type=str, help='save model dir')
    parser.add_argument('--data_path', default='../../examples/THUCNews/data/train.txt', type=str,
                        help='sample data file path')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    model_dir = args.model_dir
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # create model
    model = LogisticRegression(solver='lbfgs', fit_intercept=False)  # 快，准确率一般。val mean acc:0.91
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    stopwords = set(load_list(default_stopwords_path))
    SEED = 1 # 保持结果一致
    # load data
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    train(X_train, y_train)
    evaluate(X_test, y_test)
    preds = predict(X_train[:3])
    for text, pred in zip(X_train[:3], preds):
        print(text, pred)
