# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import jieba
import sys

sys.path.append('../..')
from pytextclassifier.log import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')

tfidf = TfidfVectorizer(ngram_range=(1, 2))


def get_model(model_type):
    if model_type in ["lr", "logistic_regression"]:
        model = LogisticRegression(solver='lbfgs', fit_intercept=False)  # 快，准确率一般。val mean acc:0.91
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=300)  # 速度还行，准确率一般。val mean acc:0.93125
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()  # 速度快，准确率低。val mean acc:0.62
    elif model_type == "knn":
        model = KNeighborsClassifier()  # 速度一般，准确率低。val mean acc:0.675
    elif model_type == "bayes":
        model = MultinomialNB(alpha=0.1, fit_prior=False)  # 速度快，准确率低。val mean acc:0.62
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()  # 速度慢，准确率高。val mean acc:0.95
    elif model_type == "svm":
        model = SVC(kernel='linear', probability=True)  # 速度慢，准确率高，val mean acc:0.945
    else:
        raise ValueError('model type set error.')
    return model


lr = get_model('lr')


def load_list(path):
    return [word for word in open(path, 'r', encoding='utf-8').read().split()]


stopwords = set(load_list(default_stopwords_path))


def load_data(data_filepath, header=None, delimiter='\t', names=['labels', 'text']):
    data_df = pd.read_csv(data_filepath, header=header, delimiter=delimiter, names=names)
    X, y = data_df['text'], data_df['labels']
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


def train(X_train, y_train, model_dir='', model=None, vectorizer=None):
    """
    Train model
    """
    if vectorizer is None:
        vectorizer = tfidf
    if model is None:
        model = lr
    X_train_tokens = _encode_data(X_train)
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    # fit
    model.fit(X_train_vec, y_train)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
    save_pkl(vectorizer, vectorizer_path)
    model_path = os.path.join(model_dir, 'classifier_model.pkl')
    save_pkl(model, model_path)
    logger.debug(f'Saved model: {model_path}, vectorizer_path: {vectorizer_path}')
    return model, vectorizer


def predict(input_text_list, model, vectorizer):
    """
    Predict label
    :param input_text_list: list, input text list, eg: [text1, text2, ...]
    :return: list, label name
    """
    # tokenize text
    X_tokens = _encode_data(input_text_list)
    # transform
    X_vec = vectorizer.transform(X_tokens)
    predict_label = model.predict(X_vec)
    probas = model.predict_proba(X_vec)
    predict_proba = [prob[np.where(model.classes_ == label)][0] for label, prob in zip(predict_label, probas)]
    return predict_label, predict_proba


def evaluate(X_test, y_test, model, vectorizer):
    """
    Evaluate model with data
    :param data_list: list of (label, text), eg: [(label, text), (label, text) ...]
    :return: acc score
    """
    y_pred, _ = predict(X_test, model, vectorizer)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    return acc_score


def load_model(model_dir=''):
    """
    Load model from self.model_dir
    :return: None
    """
    model_path = os.path.join(model_dir, 'classifier_model.pkl')
    if model_path and not os.path.exists(model_path):
        raise ValueError("model is not found. please train and save model first.")
    model = load_pkl(model_path)
    vectorizer_path = os.path.join(model_dir, 'classifier_vectorizer.pkl')
    vectorizer = load_pkl(vectorizer_path)

    return model, vectorizer


def get_args():
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--model_dir', default='lr', type=str, help='save model dir')
    parser.add_argument('--data_path', default=os.path.join(pwd_path, '../../examples/thucnews_train_10w.txt'),
                        type=str, help='sample data file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    SEED = 1  # 保持结果一致
    # load data
    X, y = load_data(args.data_path)
    print(f'loaded data list, X size: {len(X)}, y size: {len(y)}')
    assert len(X) == len(y)
    print(f'num_classes:{len(set(y))}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    print(f"X_train size: {len(X_train)}, y_train size: {len(y_train)}")
    print(f'X_train:{X_train[:1]}, y_train:{y_train[:1]}')
    model, vectorizer = train(X_train, y_train, model_dir, model=lr, vectorizer=tfidf)

    # evaluate model
    print(f"X_test size: {len(X_test)}")
    acc_score = evaluate(X_test, y_test, model, vectorizer)
    print(f'evaluate model done, accuracy_score: {acc_score}')
    # predict
    predict_label, predict_proba = predict(X_train[:3], model, vectorizer)
    for text, pred_label, pred_proba in zip(X_train[:3], predict_label, predict_proba):
        print(text, pred_label, pred_proba)
    # load new model and predict
    new_model, new_vec = load_model(model_dir)
    print('model loaded {}'.format(model_dir))
    predict_label, predict_proba = predict(X_train[:3], new_model, new_vec)
    for text, pred_label, pred_proba in zip(X_train[:3], predict_label, predict_proba):
        print(text, pred_label, pred_proba)
