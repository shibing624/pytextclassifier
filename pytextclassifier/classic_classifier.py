# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Classic Classifier, support Naive Bayes, Logistic Regression, Random Forest, SVM, XGBoost
    and so on sklearn classification model
"""
import argparse
import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from loguru import logger

sys.path.append('..')
from pytextclassifier.base_classifier import ClassifierABC, load_data
from pytextclassifier.tokenizer import Tokenizer

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, 'stopwords.txt')


class ClassicClassifier(ClassifierABC):
    def __init__(self, model_dir, model_name_or_model='lr', feature_name_or_feature='tfidf',
                 stopwords_path=default_stopwords_path, tokenizer=None):
        """
        经典机器学习分类模型，支持lr, random_forest, decision_tree, knn, bayes, svm, xgboost
        @param model_dir: 模型保存路径
        @param model_name_or_model:
        @param feature_name_or_feature:
        @param stopwords_path:
        @param tokenizer: 切词器，默认为jieba切词
        """
        self.model_dir = model_dir
        if isinstance(model_name_or_model, str):
            model_name = model_name_or_model.lower()
            if model_name not in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
                raise ValueError('model_name not found.')
            logger.debug(f'model_name: {model_name}')
            self.model = self.get_model(model_name)
        elif hasattr(model_name_or_model, 'fit'):
            self.model = model_name_or_model
        else:
            raise ValueError('model_name_or_model set error.')
        if isinstance(feature_name_or_feature, str):
            feature_name = feature_name_or_feature.lower()
            if feature_name not in ['tfidf', 'count']:
                raise ValueError('feature_name not found.')
            logger.debug(f'feature_name: {feature_name}')
            if feature_name == 'tfidf':
                self.feature = TfidfVectorizer(ngram_range=(1, 2))
            else:
                self.feature = CountVectorizer(ngram_range=(1, 2))
        elif hasattr(feature_name_or_feature, 'fit_transform'):
            self.feature = feature_name_or_feature
        else:
            raise ValueError('feature_name_or_feature set error.')
        self.is_trained = False
        self.stopwords = set(self.load_list(stopwords_path)) if stopwords_path and os.path.exists(
            stopwords_path) else set()
        self.tokenizer = tokenizer if tokenizer else Tokenizer()

    def __str__(self):
        return f'ClassicClassifier instance ({self.model}, stopwords size: {len(self.stopwords)})'

    @staticmethod
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

    @staticmethod
    def load_list(path):
        return [word for word in open(path, 'r', encoding='utf-8').read().split()]

    def tokenize_sentences(self, sentences):
        """
        Tokenize input text
        :param sentences: list of text, eg: [text1, text2, ...]
        :return: X_tokens
        """
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in
                    sentences]
        return X_tokens

    def load_pkl(self, pkl_path):
        """
        加载词典文件
        :param pkl_path:
        :return:
        """
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        return result

    def save_pkl(self, vocab, pkl_path, overwrite=True):
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
                pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)  # python3

    def train(self, data_list_or_path, header=None, names=('labels', 'text'), delimiter='\t', test_size=0.1):
        """
        Train model with data_list_or_path and save model to model_dir
        @param data_list_or_path:
        @param header:
        @param names:
        @param delimiter:
        @param test_size:
        @return:
        """
        # load data
        X, y, data_df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter, is_train=True)
        # split validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        # train model
        logger.debug(f"X_train size: {len(X_train)}, X_test size: {len(X_test)}")
        assert len(X_train) == len(y_train)
        logger.debug(f'X_train sample:\n{X_train[:3]}\ny_train sample:\n{y_train[:3]}')
        logger.debug(f'num_classes:{len(set(y))}')
        # tokenize text
        X_train_tokens = self.tokenize_sentences(X_train)
        logger.debug(f'X_train_tokens sample:\n{X_train_tokens[:3]}')
        X_train_feat = self.feature.fit_transform(X_train_tokens)
        # fit
        self.model.fit(X_train_feat, y_train)
        self.is_trained = True
        # evaluate
        test_acc = self.evaluate(X_test, y_test)
        logger.debug(f'evaluate, X size: {len(X_test)}, y size: {len(y_test)}, acc: {test_acc}')
        # save model
        self.save_model()
        return test_acc

    def predict(self, sentences: list):
        """
        Predict labels and label probability for sentences.
        @param sentences: list, input text list, eg: [text1, text2, ...]
        @return: predict_label, predict_prob
        """
        if not self.is_trained:
            raise ValueError('model not trained.')
        # tokenize text
        X_tokens = self.tokenize_sentences(sentences)
        # transform
        X_feat = self.feature.transform(X_tokens)
        predict_labels = self.model.predict(X_feat)
        probs = self.model.predict_proba(X_feat)
        predict_probs = [prob[np.where(self.model.classes_ == label)][0] for label, prob in zip(predict_labels, probs)]
        return predict_labels, predict_probs

    def evaluate_model(self, data_list_or_path, header=None, names=('labels', 'text'), delimiter='\t'):
        X_test, y_test, df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter)
        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model.
        @param X_test:
        @param y_test:
        @return: accuracy score
        """
        if not self.is_trained:
            raise ValueError('model not trained.')
        # evaluate the model
        y_pred, _ = self.predict(X_test)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        return acc_score

    def load_model(self):
        """
        Load model from model_dir
        @return:
        """
        model_path = os.path.join(self.model_dir, 'classifier_model.pkl')
        if os.path.exists(model_path):
            self.model = self.load_pkl(model_path)
            feature_path = os.path.join(self.model_dir, 'classifier_feature.pkl')
            self.feature = self.load_pkl(feature_path)
            logger.info(f'Loaded model: {model_path}.')
            self.is_trained = True
        else:
            logger.error(f'{model_path} not exists.')
            self.is_trained = False
        return self.is_trained

    def save_model(self):
        """
        Save model to model_dir
        @return:
        """
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        if self.is_trained:
            feature_path = os.path.join(self.model_dir, 'classifier_feature.pkl')
            self.save_pkl(self.feature, feature_path)
            model_path = os.path.join(self.model_dir, 'classifier_model.pkl')
            self.save_pkl(self.model, model_path)
            logger.info(f'Saved model: {model_path}, feature_path: {feature_path}')
        else:
            logger.error('model is not trained, please train model first')
        return self.model, self.feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--model_name', default='lr', type=str, help='model name')
    parser.add_argument('--model_dir', default='models/lr', type=str, help='model dir')
    parser.add_argument('--feature_name', default='tfidf', type=str, help='feature name')
    parser.add_argument('--data_path', default=os.path.join(pwd_path, '../examples/thucnews_train_1w.txt'),
                        type=str, help='sample data file path')
    args = parser.parse_args()
    print(args)
    # create model
    m = ClassicClassifier(args.model_dir, model_name_or_model=args.model_name,
                          feature_name_or_feature=args.feature_name)
    # train model
    m.train(args.data_path)
    # load best trained model and predict
    m.load_model()
    X, y, _ = load_data(args.data_path)
    X = X[:5]
    y = y[:5]
    predict_labels, predict_probs = m.predict(X)
    for text, pred_label, pred_prob, y_truth in zip(X, predict_labels, predict_probs, y):
        print(text, 'pred:', pred_label, pred_prob, ' truth:', y_truth)
