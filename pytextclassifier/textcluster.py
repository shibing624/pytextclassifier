# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from codecs import open
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from pytextclassifier.tokenizer import Tokenizer

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_path = os.path.join(pwd_path, 'stopwords.txt')


class TextCluster(object):
    def __init__(
            self,
            model_dir,
            model=None, tokenizer=None, feature=None,
            stopwords_path=default_stopwords_path,
            n_clusters=3, n_init=10, ngram_range=(1, 2), **kwargs
    ):
        self.model_dir = model_dir
        self.model = model if model else MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.feature = feature if feature else TfidfVectorizer(ngram_range=ngram_range, **kwargs)
        self.stopwords = set(self.load_list(stopwords_path)) if stopwords_path and os.path.exists(
            stopwords_path) else set()
        self.is_trained = False

    def __str__(self):
        return 'TextCluster instance ({}, {}, {})'.format(self.model, self.tokenizer, self.feature)

    @staticmethod
    def load_file_data(file_path, sep='\t', use_col=1):
        """
        Load text file, format(txt): text
        :param file_path: str
        :param sep: \t
        :param use_col: int or None
        :return: list, text list
        """
        contents = []
        if not os.path.exists(file_path):
            raise ValueError('file not found. path: {}'.format(file_path))
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if use_col:
                    contents.append(line.split(sep)[use_col])
                else:
                    contents.append(line)
        logger.info('load file done. path: {}, size: {}'.format(file_path, len(contents)))
        return contents

    @staticmethod
    def load_list(path):
        """
        加载停用词
        :param path:
        :return: list
        """
        return [word for word in open(path, 'r', encoding='utf-8').read().split()]

    @staticmethod
    def load_pkl(pkl_path):
        """
        加载词典文件
        :param pkl_path:
        :return:
        """
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        return result

    @staticmethod
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
                pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(vocab, f, protocol=2)  # 兼容python2和python3
            print("save %s ok." % pkl_path)

    @staticmethod
    def show_plt(feature_matrix, labels, image_file='cluster.png'):
        """
        Show cluster plt
        :param feature_matrix:
        :param labels:
        :param image_file:
        :return:
        """
        from sklearn.decomposition import TruncatedSVD
        import matplotlib.pyplot as plt
        svd = TruncatedSVD()
        plot_columns = svd.fit_transform(feature_matrix)
        plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
        if image_file:
            plt.savefig(image_file)
        plt.show()

    def tokenize_sentences(self, sentences):
        """
        Encoding input text
        :param sentences: list of text, eg: [text1, text2, ...]
        :return: X_tokens
        """
        X_tokens = [' '.join([w for w in self.tokenizer.tokenize(line) if w not in self.stopwords]) for line in
                    sentences]
        return X_tokens

    def train(self, sentences):
        """
        Train model and save model
        :param sentences: list of text, eg: [text1, text2, ...]
        :return: model
        """
        logger.debug('train model')
        X_tokens = self.tokenize_sentences(sentences)
        logger.debug('data tokens top 1: {}'.format(X_tokens[:1]))
        feature = self.feature.fit_transform(X_tokens)
        # fit cluster
        self.model.fit(feature)
        labels = self.model.labels_
        logger.debug('cluster labels:{}'.format(labels))
        model_dir = self.model_dir
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        feature_path = os.path.join(model_dir, 'cluster_feature.pkl')
        self.save_pkl(self.feature, feature_path)
        model_path = os.path.join(model_dir, 'cluster_model.pkl')
        self.save_pkl(self.model, model_path)
        logger.info('save done. feature path: {}, model path: {}'.format(feature_path, model_path))

        self.is_trained = True
        return feature, labels

    def predict(self, X):
        """
        Predict label
        :param X: list, input text list, eg: [text1, text2, ...]
        :return: list, label name
        """
        if not self.is_trained:
            raise ValueError('model is None, run train first.')
        # tokenize text
        X_tokens = self.tokenize_sentences(X)
        # transform
        feat = self.feature.transform(X_tokens)
        return self.model.predict(feat)

    def show_clusters(self, feature_matrix, labels, image_file='cluster.png'):
        """
        Show cluster plt image
        :param feature_matrix:
        :param labels:
        :param image_file:
        :return:
        """
        if not self.is_trained:
            raise ValueError('model is None, run train first.')
        self.show_plt(feature_matrix, labels, image_file)

    def load_model(self):
        """
        Load model from model_dir
        :param model_dir: path
        :return: None
        """
        model_path = os.path.join(self.model_dir, 'cluster_model.pkl')
        if not os.path.exists(model_path):
            raise ValueError("model is not found. please train and save model first.")
        self.model = self.load_pkl(model_path)
        feature_path = os.path.join(self.model_dir, 'cluster_feature.pkl')
        self.feature = self.load_pkl(feature_path)
        self.is_trained = True
        logger.info('model loaded {}'.format(self.model_dir))
        return self.is_trained


if __name__ == '__main__':
    m = TextCluster(model_dir='models/cluster', n_clusters=2)
    print(m)
    data = [
        'Student debt to cost Britain billions within decades',
        'Chinese education for TV experiment',
        'Abbott government spends $8 million on higher education',
        'Middle East and Asia boost investment in top level sports',
        'Summit Series look launches HBO Canada sports doc series: Mudhar'
    ]
    feat, labels = m.train(data)
    m.show_clusters(feat, labels, image_file='models/cluster/cluster.png')
    m.load_model()
    r = m.predict(['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports'])
    print(r)
