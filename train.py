# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import time

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import config
from models.classic_model import get_model
from models.deep_model import fasttext_model, cnn_model, rnn_model, han_model
from models.evaluate import eval, plt_history
from models.feature import Feature
from models.reader import data_reader
from models.xgboost_lr_model import XGBLR
from utils.data_utils import dump_pkl, write_vocab, load_pkl, build_vocab, load_vocab


def train_classic(model_type='logistic_regression',
                  data_path='',
                  model_save_path='',
                  vectorizer_path='',
                  col_sep='\t',
                  feature_type='tfidf_word',
                  min_count=1,
                  word_vocab_path='',
                  label_vocab_path='',
                  pr_figure_path=''):
    # load data
    data_content, data_lbl = data_reader(data_path, col_sep)
    word_lst = []
    for i in data_content:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_vocab(word_lst, min_count=min_count, sort=True, lower=True)
    # save word vocab
    write_vocab(word_vocab, word_vocab_path)
    # label
    label_vocab = build_vocab(data_lbl)
    # save label vocab
    write_vocab(label_vocab, label_vocab_path)
    label_id = load_vocab(label_vocab_path)
    print(label_id)
    data_label = [label_id[i] for i in data_lbl]
    num_classes = len(set(data_label))
    print('num_classes:', num_classes)

    # init feature
    if feature_type in ['doc_vectorize', 'vectorize']:
        print('feature type error. use tfidf_word replace.')
        feature_type = 'tfidf_word'
    feature = Feature(data=data_content, feature_type=feature_type,
                      feature_vec_path=vectorizer_path, min_count=min_count, word_vocab=word_vocab)
    # get data feature
    data_feature = feature.get_feature()

    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.1, random_state=0)
    if model_type == 'xgboost_lr':
        model = XGBLR(model_save_path=model_save_path)
    else:
        model = get_model(model_type)
    # fit
    model.fit(X_train, y_train)
    # save model
    if model_type != 'xgboost_lr':
        dump_pkl(model, model_save_path, overwrite=True)
    # analysis lr model
    if model_type == "logistic_regression" and config.is_debug:
        # print each category top features
        weights = model.coef_
        vectorizer = load_pkl(vectorizer_path)
        print("20 top features of each category:")
        features = dict()
        for idx, weight in enumerate(weights):
            feature_sorted = sorted(zip(vectorizer.get_feature_names(), weight), key=lambda k: k[1], reverse=True)
            print("category_" + str(idx) + ":")
            print(feature_sorted[:20])
            feature_dict = {k[0]: k[1] for k in feature_sorted}
            features[idx] = feature_dict
        dump_pkl(features, 'output/lr_features.pkl', overwrite=True)

    # evaluate
    eval(model, X_val, y_val, num_classes=num_classes, pr_figure_path=pr_figure_path)


def train_deep_model(model_type='cnn',
                     data_path='',
                     model_save_path='',
                     word_vocab_path='',
                     label_vocab_path='',
                     min_count=1,
                     max_len=300,
                     batch_size=128,
                     nb_epoch=10,
                     embedding_dim=128,
                     hidden_dim=128,
                     col_sep='\t',
                     num_filters=512,
                     filter_sizes='3,4,5',
                     dropout=0.5):
    # data reader
    data_content, data_lbl = data_reader(data_path, col_sep)
    word_lst = []
    for i in data_content:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_vocab(word_lst, min_count=min_count, sort=True, lower=True)
    write_vocab(word_vocab, word_vocab_path)

    # label
    label_vocab = build_vocab(data_lbl)
    write_vocab(label_vocab, label_vocab_path)
    label_id = load_vocab(label_vocab_path)
    print(label_id)
    data_label = [label_id[i] for i in data_lbl]
    # category
    num_classes = len(set(data_label))
    print('num_classes:', num_classes)
    data_label = to_categorical(data_label, num_classes=num_classes)
    print('Shape of Label Tensor:', data_label.shape)

    # init feature
    # han model need [doc sentence dim] feature(shape 3); others is [sentence dim] feature(shape 2)
    if model_type == 'han':
        print('Hierarchical Attention Network model feature_type must be: doc_vectorize')
        feature_type = 'doc_vectorize'
    else:
        print('feature_type: vectorize')
        feature_type = 'vectorize'
    feature = Feature(data=data_content, feature_type=feature_type, word_vocab=word_vocab, max_len=max_len)
    # get data feature
    data_feature = feature.get_feature()

    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.1, random_state=0)
    if model_type == 'fasttext':
        model = fasttext_model(max_len=max_len,
                               vocabulary_size=len(word_vocab),
                               embedding_dim=embedding_dim,
                               num_classes=num_classes)
    elif model_type == 'cnn':
        model = cnn_model(max_len,
                          vocabulary_size=len(word_vocab),
                          embedding_dim=embedding_dim,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes,
                          num_classses=num_classes,
                          dropout=dropout)
    elif model_type == 'rnn':
        model = rnn_model(max_len=max_len,
                          vocabulary_size=len(word_vocab),
                          embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim,
                          num_classes=num_classes)
    else:
        model = han_model(max_len=max_len,
                          vocabulary_size=len(word_vocab),
                          embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim,
                          num_classes=num_classes)
    cp = ModelCheckpoint(model_save_path, monitor='val_acc', verbose=1, save_best_only=True)
    # fit and save model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                        validation_data=(X_val, y_val), callbacks=[cp])
    print('save model:', model_save_path)
    plt_history(history, model_name=model_type)


if __name__ == '__main__':
    start_time = time.time()
    if config.model_type in ['fasttext', 'cnn', 'rnn', 'han']:
        train_deep_model(model_type=config.model_type,
                         data_path=config.train_seg_path,
                         model_save_path=config.model_save_path,
                         word_vocab_path=config.word_vocab_path,
                         label_vocab_path=config.label_vocab_path,
                         min_count=config.min_count,
                         max_len=config.max_len,
                         batch_size=config.batch_size,
                         nb_epoch=config.nb_epoch,
                         embedding_dim=config.embedding_dim,
                         hidden_dim=config.hidden_dim,
                         col_sep=config.col_sep,
                         dropout=config.dropout)
    else:
        train_classic(model_type=config.model_type,
                      data_path=config.train_seg_path,
                      model_save_path=config.model_save_path,
                      vectorizer_path=config.vectorizer_path,
                      col_sep=config.col_sep,
                      feature_type=config.feature_type,
                      min_count=config.min_count,
                      word_vocab_path=config.word_vocab_path,
                      label_vocab_path=config.label_vocab_path,
                      pr_figure_path=config.pr_figure_path)
    print("spend time %ds." % (time.time() - start_time))
    print("finish train.")
