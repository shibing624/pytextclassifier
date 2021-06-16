# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import time

from sklearn.model_selection import train_test_split

from pytextclassifier import config
from pytextclassifier.models.classic_model import get_model
from pytextclassifier.models.evaluate import eval, plt_history
from pytextclassifier.models.feature import Feature
from pytextclassifier.models.reader import data_reader
from pytextclassifier.models.xgboost_lr_model import XGBLR
from pytextclassifier.utils.data_utils import save_pkl, write_vocab, build_vocab, load_vocab, save_dict
from pytextclassifier.utils.log import logger


def train_classic(model_type='logistic_regression',
                  data_path='',
                  model_save_path='',
                  feature_vec_path='',
                  col_sep='\t',
                  feature_type='tfidf_word',
                  min_count=1,
                  word_vocab_path='',
                  label_vocab_path='',
                  pr_figure_path=''):
    logger.info("train classic model, model_type:{}, feature_type:{}".format(model_type, feature_type))
    # load data
    data_content, data_lbl = data_reader(data_path, col_sep)
    word_lst = []
    for i in data_content:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_vocab(word_lst, min_count=min_count, sort=True, lower=True)
    # save word vocab
    write_vocab(word_vocab, word_vocab_path)
    word_id = load_vocab(word_vocab_path)
    # label
    label_vocab = build_vocab(data_lbl)
    # save label vocab
    write_vocab(label_vocab, label_vocab_path)
    label_id = load_vocab(label_vocab_path)
    print(label_id)
    data_label = [label_id[i] for i in data_lbl]
    num_classes = len(set(data_label))
    logger.info('num_classes:%d' % num_classes)
    logger.info('data size:%d' % len(data_content))
    logger.info('label size:%d' % len(data_lbl))

    # init feature
    if feature_type in ['doc_vectorize', 'vectorize']:
        logger.error('feature type error. use tfidf_word replace.')
        feature_type = 'tfidf_word'
    feature = Feature(data=data_content, feature_type=feature_type,
                      feature_vec_path=feature_vec_path, word_vocab=word_vocab, is_infer=False)
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
        save_pkl(model, model_save_path, overwrite=True)
    # evaluate
    eval(model, X_val, y_val, num_classes=num_classes, pr_figure_path=pr_figure_path)

    # analysis lr model
    if config.debug and model_type == "logistic_regression":
        feature_weight = {}
        word_dict_rev = sorted(word_id.items(), key=lambda x: x[1])
        for feature, index in word_dict_rev:
            feature_weight[feature] = list(map(float, model.coef_[:, index]))
        save_dict(feature_weight, config.lr_feature_weight_path)


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
    from keras.callbacks import ModelCheckpoint
    from keras.utils import to_categorical

    from pytextclassifier.models.deep_model import fasttext_model, cnn_model, rnn_model, han_model
    logger.info("train deep model, model_type:{}, data_path:{}".format(model_type, data_path))
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
    logger.info(label_id)
    data_label = [label_id[i] for i in data_lbl]
    # category
    num_classes = len(set(data_label))
    logger.info('num_classes:%s' % num_classes)
    data_label = to_categorical(data_label, num_classes=num_classes)
    print('Shape of Label Tensor:', data_label.shape)

    # init feature
    # han model need [doc sentence dim] feature(shape 3); others is [sentence dim] feature(shape 2)
    if model_type == 'han':
        logger.warn('Hierarchical Attention Network model feature_type must be: doc_vectorize')
        feature_type = 'doc_vectorize'
    else:
        logger.warn('feature_type: vectorize')
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
    logger.info('save model:%s' % model_save_path)
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
                      feature_vec_path=config.feature_vec_path,
                      col_sep=config.col_sep,
                      feature_type=config.feature_type,
                      min_count=config.min_count,
                      word_vocab_path=config.word_vocab_path,
                      label_vocab_path=config.label_vocab_path,
                      pr_figure_path=config.pr_figure_path)
    logger.info("spend time %s s." % (time.time() - start_time))
    logger.info("finish train.")
