# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os

from sklearn.model_selection import train_test_split

import config
from models.build_w2v import build
from models.classic_model import get_model
from models.cnn_model import Model
from models.evaluate import eval, simple_evaluate
from models.feature import Feature
from models.reader import build_dict
from models.reader import build_pos_embedding
from models.reader import build_vocab
from models.reader import build_word_embedding
from models.reader import data_reader
from models.reader import test_reader
from models.reader import train_reader
from models.xgboost_lr_model import XGBLR
from utils.data_utils import dump_pkl, write_vocab, load_pkl
from utils.io_utils import clear_directory


def train_classic(model_type,
                  data_path='',
                  pr_figure_path='',
                  model_save_path='',
                  vectorizer_path='',
                  col_sep='\t',
                  thresholds=0.5,
                  num_classes=2,
                  feature_type='tfidf_word',
                  min_count=1,
                  word_vocab_path=''):
    data_content, data_lbl = data_reader(data_path, col_sep)
    word_lst = []
    for i in data_content:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_dict(word_lst, start=0,
                            min_count=min_count, sort=True, lower=True)
    write_vocab(word_vocab, word_vocab_path)
    # init feature
    feature = Feature(data=data_content, feature_type=feature_type,
                      feature_vec_path=vectorizer_path, min_count=min_count, word_vocab=word_vocab)
    # get data feature
    data_feature = feature.get_feature()
    # label
    data_label = [int(i) for i in data_lbl]

    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.1, random_state=0)
    model = get_model(model_type)
    # fit
    model.fit(X_train, y_train)
    # save model
    dump_pkl(model, model_save_path, overwrite=True)

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
        dump_pkl(features, 'output/features.pkl', overwrite=True)
    # evaluate
    eval(model, X_val, y_val, thresholds=thresholds, num_classes=num_classes, pr_figure_path=pr_figure_path)


def train_cnn(train_seg_path='', test_seg_path='', word_vocab_path='',
              pos_vocab_path='', label_vocab_path='', sentence_w2v_path='',
              sentence_w2v_bin_path='', sentence_path='', w2v_path='', p2v_path='',
              word_vocab_start=2, pos_vocab_start=1,
              w2v_dim=256, pos_dim=64, max_len=300, min_count=5,
              model_save_temp_dir='',
              output_dir='',
              batch_size=128,
              nb_epoch=5,
              keep_prob=0.5,
              word_keep_prob=0.9,
              pos_keep_prob=0.9,
              col_sep='\t'):
    # build w2v
    if not os.path.exists(sentence_w2v_path):
        build(train_seg_path,
              test_seg_path,
              out_path=sentence_w2v_path,
              sentence_path=sentence_path,
              w2v_bin_path=sentence_w2v_bin_path,
              min_count=min_count,
              col_sep=col_sep)

    # 1.build vocab for train data
    word_vocab, pos_vocab, label_vocab = build_vocab(train_seg_path, word_vocab_path,
                                                     pos_vocab_path, label_vocab_path,
                                                     min_count=min_count,
                                                     col_sep=col_sep)
    # 2.embedding
    word_emb = build_word_embedding(w2v_path, overwrite=True, sentence_w2v_path=sentence_w2v_path,
                                    word_vocab_path=word_vocab_path, word_vocab_start=word_vocab_start,
                                    w2v_dim=w2v_dim)
    pos_emb = build_pos_embedding(p2v_path, overwrite=True, pos_vocab_path=pos_vocab_path,
                                  pos_vocab_start=pos_vocab_start, pos_dim=pos_dim)
    # 3.data reader
    words, pos, labels = train_reader(train_seg_path, word_vocab, pos_vocab, label_vocab, col_sep=col_sep)
    word_test, pos_test = test_reader(test_seg_path, word_vocab, pos_vocab, label_vocab, col_sep=col_sep)
    labels_test = None

    # clear
    clear_directory(model_save_temp_dir)

    # division of training, development, and test set
    word_train, word_dev, pos_train, pos_dev, label_train, label_dev = train_test_split(
        words, pos, labels, test_size=0.1, random_state=0)

    # init model
    model = Model(max_len, word_emb, pos_emb, label_vocab=label_vocab)
    # fit model
    model.fit(word_train, pos_train, label_train,
              word_dev, pos_dev, label_dev,
              word_test, pos_test, labels_test,
              batch_size, nb_epoch, keep_prob,
              word_keep_prob, pos_keep_prob, model_save_temp_dir)

    # chose best model
    [p_test, r_test, f_test], nb_epoch = model.get_best_score()
    print('P@test:%f, R@test:%f, F@test:%f, num_best_epoch:%d' % (p_test, r_test, f_test, nb_epoch + 1))
    # save best pred label
    cmd = 'cp %s/epoch_%d.csv %s/best.csv' % (model_save_temp_dir, nb_epoch + 1, output_dir)
    print(cmd)
    os.popen(cmd)
    # clear model
    model.clear_model()


def train_xgboost_lr(data_path,
                     vectorizer_path=None, xgblr_xgb_model_path=None, xgblr_lr_model_path=None,
                     feature_encoder_path=None, feature_type='tfidf_char', col_sep='\t'):
    data_content, data_lbl = data_reader(data_path, col_sep)
    # init feature
    feature = Feature(data=data_content, feature_type=feature_type, feature_vec_path=vectorizer_path)
    # get data feature
    data_feature = feature.get_feature()
    # label
    data_label = [int(i) for i in data_lbl]
    X_train, X_val, y_train, y_val = train_test_split(
        data_feature, data_label, test_size=0.1, random_state=0)
    model = XGBLR(xgblr_xgb_model_path, xgblr_lr_model_path, feature_encoder_path)
    # fit
    model.train_model(X_train, y_train)
    # evaluate
    label_pred = model.predict(X_val)
    simple_evaluate(y_val, label_pred)


if __name__ == '__main__':
    if config.model_type == 'cnn':
        train_cnn(config.train_seg_path, config.test_seg_path, config.word_vocab_path,
                  config.pos_vocab_path, config.label_vocab_path, config.sentence_w2v_path,
                  config.sentence_w2v_bin_path,
                  sentence_path=config.sentence_path,
                  w2v_path=config.w2v_path,
                  p2v_path=config.p2v_path,
                  max_len=config.max_len,
                  min_count=config.min_count,
                  model_save_temp_dir=config.model_save_temp_dir,
                  output_dir=config.output_dir,
                  batch_size=config.batch_size,
                  nb_epoch=config.nb_epoch,
                  col_sep=config.col_sep)
    elif config.model_type == 'xgboost_lr':
        train_xgboost_lr(config.train_seg_path,
                         config.vectorizer_path,
                         config.xgblr_xgb_model_path,
                         config.xgblr_lr_model_path,
                         config.feature_encoder_path,
                         config.feature_type,
                         config.col_sep)
    else:
        train_classic(config.model_type,
                      config.train_seg_path,
                      config.pr_figure_path,
                      config.model_save_path,
                      config.vectorizer_path,
                      config.col_sep,
                      config.pred_thresholds,
                      config.num_classes,
                      config.feature_type,
                      config.min_count,
                      config.word_vocab_path)
