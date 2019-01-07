# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time

import tensorflow as tf
from sklearn.metrics import classification_report

import config
from models.cnn_model import Model
from models.feature import Feature
from models.reader import data_reader
from models.reader import test_reader
from models.xgboost_lr_model import XGBLR
from utils.data_utils import load_vocab, load_pkl
from utils.tensor_utils import get_ckpt_path

label_revserv_dict = {v: k for k, v in config.label_dict.items()}


def save(label_pred, test_ids=None, pred_save_path=None, data_set=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(label_pred)):
                if test_ids and len(test_ids) > 0:
                    assert len(test_ids) == len(label_pred)
                    if data_set:
                        f.write(str(test_ids[i]) + '\t' + label_revserv_dict[label_pred[i]] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(str(test_ids[i]) + '\t' + label_revserv_dict[label_pred[i]] + '\n')
                else:
                    if data_set:
                        f.write(str(label_pred[i]) + '\t' + label_revserv_dict[label_pred[i]] + '\t' + \
                                data_set[i] + '\n')
                    else:
                        f.write(str(label_pred[i]) + '\t' + label_revserv_dict[label_pred[i]] + '\n')
        print("pred_save_path:", pred_save_path)


def infer_classic(model_save_path,
                  test_data_path,
                  thresholds=0.5,
                  pred_save_path='',
                  vectorizer_path='',
                  col_sep='\t',
                  num_classes=2,
                  feature_type='tfidf_word'):
    # load model
    model = load_pkl(model_save_path)
    # load data content
    data_set, data_label = data_reader(test_data_path, col_sep)
    # init feature
    feature = Feature(data_set, feature_type=feature_type,
                      feature_vec_path=vectorizer_path, is_infer=True)
    # get data feature
    data_feature = feature.get_feature()
    if num_classes == 2:
        # binary classification
        label_pred_probas = model.predict_proba(data_feature)[:, 1]
        label_pred = label_pred_probas > thresholds
    else:
        label_pred = model.predict(data_feature)
    save(label_pred, test_ids=None, pred_save_path=pred_save_path, data_set=data_set)
    if data_label:
        # evaluate
        data_label = [int(i) for i in data_label]
        print(classification_report(data_label, label_pred))
    print("finish prediction.")
    if 'logistic_regression' in model_save_path and config.is_debug:
        count = 0
        features = load_pkl('output/features.pkl')
        for line in data_set:
            if count > 5:
                break
            count += 1
            print(line)
            words = line.split()
            for category, category_feature in features.items():
                print('*' * 43)
                print(category)
                category_score = 0
                for w in words:
                    if w in category_feature:
                        category_score += category_feature[w]
                        print(w, category_feature[w])
                print(category, category_score)
                print('=' * 43)
                print()


def infer_cnn(data_path, model_path,
              word_vocab_path, pos_vocab_path, label_vocab_path,
              word_emb_path, pos_emb_path, pred_save_path=None):
    # init dict
    word_vocab, pos_vocab, label_vocab = load_vocab(word_vocab_path), load_vocab(pos_vocab_path), load_vocab(
        label_vocab_path)
    word_emb, pos_emb = load_pkl(word_emb_path), load_pkl(pos_emb_path)
    word_test, pos_test = test_reader(data_path, word_vocab, pos_vocab, label_vocab)
    # init model
    model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
    ckpt_path = get_ckpt_path(model_path)
    if ckpt_path:
        print("Read model parameters from %s" % ckpt_path)
        # init model
        model.sess.run(tf.global_variables_initializer())
        model.saver.restore(model.sess, ckpt_path)
    else:
        print("Can't find the checkpoint.going to stop")
        return
    label_pred = model.predict(word_test, pos_test)
    data_set, data_label = data_reader(data_path)
    save(label_pred, test_ids=None, pred_save_path=pred_save_path, data_set=data_set)

    if data_label:
        # evaluate
        data_label = [int(i) for i in data_label]
        print(classification_report(data_label, label_pred))
    print("finish prediction.")


def infer_xgboost_lr(test_data_path,
                     vectorizer_path=None, xgblr_xgb_model_path=None, xgblr_lr_model_path=None,
                     feature_encoder_path=None, col_sep='\t', pred_save_path=None, feature_type='tfidf_char'):
    # load data content
    data_set, data_label = data_reader(test_data_path, col_sep)
    # init feature
    feature = Feature(data_set, feature_type=feature_type, feature_vec_path=vectorizer_path, is_infer=True)
    # get data feature
    data_feature = feature.get_feature()
    # load model
    model = XGBLR(xgblr_xgb_model_path, xgblr_lr_model_path, feature_encoder_path)
    # predict
    label_pred = model.predict(data_feature)
    save(label_pred, test_ids=None, pred_save_path=pred_save_path, data_set=data_set)
    if data_label:
        # evaluate
        data_label = [int(i) for i in data_label]
        print(classification_report(data_label, label_pred))
    print("finish prediction.")


if __name__ == "__main__":
    start_time = time.time()
    if config.model_type == 'cnn':
        infer_cnn(config.test_seg_path,
                  config.model_save_temp_dir,
                  config.word_vocab_path,
                  config.pos_vocab_path,
                  config.label_vocab_path,
                  config.w2v_path,
                  config.p2v_path,
                  config.pred_save_path)
    elif config.model_type == 'xgboost_lr':
        infer_xgboost_lr(config.test_seg_path,
                         config.vectorizer_path,
                         config.xgblr_xgb_model_path,
                         config.xgblr_lr_model_path,
                         config.feature_encoder_path,
                         config.col_sep,
                         pred_save_path=config.pred_save_path,
                         feature_type=config.feature_type)
    else:
        infer_classic(config.model_save_path,
                      config.test_seg_path,
                      thresholds=config.pred_thresholds,
                      pred_save_path=config.pred_save_path,
                      vectorizer_path=config.vectorizer_path,
                      col_sep=config.col_sep,
                      num_classes=config.num_classes,
                      feature_type=config.feature_type)
    print("spend time %ds." % (time.time() - start_time))
