# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time

from keras.models import load_model
from sklearn.metrics import classification_report

import config
from models.feature import Feature
from models.reader import data_reader
from models.xgboost_lr_model import XGBLR
from utils.data_utils import load_pkl, load_vocab


def save(pred_labels, ture_labels=None, pred_save_path=None, data_set=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(pred_labels)):
                if ture_labels and len(ture_labels) > 0:
                    assert len(ture_labels) == len(pred_labels)
                    if data_set:
                        f.write(
                            str(ture_labels[i]) + '\t' + data_set[i] + '\n')
                    else:
                        f.write(str(ture_labels[i]) + '\n')
                else:
                    if data_set:
                        f.write(str(pred_labels[i]) + '\t' + data_set[i] + '\n')
                    else:
                        f.write(str(pred_labels[i]) + '\n')
        print("pred_save_path:", pred_save_path)


def infer_classic(model_save_path,
                  test_data_path,
                  pred_save_path='',
                  vectorizer_path='',
                  col_sep='\t',
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
    label_pred = model.predict(data_feature)
    save(label_pred, ture_labels=None, pred_save_path=pred_save_path, data_set=data_set)
    if 'logistic_regression' in model_save_path and config.is_debug:
        count = 0
        features = load_pkl('output/lr_features.pkl')
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
    save(label_pred, ture_labels=None, pred_save_path=pred_save_path, data_set=data_set)
    if data_label:
        # evaluate
        data_label = [int(i) for i in data_label]
        print(classification_report(data_label, label_pred))
    print("finish prediction.")


def infer_deep_model(model_type='cnn',
                     data_path='',
                     model_save_path='',
                     label_vocab_path='',
                     max_len=300,
                     batch_size=128,
                     col_sep='\t',
                     pred_save_path=None):
    # load data content
    data_set, true_labels = data_reader(data_path, col_sep)
    # init feature
    # han model need [doc sentence dim] feature(shape 3); others is [sentence dim] feature(shape 2)
    if model_type == 'han':
        feature_type = 'doc_vectorize'
    else:
        feature_type = 'vectorize'
    feature = Feature(data_set, feature_type=feature_type, is_infer=True, max_len=max_len)
    # get data feature
    data_feature = feature.get_feature()

    # load model
    model = load_model(model_save_path)
    # predict
    pred_label_probs = model.predict(data_feature, batch_size=batch_size)
    pred_labels = [prob.argmax() for prob in pred_label_probs]
    # label id map
    label_id = load_vocab(label_vocab_path)
    id_label = {v: k for k, v in label_id.items()}
    pred_labels = [int(id_label[i]) for i in pred_labels]
    save(pred_labels, ture_labels=None, pred_save_path=pred_save_path, data_set=data_set)
    if true_labels:
        # evaluate
        true_labels = [int(i) for i in true_labels]
        assert len(pred_labels) == len(true_labels)
        for label, prob in zip(true_labels, pred_label_probs):
            print('label_true:%s\tprob_label:%s\tprob:%s' % (label, id_label[prob.argmax()], prob.max()))
        print('total eval:')
        print(classification_report(true_labels, pred_labels))
    print("finish prediction.")


if __name__ == "__main__":
    start_time = time.time()
    if config.model_type in ['fasttext', 'cnn', 'rnn', 'han']:
        infer_deep_model(model_type=config.model_type,
                         data_path=config.test_seg_path,
                         model_save_path=config.model_save_path,
                         label_vocab_path=config.label_vocab_path,
                         max_len=config.max_len,
                         batch_size=config.batch_size,
                         col_sep=config.col_sep,
                         pred_save_path=config.pred_save_path)
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
                      pred_save_path=config.pred_save_path,
                      vectorizer_path=config.vectorizer_path,
                      col_sep=config.col_sep,
                      feature_type=config.feature_type)
    print("spend time %ds." % (time.time() - start_time))
