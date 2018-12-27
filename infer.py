# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time

import config
from models.cnn_model import Model
from models.reader import data_reader
from models.reader import test_reader
from utils.data_utils import load_vocab, load_pkl
from utils.tensor_utils import get_ckpt_path
from models.feature import Feature
from models.xgboost_lr_model import XGBLR
from sklearn.metrics import classification_report

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


def infer_classic(model_save_path, test_data_path, thresholds=0.5,
                  pred_save_path=None, vectorizer_path=None, col_sep=',',
                  num_classes=2, feature_type='tf'):
    # load model
    model = load_pkl(model_save_path)
    # load data content
    data_set, test_ids = data_reader(test_data_path, col_sep)
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
    if test_ids:
        # evaluate
        test_ids = [int(i) for i in test_ids]
        print(classification_report(test_ids, label_pred))
    print("finish prediction.")


def infer_cnn(data_path, model_path,
              word_vocab_path, pos_vocab_path, label_vocab_path,
              word_emb_path, pos_emb_path, batch_size, pred_save_path=None):
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
        model.saver.restore(model.sess, ckpt_path)
    else:
        print("Can't find the checkpoint.going to stop")
        return
    label_pred = model.predict(word_test, pos_test, batch_size)
    save(label_pred, pred_save_path=pred_save_path)
    print("finish prediction.")


def infer_xgboost_lr(test_data_path,
                     vectorizer_path=None, xgblr_xgb_model_path=None, xgblr_lr_model_path=None,
                     feature_encoder_path=None, col_sep='\t', pred_save_path=None, feature_type='tfidf_char'):
    # load data content
    data_set, test_ids = data_reader(test_data_path, col_sep)
    # init feature
    feature = Feature(data_set, feature_type=feature_type, feature_vec_path=vectorizer_path, is_infer=True)
    # get data feature
    data_feature = feature.get_feature()
    # load model
    model = XGBLR(xgblr_xgb_model_path, xgblr_lr_model_path, feature_encoder_path)
    # predict
    label_pred = model.predict(data_feature)
    save(label_pred, test_ids=None, pred_save_path=pred_save_path, data_set=data_set)
    if test_ids:
        # evaluate
        test_ids = [int(i) for i in test_ids]
        print(classification_report(test_ids, label_pred))
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
                  config.batch_size,
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
                      config.pred_thresholds,
                      config.pred_save_path,
                      config.vectorizer_path,
                      config.col_sep,
                      config.num_classes,
                      config.feature_type)
    print("spend time %ds." % (time.time() - start_time))
