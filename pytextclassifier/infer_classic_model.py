# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import time

from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.append('..')
from pytextclassifier import config
from pytextclassifier.utils.evaluate import cal_multiclass_lr_predict
from pytextclassifier.feature import Feature
from pytextclassifier.models.xgboost_lr_model import XGBLR
from pytextclassifier.utils.data_utils import load_pkl, load_vocab, save_predict_result, load_dict, data_reader
from pytextclassifier.utils.log import logger


def infer_classic(model_type='xgboost_lr',
                  model_save_path='',
                  label_vocab_path='',
                  test_data_path='',
                  pred_save_path='',
                  feature_vec_path='',
                  col_sep='\t',
                  feature_type='tfidf_word'):
    # load data content
    data_set, true_labels = data_reader(test_data_path, col_sep)
    # init feature
    feature = Feature(data=data_set, feature_type=feature_type,
                      feature_vec_path=feature_vec_path, is_infer=True)
    # get data feature
    data_feature = feature.get_feature()
    # load model
    if model_type == 'xgboost_lr':
        model = XGBLR(model_save_path)
    else:
        model = load_pkl(model_save_path)

    # predict
    pred_label_probs = model.predict_proba(data_feature)

    # label id map
    label_id = load_vocab(label_vocab_path)
    id_label = {v: k for k, v in label_id.items()}

    pred_labels = [id_label[prob.argmax()] for prob in pred_label_probs]
    pred_output = [id_label[prob.argmax()] + col_sep + str(prob.max()) for prob in pred_label_probs]
    logger.info("save infer label and prob result to:%s" % pred_save_path)
    save_predict_result(pred_output, ture_labels=None, pred_save_path=pred_save_path, data_set=data_set)

    # evaluate
    if true_labels:
        try:
            print(classification_report(true_labels, pred_labels))
            print(confusion_matrix(true_labels, pred_labels))
        except UnicodeEncodeError:
            true_labels_id = [label_id[i] for i in true_labels]
            pred_labels_id = [label_id[i] for i in pred_labels]
            print(classification_report(true_labels_id, pred_labels_id))
            print(confusion_matrix(true_labels_id, pred_labels_id))
        except Exception:
            print("error. no true labels")

    # analysis lr model
    if config.debug and model_type == "logistic_regression":
        feature_weight_dict = load_dict(config.lr_feature_weight_path)
        pred_labels = cal_multiclass_lr_predict(data_set, feature_weight_dict, id_label)
        print(pred_labels[:5])


if __name__ == "__main__":
    start_time = time.time()
    if config.model_type in ['logistic_regression', 'random_forest', 'bayes', 'decision_tree', 'svm', 'knn', 'xgboost',
                             'xgboost_lr']:
        infer_classic(model_type=config.model_type,
                      model_save_path=config.model_save_path,
                      label_vocab_path=config.label_vocab_path,
                      test_data_path=config.test_seg_path,
                      pred_save_path=config.pred_save_path,
                      feature_vec_path=config.feature_vec_path,
                      col_sep=config.col_sep,
                      feature_type=config.feature_type)
    logger.info("spend time %ds." % (time.time() - start_time))
    logger.info("finish predict.")
