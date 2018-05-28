# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys
import config
from reader import data_reader

sys.path.append('..')
from utils.io_utils import load_pkl


def infer(model_save_path, test_data_path, thresholds=0.5,
          pred_save_path=None, vectorizer_path=None, col_sep=',',
          num_classes=2,model_type='svm'):
    # load model
    model = load_pkl(model_save_path)
    # load data content
    data_content, _ = data_reader(test_data_path, col_sep)
    # data feature
    tfidf_vectorizer = load_pkl(vectorizer_path)
    data_feature = tfidf_vectorizer.transform(data_content)
    if num_classes == 2 and model_type != 'svm':
        # binary classification
        label_pred_probas = model.predict_proba(data_feature)[:, 1]
        label_pred = label_pred_probas > thresholds
    else:
        label_pred = model.predict(data_feature)  # same
    save(label_pred, pred_save_path)
    print("finish prediction.")


def save(label_pred, pred_save_path=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in label_pred:
                f.write(str(i) + '\n')
        print("pred_save_path:", pred_save_path)


if __name__ == "__main__":
    infer(config.model_save_path,
          config.test_seg_path,
          config.pred_thresholds,
          config.pred_save_path,
          config.vectorizer_path,
          config.col_sep,
          config.num_classes,
          config.model_type)
