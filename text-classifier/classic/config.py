# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
# data
import os

train_path = "../data/train_data/train.txt"  # training file
train_seg_path = "../data/classic/food_train_seg.txt"  # segment of train file
test_seg_path = "../data/classic/food_test_seg.txt"  # segment of test file
pr_figure_path = "../data/classic/R_P.png"  # precision recall figure
model_save_path = "../data/classic/model.pkl"  # save model path
vectorizer_path = "../data/classic/tfidf_vectorizer.pkl"
col_sep = '\t'  # separate label and content of train data

pred_save_path = "../data/classic/food_pred.txt"  # infer data result
pred_thresholds = 0.5

# one of "logistic_regression or random_forest or gbdt or bayes or decision_tree or svm or knn"
model_type = "logistic_regression"
model_save_dir = "../data/classic"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
