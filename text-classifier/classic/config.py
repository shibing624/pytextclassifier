# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
# data
import os

train_path = "../data/train_data/train.txt"  # 输入的文件
train_seg_path = "../data/classic/train_seg.txt"  # 输入的文件
test_seg_path = "../data/classic/test_seg.txt"  # 输入的文件
feature_space_path = "../data/classic/tfidf.dat"  # 输出的文件
pr_figure_path = "../data/classic/R_P.png"  # 保存P_R曲线图
pred_save_path = "../data/classic/pred.txt"


# one of "logistic_regression or random_forest or gbdt or bayes or decision_tree or svm or knn"
model_type = "logistic_regression"
model_save_dir = "../data/classic"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
