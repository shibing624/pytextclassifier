# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

train_path = os.path.join(pwd_path, "data/train.txt")
test_path = os.path.join(pwd_path, "data/test.txt")
# run preprocess.py to segment train and test data
train_seg_path = os.path.join(pwd_path, "data/train_seg_sample.txt")  # segment of train file
test_seg_path = os.path.join(pwd_path, "data/test_seg_sample.txt")  # segment of test file

col_sep = '\t'  # separate label and content of train data

# one of "logistic_regression, random_forest, bayes, decision_tree, svm, knn, xgboost, xgboost_lr,
# mlp, ensemble, stack, fasttext, cnn, rnn, han"
model_type = "cnn"

# feature type
# classic text classification usage:  one of "tfidf_char, tfidf_word, tf_word",
# deep text classification usage: cnn/rnn/fasttext is "vectorize"
feature_type = 'vectorize'

debug = False
# default params
sentence_symbol_path = os.path.join(pwd_path, 'data/sentence_symbol.txt')
stop_words_path = os.path.join(pwd_path, 'data/stop_words.txt')

output_dir = os.path.join(pwd_path, "../output")  # output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
word_vocab_path = os.path.join(output_dir, "vocab_{}_{}.txt".format(feature_type, model_type))  # vocab path
label_vocab_path = os.path.join(output_dir, "label_{}_{}.txt".format(feature_type, model_type))  # label path
pr_figure_path = os.path.join(output_dir, "R_P_{}_{}.png".format(feature_type, model_type))  # precision recall figure
feature_vec_path = os.path.join(output_dir, "feature_{}.pkl".format(feature_type))  # vector path
model_save_path = os.path.join(output_dir, "model_{}_{}.pkl".format(feature_type, model_type))  # save model path
lr_feature_weight_path = os.path.join(output_dir, "lr_feature_weight.txt")
# predict
pred_save_path = os.path.join(output_dir, "pred_result_{}_{}.txt".format(feature_type, model_type))

# --- deep model for train ---
max_len = 300  # max len words of sentence
min_count = 1  # word will not be added to dictionary if it's frequency is less than min_count
batch_size = 64
nb_epoch = 10
embedding_dim = 128
hidden_dim = 128
dropout = 0.5
