# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
# data
import os

label_dict = {"": 0,
              "减肥": 1,
              "丰胸": 2,
              "补肾壮阳": 3,
              "女性保健": 4,
              "增高": 5,
              "祛斑美容": 6,
              "改善消化": 7,
              "生发育发": 8,
              "辅助降三高": 9,
              "增强免疫力": 10,
              "增加骨密度": 11,
              "眼科治疗": 12,
              "改善睡眠": 13,
              "护肝": 14,
              "皮肤性病治疗": 15,
              "风湿骨科治疗": 16,
              "耳鼻喉科治疗": 17,
              "狐臭口臭治疗": 18,
              "泌尿内科治疗": 19,
              }
is_pos = False
train_path = "data/bjp.train.txt"
test_path = "data/test_sample.txt"
train_seg_path = "data/bjp.train.seg.txt"  # segment of train file
test_seg_path = "data/test_seg_sample.txt"  # segment of test file

sentence_symbol_path = 'data/sentence_symbol.txt'
stop_words_path = 'data/stop_words.txt'

# one of "logistic_regression, random_forest, bayes, decision_tree, svm, knn, xgboost, xgboost_lr, mlp, ensemble, stack, cnn"
model_type = "logistic_regression"
# one of "tfidf_char, tfidf_word, language, tfidf_char_language", ignore when model_type="cnn"
feature_type = 'tfidf_char'
output_dir = "output"

pr_figure_path = output_dir + "/R_P.png"  # precision recall figure
model_save_path = output_dir + "/model_" + feature_type + "_" + model_type + ".pkl"  # save model path
vectorizer_path = output_dir + "/vectorizer_" + feature_type + ".pkl"

# xgboost_lr model
xgblr_xgb_model_path = output_dir + "/xgblr_xgb.pkl"
xgblr_lr_model_path = output_dir + "/xgblr_lr.pkl"
feature_encoder_path = output_dir + "/xgblr_encoder.pkl"

pred_save_path = output_dir + "/pred_result.txt"  # infer data result
col_sep = '\t'  # separate label and content of train data
pred_thresholds = 0.5
num_classes = len(label_dict)  # num of data label classes

# --- build_w2v.py ---
# path of train sentence, if this file does not exist,
# it will be built from train_seg_path data by train_w2v_model.py train
# word2vec bin path
sentence_w2v_bin_path = output_dir + "/sentence_w2v.bin"
# sentence w2v vocab saved path
sentence_w2v_path = output_dir + "/sentence_w2v.pkl"
sentence_path = output_dir + '/sentences.txt'

# --- train ---
word_vocab_path = output_dir + "/word_vocab.txt"
pos_vocab_path = output_dir + "/pos_vocab.txt"
label_vocab_path = output_dir + "/label_vocab.txt"
word_vocab_start = 2
pos_vocab_start = 1

# embedding
w2v_path = output_dir + "/w2v.pkl"
p2v_path = output_dir + "/p2v.pkl"  # pos vector path
w2v_dim = 256
pos_dim = 64

# param
max_len = 400  # max len words of sentence
min_count = 10  # word will not be added to dictionary if it's frequency is less than min_count
batch_size = 128
nb_epoch = 5
keep_prob = 0.5
word_keep_prob = 0.9
pos_keep_prob = 0.9

# directory to save the trained model
# create a new directory if the dir does not exist
model_save_temp_dir = output_dir + "/save_model"
best_result_path = output_dir + "/best_result.csv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_save_temp_dir):
    os.mkdir(model_save_temp_dir)
