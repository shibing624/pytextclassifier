# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os

# path of training data and test data
train_path = "../data/train_data/train.txt"
test_path = "../data/test_data/test.txt"

# --- segment ---
result_root = '../data/nn'
if not os.path.exists(result_root):
    os.makedirs(result_root)
# path of train segment data, train_seg_path will be build by segment
train_seg_path = result_root + "/train_seg.txt"
# test_seg_path is part of train_seg_path
test_seg_path = result_root + "/test_seg.txt"

# --- train_w2v_model ---
# path of train sentence, if this file does not exist,
# it will be built from train_seg_path data by train_w2v_model.py train
sentence_path = result_root + "/sentence.txt"
# word2vec bin path
sentence_w2v_bin_path = result_root + "/sentence_w2v.bin"
# word_dict saved path
sentence_w2v_path = result_root + "/sentence_w2v.pkl"
# separate labels and text
col_sep = ','

# --- train ---
word_vocab_path = result_root + "/word_vocab.pkl"
pos_vocab_path = result_root + "/pos_vocab.pkl"
label_vocab_path = result_root + "/label_vocab.pkl"
word_vocab_start = 2
pos_vocab_start = 1

# embedding
w2v_path = result_root + "/w2v.pkl"
p2v_path = result_root + "/p2v.pkl"  # pos vector path
w2v_dim = 256
pos_dim = 64

# param
max_len = 300  # max len words of sentence
min_count = 5  # word will not be added to dictionary if it's frequency is less than min_count
batch_size = 128
nb_epoch = 10
keep_prob = 0.5
word_keep_prob = 0.9
pos_keep_prob = 0.9

# directory to save the trained model
# create a new directory if the dir does not exist
model_save_dir = "../data/nn/output_model"
model_save_temp_dir = "../data/nn/temp_output_model"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(model_save_temp_dir):
    os.mkdir(model_save_temp_dir)

best_result_path = model_save_dir + "/best_result.csv"
