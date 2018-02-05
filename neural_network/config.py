# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os

################## for text classification  ##################

# path of training data
train_path = "../data/corpus/training.csv"
# path of train segment data, train_seg_path will be build from prepare_data.py segment
train_seg_path = "../data/corpus/train_seg.txt"
# part of train_seg_path
test_seg_path = "../data/corpus/test_seg.txt"
# path of train sentence, if this file does not exist,
# it will be built from train_seg_path data by w2v_model.py train
sentence_path = "../data/corpus/sentence.txt"
sentence_w2v_bin_path = "../data/corpus/sentence_w2v.bin"
sentence_w2v_path = "../data/corpus/sentence_w2v.pkl"

# vocab
word_vocab_path = "../data/nn/word_vocab.pkl"
word_vocab_start = 2
pos_vocab_path = "../data/nn/pos_vocab.pkl"
pos_vocab_start = 1
label_vocab_path = "../data/nn/label_vocab.pkl"

# embedding
w2v_dim = 256
w2v_path = "../data/nn/w2v.pkl"
w2v_train_path = "../data/nn/w2v_train.pkl"
t2v_path = "../data/nn/pos2v.pkl"
pos_dim = 64

# train param
max_len = 300  # max len words of sentence
num_workers = 4  # threads
use_gpu = False  # to use gpu or not

num_batches_to_log = 50
num_batches_to_save_model = 400  # number of batches to output model

# directory to save the trained model
# create a new directory if the directoy does not exist
model_save_dir = "output"

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
