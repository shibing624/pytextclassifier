# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os

################## for text classification  ##################

# path of training data
train_path = "../data/corpus/training.csv"
train_seg_path = "../data/corpus/train_seg.txt"
# path of testing data, if testing file does not exist,
# testing will not be performed at the end of each training pass
test_seg_path = "../data/corpus/test_seg.txt"
# path of word dictionary, if this file does not exist,
# word dictionary will be built from training data.
sentence_path = "../data/corpus/sentence.txt"
w2v_bin_path="../data/corpus/sentence_w2v.bin"
w2v_path="../data/corpus/sentence_w2v.pkl"


num_workers = 1 # threads
use_gpu = False  # to use gpu or not

num_batches_to_log = 50
num_batches_to_save_model = 400  # number of batches to output model

# directory to save the trained model
# create a new directory if the directoy does not exist
model_save_dir = "output"



if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)