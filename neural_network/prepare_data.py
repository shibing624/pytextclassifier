# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
from time import time

import numpy as np

import config
from neural_network.utils.data_util import build_dict
from neural_network.utils.data_util import dump_pkl
from neural_network.utils.data_util import load_pkl
from neural_network.utils.io_util import read_lines


def load_vocab(train_path, test_path, word_sep=' ', pos_sep='/'):
    lines = read_lines(train_path)
    lines += read_lines(test_path)
    word_lst = []
    pos_lst = []
    label_lst = []
    for line in lines:
        index = line.index(',')
        label = line[:index + 1]
        if pos_sep in label:
            label = label.split(pos_sep)[0]
        label_lst.extend(label)
        sentence = line[index + 1:]
        # word and pos
        word_pos_list = sentence.split(word_sep)
        word, pos = [], []
        for item in word_pos_list:
            r_index = item.rindex(pos_sep)
            w, p = item[:r_index], item[r_index + 1:]
            if w == '' or p == '':
                continue
            word.append(w)
            pos.append(p)
        word_lst.extend(word)
        pos_lst.extend(pos)
    return word_lst, pos_lst, label_lst


def build_vocab(word_list, pos_list, label_list, word_vocab_path, pos_vocab_path, label_vocab_path):
    # word vocab
    word_dict = build_dict(word_list, start=config.word_vocab_start,
                           min_count=config.min_count, sort=True, lower=True)
    # save
    dump_pkl(word_dict, word_vocab_path, overwrite=True)
    print("save %s ok." % word_vocab_path)
    # pos vocab
    pos_dict = build_dict(pos_list, start=config.pos_vocab_start,
                          sort=True, lower=False)
    # save
    dump_pkl(pos_dict, pos_vocab_path, overwrite=True)
    print("save %s ok." % pos_vocab_path)
    # label vocab
    # label_types = [str(i) for i in range(1, 12)]
    label_types = [str(i) for i in label_list]
    label_dict = build_dict(label_types)
    # save
    dump_pkl(label_dict, label_vocab_path, overwrite=True)
    print("save %s ok." % label_vocab_path)


def build_word_embedding(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        print("already has $s and use it." % path)
        return
    w2v_dict_full = load_pkl(config.w2v_path)
    word_vocab = load_pkl(config.word_vocab_path)
    word_vocab_count = len(w2v_dict_full) + config.word_vocab_start
    word_emb = np.zeros((word_vocab_count, config.w2v_dim), dtype='float32')
    for word in word_vocab:
        index = word_vocab[word]
        if word in w2v_dict_full:
            word_emb[index, :] = w2v_dict_full[word]
        else:
            random_vec = np.random.uniform(-0.25, 0.25, size=(config.w2v_dim,)).astype('float32')
            word_emb[index, :] = random_vec
    # write to pkl
    dump_pkl(word_emb, path)
    print("save %s ok." % path)


def build_pos_embedding(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        return
    pos_vocab = load_pkl(config.pos_vocab_path)
    pos_vocab_count = len(pos_vocab) + config.pos_vocab_start
    pos_emb = np.random.normal(size=(pos_vocab_count, config.pos_dim,)).astype('float32')
    for i in range(config.pos_vocab_start):
        pos_emb[i, :] = 0.
    # write to pkl
    dump_pkl(pos_emb, path)
    print("save %s ok." % path)


if __name__ == '__main__':
    start_time = time()
    # 1生成训练数据词典
    print("build train vocab...")
    word_list, pos_list, label_list = load_vocab(config.train_seg_path, config.test_seg_path)
    build_vocab(word_list, pos_list, label_list, config.word_vocab_path,
                config.pos_vocab_path, config.label_vocab_path)
    # 2生成词向量
    print("build embedding...")
    build_word_embedding(config.w2v_train_path, overwrite=True)
    build_pos_embedding(config.p2v_path, overwrite=True)
    end_time = time()
    print("spend time:", end_time - start_time)