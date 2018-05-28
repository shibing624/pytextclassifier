# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import time

import numpy as np

import config
import sys
sys.path.append('..')
from utils.data_utils import build_dict
from utils.io_utils import dump_pkl
from utils.io_utils import load_pkl
from utils.data_utils import map_item2id
from utils.data_utils import read_lines


def _load_data(path, col_sep=',', word_sep=' ', pos_sep='/'):
    lines = read_lines(path)
    word_lst = []
    pos_lst = []
    label_lst = []
    for line in lines:
        index = line.index(col_sep)
        label = line[:index]
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


def build_vocab(train_path, word_vocab_path, pos_vocab_path, label_vocab_path):
    word_lst, pos_lst, label_lst = _load_data(train_path, col_sep=config.col_sep)
    # word vocab
    word_vocab = build_dict(word_lst, start=config.word_vocab_start,
                            min_count=config.min_count, sort=True, lower=True)
    # save
    dump_pkl(word_vocab, word_vocab_path, overwrite=True)
    # pos vocab
    pos_vocab = build_dict(pos_lst, start=config.pos_vocab_start, sort=True, lower=False)
    # save
    dump_pkl(pos_vocab, pos_vocab_path, overwrite=True)
    # label vocab
    label_types = [str(i) for i in label_lst]
    label_vocab = build_dict(label_types)
    # save
    dump_pkl(label_vocab, label_vocab_path, overwrite=True)


def load_vocab(word_vocab_path, pos_vocab_path, label_vocab_path):
    """
    load vocab dict
    :param word_vocab_path:
    :param pos_vocab_path:
    :param label_vocab_path:
    :return:
    """
    return load_pkl(word_vocab_path), load_pkl(pos_vocab_path), load_pkl(label_vocab_path)


def build_word_embedding(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        print("already has $s and use it." % path)
        return
    w2v_dict_full = load_pkl(config.sentence_w2v_path)
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
    # save
    dump_pkl(word_emb, path)


def build_pos_embedding(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        return
    pos_vocab = load_pkl(config.pos_vocab_path)
    pos_vocab_count = len(pos_vocab) + config.pos_vocab_start
    pos_emb = np.random.normal(size=(pos_vocab_count, config.pos_dim,)).astype('float32')
    for i in range(config.pos_vocab_start):
        pos_emb[i, :] = 0.
    # save
    dump_pkl(pos_emb, path)


def load_emb(w2v_path, p2v_path):
    """
    加载词向量、词性向量
    :param w2v_path:
    :param p2v_path:
    :return:
    """
    return load_pkl(w2v_path), load_pkl(p2v_path)


def _get_word_arr(word_pos_vocab, word_vocab, pos_vocab, pos_sep='/'):
    """
    获取词序列
    :param word_pos_vocab: list，句子和词性
    :param word_vocab: 词表
    :param pos_vocab: 词性表
    :param pos_sep: 词性表
    :return: word_arr, np.array, 字符id序列
             pos_arr, np.array, 词性标记序列
    """
    word_literal, pos_literal = [], []
    for item in word_pos_vocab:
        r_index = item.rindex(pos_sep)
        w, p = item[:r_index], item[r_index + 1:]
        if w == '' or p == '':
            continue
        word_literal.append(w)
        pos_literal.append(p)
    # word list
    word_arr = map_item2id(word_literal, word_vocab, config.max_len, lower=True)
    # pos list
    pos_arr = map_item2id(pos_literal, pos_vocab, config.max_len, lower=False)
    return word_arr, pos_arr, len(word_literal)


def _init_data(lines, word_vocab, pos_vocab, label_vocab, col_sep=',', word_sep=' ', pos_sep='/'):
    """
    load data
    :param lines:
    :param word_vocab:
    :param pos_vocab:
    :param lable_vocab:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    data_count = len(lines)
    # init
    words = np.zeros((data_count, config.max_len), dtype='int32')
    pos = np.zeros((data_count, config.max_len), dtype='int32')
    sentence_actual_lengths = np.zeros((data_count), dtype='int32')
    labels = np.zeros((data_count), dtype='int32')
    instance_index = 0
    # set data
    for i in range(data_count):
        index = lines[i].index(col_sep)
        label = lines[i][:index]
        if pos_sep in label:
            label = label.split(pos_sep)[0]
        sentence = lines[i][index + 1:]
        word_pos_vocab = sentence.split(word_sep)
        word_arr, pos_arr, actual_len = _get_word_arr(word_pos_vocab, word_vocab, pos_vocab)

        words[instance_index, :] = word_arr
        pos[instance_index, :] = pos_arr
        sentence_actual_lengths[instance_index] = actual_len
        labels[instance_index] = label_vocab[label] if label in label_vocab else 0
        instance_index += 1
    return words, pos, labels


def train_reader(path, word_vocab, pos_vocab, label_vocab):
    """
    load train data
    :param word_vocab:
    :param pos_vocab:
    :param label_vocab:
    :return:
    """
    return _init_data(read_lines(path),
                      word_vocab, pos_vocab, label_vocab, col_sep=config.col_sep)


def test_reader(path, word_vocab, pos_vocab, label_vocab):
    """
    load test data
    :param word_vocab:
    :param pos_vocab:
    :param label_vocab:
    :return:
    """
    sentences, pos, _ = _init_data(read_lines(path),
                                   word_vocab, pos_vocab, label_vocab, col_sep=config.col_sep)
    return sentences, pos


if __name__ == '__main__':
    start_time = time.time()
    print("build train vocab...")
    # 1.build dict for train data
    build_vocab(config.train_seg_path, config.word_vocab_path,
                config.pos_vocab_path, config.label_vocab_path)
    # 2.build embedding
    build_word_embedding(config.w2v_path, overwrite=True)
    build_pos_embedding(config.p2v_path, overwrite=True)
    # 3.load vocab
    word_vocab, pos_vocab, label_vocab = load_vocab(config.word_vocab_path,
                                                    config.pos_vocab_path,
                                                    config.label_vocab_path)
    # 4.load emb
    word_emb, pos_emb = load_emb(config.w2v_path, config.p2v_path)
    print(word_emb.shape)
    print(pos_emb.shape)

    # 5. train data reader
    words, pos, labels = train_reader(config.train_seg_path, word_vocab, pos_vocab, label_vocab)
    print(words.shape)
    print(pos.shape)
    print(labels.shape)
    print("spend time %ds." % (time.time() - start_time))
