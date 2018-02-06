# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import numpy as np
import config
from utils.io_util import read_lines
from utils.data_util import map_item2id
import pickle


def get_sentence_arr(word_pos_vocab, word_vocab, pos_vocab, pos_sep='/'):
    """
    获取词序列
    :param word_pos_vocab: list，句子和词性
    :param word_vocab: 词表
    :param pos_vocab: 词性表
    :param pos_sep: 词性表
    :return: sentence_arr, np.array, 字符id序列
             pos_arr, np.array, 词性标记序列
    """
    word_iteral, pos_iteral = [], []
    for item in word_pos_vocab:
        rindex = item.rindex(pos_sep)
        word_iteral.append(item[:rindex])
        pos_iteral.append(item[rindex + 1:])
    # sentence list
    sentence_arr = map_item2id(word_iteral, word_vocab, config.max_len, lower=True)
    # pos list
    pos_arr = map_item2id(pos_iteral, pos_vocab, config.max_len, lower=False)
    return sentence_arr, pos_arr, len(word_iteral)


def load_emb(w2v_train_path, t2v_path):
    """
    加载词向量、词性向量
    :param w2v_train_path:
    :param t2v_path:
    :return:
    """
    with open(w2v_train_path, 'rb') as f:
        word_vec = pickle.load(f)
    with open(t2v_path, 'rb') as f:
        pos_vec = pickle.load(f)
    return word_vec, pos_vec


def load_pkl_dict(pkl_dict_path):
    """
    加载词典
    :param pkl_dict_path:
    :return:
    """
    with open(pkl_dict_path, 'rb') as f:
        result = pickle.load(f)
    return result


def init_data(lines, word_vocab, pos_vocab, lable_vocab):
    """
    load data
    :param lines:
    :param word_vocab:
    :param pos_vocab:
    :param lable_vocab:
    :return:
    """
    data_count = len(lines)
    sentences = np.zeros((data_count, config.max_len), dtype='int32')
    pos = np.zeros((data_count, config.max_len), dtype='int32')

