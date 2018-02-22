# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import time

import numpy as np

import config
from utils.data_util import load_pkl
from utils.data_util import map_item2id
from utils.io_util import read_lines


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
    word_literal, pos_literal = [], []
    for item in word_pos_vocab:
        r_index = item.rindex(pos_sep)
        w, p = item[:r_index], item[r_index + 1:]
        if w == '' or p == '':
            continue
        word_literal.append(w)
        pos_literal.append(p)
    # sentence list
    sentence_arr = map_item2id(word_literal, word_vocab, config.max_len, lower=True)
    # pos list
    pos_arr = map_item2id(pos_literal, pos_vocab, config.max_len, lower=False)
    return sentence_arr, pos_arr, len(word_literal)


def load_emb(w2v_train_path, t2v_path):
    """
    加载词向量、词性向量
    :param w2v_train_path:
    :param t2v_path:
    :return:
    """
    return load_pkl(w2v_train_path), load_pkl(t2v_path)


def load_vocab(word_vocab_path, pos_vocab_path, label_vocab_path):
    """
    load vocab dict
    :param word_vocab_path:
    :param pos_vocab_path:
    :param label_vocab_path:
    :return:
    """
    return load_pkl(word_vocab_path), load_pkl(pos_vocab_path), load_pkl(label_vocab_path)


def init_data(lines, word_vocab, pos_vocab, label_vocab, word_sep=' ', pos_sep='/'):
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
    sentences = np.zeros((data_count, config.max_len), dtype='int32')
    pos = np.zeros((data_count, config.max_len), dtype='int32')
    sentence_actual_lengths = np.zeros((data_count), dtype='int32')
    labels = np.zeros((data_count), dtype='int32')
    instance_index = 0
    # set data
    for i in range(data_count):
        index = lines[i].index(',')
        label = lines[i][:index]
        if pos_sep in label:
            label = label.split(pos_sep)[0]
        sentence = lines[i][index + 1:]
        word_pos_vocab = sentence.split(word_sep)
        sentence_arr, pos_arr, actual_len = get_sentence_arr(word_pos_vocab, word_vocab, pos_vocab)

        sentences[instance_index, :] = sentence_arr
        pos[instance_index, :] = pos_arr
        sentence_actual_lengths[instance_index] = actual_len
        labels[instance_index] = label_vocab[label] if label in label_vocab else 0
        instance_index += 1
    return sentences, pos, labels


def load_train_data(word_vocab, pos_vocab, label_vocab):
    """
    load train data
    :param word_vocab:
    :param pos_vocab:
    :param label_vocab:
    :return:
    """
    return init_data(read_lines(config.train_seg_path),
                     word_vocab, pos_vocab, label_vocab)


def load_test_data(word_vocab, pos_vocab, label_vocab):
    """
    load test data
    :param word_vocab:
    :param pos_vocab:
    :param label_vocab:
    :return:
    """
    sentences, pos, _ = init_data(read_lines(config.test_seg_path),
                                  word_vocab, pos_vocab, label_vocab)
    return sentences, pos


if __name__ == "__main__":
    start_time = time.time()
    word_emb, pos_emb = load_emb(config.w2v_train_path, config.p2v_path)
    word_vocab, pos_vocab, label_vocab = load_vocab(config.word_vocab_path,
                                                    config.pos_vocab_path,
                                                    config.label_vocab_path)
    sentences, pos, labels = load_train_data(word_vocab, pos_vocab, label_vocab)
    print(sentences.shape)
    print(pos.shape)
    print(labels.shape)
    print(word_emb.shape)
    print(pos_emb.shape)
    print("spend time %ds." % (time.time() - start_time))
