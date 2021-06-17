# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import pickle
from codecs import open
from collections import defaultdict

import numpy as np

from pytextclassifier.utils.log import logger


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            item = item if not lower else item.lower()
            dic[item] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    return result


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], line.strip().split("\t")[-1])
                for idx, line in enumerate(open(dict_path, "r", encoding='utf-8').readlines()))


def save_dict(dict_data, dict_path):
    with open(dict_path, 'w', encoding='utf-8') as f:
        count = 0
        for k, v in dict_data.items():
            v_str = map(str, v)
            f.write(k + '\t' + ' '.join(v_str) + '\n')
            count += 1
        logger.info('save dict size:{}, dict_path: {}'.format(count, dict_path))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r", encoding='utf-8').readlines()))


def flatten_list(nest_list):
    """
    嵌套列表压扁成一个列表
    :param nest_list: 嵌套列表
    :return: list
    """
    result = []
    for item in nest_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def map_item2id(items, vocab, max_len, non_word=0, lower=False):
    """
    将word/pos等映射为id
    :param items: list，待映射列表
    :param vocab: 词表
    :param max_len: int，序列最大长度
    :param non_word: 未登录词标号，默认0
    :param lower: bool，小写
    :return: np.array, dtype=int32,shape=[max_len,]
    """
    assert type(non_word) == int
    arr = np.zeros((max_len,), dtype='int32')
    # 截断max_len长度的items
    min_range = min(max_len, len(items))
    for i in range(min_range):
        item = items[i] if not lower else items[i].lower()
        arr[i] = vocab[item] if item in vocab else non_word
    return arr


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write(word + '\n')
            else:
                f.write(word)
    logger.info("- write to {} done. {} tokens".format(filename, len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise IOError(filename)
    return d


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def save_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            # pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(vocab, f, protocol=2)  # 兼容python2和python3
        logger.info("save %s ok." % pkl_path)


def get_word_segment_data(contents, word_sep=' ', pos_sep='/'):
    data = []
    for content in contents:
        temp = []
        for word in content.split(word_sep):
            if pos_sep in word:
                temp.append(word.split(pos_sep)[0])
            else:
                temp.append(word.strip())
        data.append(word_sep.join(temp))
    return data


def get_char_segment_data(contents, word_sep=' ', pos_sep='/'):
    data = []
    for content in contents:
        temp = ''
        for word in content.split(word_sep):
            if pos_sep in word:
                temp += word.split(pos_sep)[0]
            else:
                temp += word.strip()
        # char seg with list
        data.append(word_sep.join(list(temp)))
    return data


def load_list(path):
    return [word for word in open(path, 'r', encoding='utf-8').read().split()]


def save_predict_result(pred_labels, ture_labels=None, pred_save_path=None, data_set=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(pred_labels)):
                if ture_labels and len(ture_labels) > 0:
                    assert len(ture_labels) == len(pred_labels)
                    if data_set:
                        f.write(ture_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(ture_labels[i] + '\n')
                else:
                    if data_set:
                        f.write(pred_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(pred_labels[i] + '\n')
        logger.debug("pred_save_path:", pred_save_path)
