# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
from collections import defaultdict
import numpy as np
import pickle


def build_dict(items, start=0, sort=True,
               min_count=None, lower=False):
    """
    构建字典
    :param items: list  [item1, item2, ... ]
    :param start: index 默认0
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: dict
    """
    result = dict()
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            item = item if not lower else item.lower
            dic[item] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            index = i + start
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result[key] = index
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            index = i + start
            result[item] = index
    return result


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


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def dump_pkl(vocab, pkl_path, overwrite=False):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if os.path.exists(pkl_path) and not overwrite:
        return
    with open(pkl_path, 'wb') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
