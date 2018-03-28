# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import logging
import os
import pickle

import jieba
from jieba import posseg


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
                                  datefmt='%m%d%Y %I:%M:%sS')
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    # handle.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.INFO)
    return logger


def segment(sentence):
    """
    切词
    :param sentence:
    :return: list
    """
    jieba.default_logger.setLevel(logging.ERROR)
    return jieba.lcut(sentence)


def segment_pos(sentence):
    """
    切词
    :param sentence:
    :return: list
    """
    jieba.default_logger.setLevel(logging.ERROR)
    return posseg.lcut(sentence)


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if os.path.exists(pkl_path) and not overwrite:
        return
    with open(pkl_path, 'wb') as f:
        # pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(vocab, f, protocol=0)
