# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import pickle
import numpy as np
import config
from neural_network.utils.data_util import build_dict
from time import time
import jieba


def init_vec():
    """
    init vector
    :return:
    """
    return


def segment(in_file, out_file):
    """
    segment input file to output file
    :param in_file:
    :param out_file:
    :return:
    """
    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8')as fout:
        in_sentence = []
        for line in fin:
            in_line = line.strip()
            seg_line = ' '.join(jieba.lcut(in_line))
            fout.write(seg_line + "\n")
            in_sentence.append(line.strip())
    print("input file size:", len(in_sentence))


if __name__ == '__main__':
    start_time = time()
    segment(config.train_path, config.train_seg_path)
    end_time = time()
    print("spend time:", end_time - start_time)
