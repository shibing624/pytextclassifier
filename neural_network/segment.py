# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 切词

from time import time

import jieba.posseg

import config


def segment(in_file, out_file, word_sep=' ', pos_sep='/'):
    """
    segment input file to output file
    :param in_file:
    :param out_file:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        in_sentence = []
        for line in fin:
            in_line = line.strip()
            words = jieba.posseg.cut(in_line)
            seg_line = ''
            for word, pos in words:
                seg_line += word + pos_sep + pos + word_sep
            fout.write(seg_line + "\n")
            in_sentence.append(line.strip())
    print("segment ok. input file size:", len(in_sentence))


if __name__ == '__main__':
    start_time = time()
    # 切词
    segment(config.train_path, config.train_seg_path)
    print("spend time:", time() - start_time)
