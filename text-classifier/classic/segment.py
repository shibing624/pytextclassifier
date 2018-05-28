# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from time import time

import jieba.posseg


def segment(in_file, out_file, col_sep='\t', word_sep=' '):
    """
    segment input file to output file
    :param in_file:
    :param out_file:
    :param col_sep:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        count = 0
        content_col = 1
        lbl_col = 0
        for line in fin:
            line = line.strip()
            parts = line.split(col_sep)
            content = parts[content_col]
            lbl = parts[lbl_col]
            seg_line = word_sep.join(jieba.cut(content))
            fout.write(lbl + col_sep + seg_line + "\n")
            count += 1
    print("segment ok. file size:", str(count))


if __name__ == '__main__':
    start_time = time()
    input_path = "../data/test_data/test.txt"
    output_path = "../data/test_data/test_seg.txt"
    segment(input_path, output_path, col_sep=',')
    print("spend time:", time() - start_time)
