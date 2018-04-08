# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from utils.data_utils import read_lines


def data_reader(path, col_sep=','):
    contents, labels = [], []
    word_col = 1
    lbl_col = 0
    lines = read_lines(path)
    for line in lines:
        line_split = line.split(col_sep)
        if line_split and len(line_split) > 1:
            content = line_split[word_col].strip()
            label = line_split[lbl_col].strip()
            contents.append(content)
            labels.append(label)
    return contents, labels
