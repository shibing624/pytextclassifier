# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: read train and test data


def data_reader(path, col_sep=','):
    contents, labels = [], []
    word_col = 1
    lbl_col = 0
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line_split = line.split(col_sep, 1)
            if line_split and len(line_split) > 1:
                # train data
                content = line_split[word_col].strip()
                label = line_split[lbl_col].strip()
                contents.append(content)
                labels.append(label)
            else:
                # read infer data
                contents.append(line)
    return contents, labels
