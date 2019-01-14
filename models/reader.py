# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: read train and test data


def data_reader(path, col_sep='\t'):
    contents, labels = [], []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep in line:
                index = line.index(col_sep)
                label = line[:index].strip()
                labels.append(label)
                content = line[index + 1:].strip()
            else:
                content = line
            contents.append(content)
    return contents, labels
