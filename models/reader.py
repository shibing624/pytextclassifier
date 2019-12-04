# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: read train and test data

from codecs import open


def data_reader(path, col_sep='\t'):
    contents, labels = [], []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep in line:
                index = line.index(col_sep)
                label = line[:index].strip()
                content = line[index + 1:].strip()
                if not content:
                    print('error, ', line)
                    continue
                if not label:
                    print('error, ', line)
                    continue
                labels.append(label)
                contents.append(content)
    return contents, labels
