# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: read train and test data
"""
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


if __name__ == '__main__':
    a, b = data_reader('../data/test_seg_sample.txt', '\t')
    print(a[:3], b[:3])
