# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from codecs import open

import jieba


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def seg_data(in_file, out_file, col_sep='\t', stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        count = 0
        for line in f1:
            line = line.rstrip()
            parts = line.split(col_sep)
            if len(parts) < 2:
                continue
            label = parts[0].strip()
            data = ' '.join(parts[1:])
            seg_list = jieba.lcut(data)
            seg_words = []
            for i in seg_list:
                if i in stopwords:
                    continue
                seg_words.append(i)
            seg_line = ' '.join(seg_words)
            if count % 10000 == 0:
                print('count:%d' % count)
                print(line)
                print('=' * 20)
                print(seg_line)
            count += 1
            f2.write('%s\t%s\n' % (label, seg_line))
        print('%s to %s, size: %d' % (in_file, out_file, count))


if __name__ == '__main__':
    from pytextclassifier import config
    from time import time

    start_time = time()
    seg_data(config.train_path, config.train_seg_path, col_sep=config.col_sep, stop_words_path=config.stop_words_path)
    seg_data(config.test_path, config.test_seg_path, col_sep=config.col_sep, stop_words_path=config.stop_words_path)
    print("spend time: %s s" % (time() - start_time))
