# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from time import time
import config
import jieba
from codecs import open

class Bigram_Tokenizer():
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        self.n += 1
        if self.n % 10000 == 0:
            print(self.n, end=' ')
            print(line)
            print('=' * 20)
            print(tokens)
        return tokens


def seg_data(in_file, out_file, col_sep='\t'):
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        count = 0
        for line in f1:
            line = line.rstrip()
            parts = line.split(col_sep)
            if len(parts) < 2:
                continue
            label_str = parts[0].strip()
            label = label_str
            data = ' '.join(parts[1:])
            seg_line = ' '.join(jieba.cut(data))
            if count % 10000 == 0:
                print('count:', count)
                print(line)
                print('=' * 20)
                print(seg_line)
            count += 1
            f2.write(str(label) + '\t' + seg_line + "\n")
        print('%s to %s, size: %d' % (in_file, out_file, count))


if __name__ == '__main__':
    start_time = time()
    seg_data(config.train_path, config.train_seg_path, col_sep=config.col_sep)
    seg_data(config.test_path, config.test_seg_path, col_sep=config.col_sep)
    print("spend time:", time() - start_time)
