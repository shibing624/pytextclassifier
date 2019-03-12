# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from codecs import open
from time import time

import jieba

import config
from utils.io_utils import get_logger

logger = get_logger(__name__)


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
            logger.info(self.n)
            logger.info(line)
            logger.info('=' * 20)
            logger.info(tokens)
        return tokens


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
                logger.info('count:%d' % count)
                logger.info(line)
                logger.info('=' * 20)
                logger.info(seg_line)
            count += 1
            f2.write('%s\t%s\n' % (label, seg_line))
        logger.info('%s to %s, size: %d' % (in_file, out_file, count))


if __name__ == '__main__':
    start_time = time()
    seg_data(config.train_path, config.train_seg_path, col_sep=config.col_sep, stop_words_path=config.stop_words_path)
    seg_data(config.test_path, config.test_seg_path, col_sep=config.col_sep, stop_words_path=config.stop_words_path)
    logger.info("spend time: %s s" % (time() - start_time))
