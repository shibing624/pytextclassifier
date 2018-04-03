# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

import config
from utils.data_utils import dump_pkl
from utils.io_utils import read_lines


def get_sentence(sentence_tag, word_sep=' ', pos_sep='/'):
    """
    文本拼接
    :param sentence_tag:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    words = []
    for item in sentence_tag.split(word_sep):
        index = item.rindex(pos_sep)
        words.append(item[:index])
    return word_sep.join(words)


def get_sentence_without_pos(sentence_tag, word_sep=' '):
    return sentence_tag.split(word_sep)


def extract_sentence(train_seg_path, test_seg_path, col_sep=','):
    ret = []
    lines = read_lines(train_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        index = line.index(col_sep)
        word_tag = line[index + 1:]
        sentence = ''.join(get_sentence(word_tag))
        ret.append(sentence)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def train(train_seg_path, test_seg_path, out_path, sentence_path=None, w2v_bin_path="w2v.bin"):
    sentences = extract_sentence(train_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=config.min_count,
                   workers=config.num_workers, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('公司', '市场')
    print('公司 vs 市场 similarity score:', sim)
    # save model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    train(config.train_seg_path,
          config.test_seg_path,
          config.sentence_w2v_path,
          config.sentence_path,
          config.sentence_w2v_bin_path)
