# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

import config
from neural_network.utils.data_util import dump_pkl
from neural_network.utils.io_util import read_lines


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


def extract_sentence(train_seg_path, test_seg_path, sentence_path):
    lines = read_lines(train_seg_path)
    lines += read_lines(test_seg_path)
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            index = line.index(',')
            word_tag = line[index + 1:]
            f.write('%s\n' % get_sentence(word_tag))
        return True
    return False


def train(train_seg_path, test_seg_path, sentence_path, out_path, out_bin_path="w2v.bin"):
    if not extract_sentence(train_seg_path, test_seg_path, sentence_path):
        print("extract sentence error.")
        return False
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=config.min_count,
                   workers=config.num_workers, iter=40)
    w2v.wv.save_word2vec_format(out_bin_path, binary=False)
    print("save %s ok." % out_bin_path)
    # test
    sim1 = w2v.wv.most_similar(positive=['基金', '资金'], negative=['服务'])
    print('基金 vs 资金 similarity word:', sim1)
    sim2 = w2v.wv.similarity('基金', '资金')
    print('基金 vs 资金 similarity score:', sim2)
    # save model
    model = KeyedVectors.load_word2vec_format(out_bin_path, binary=False)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)
    print("save %s ok." % out_path)


if __name__ == '__main__':
    train(config.train_seg_path,
          config.test_seg_path,
          config.sentence_path,
          config.w2v_path,
          out_bin_path=config.w2v_bin_path)
