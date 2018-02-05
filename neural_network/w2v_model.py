# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from neural_network.utils.io_util import read_lines
import config


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
    model = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                     size=256, window=5, min_count=3, workers=4, iter=40)
    model.wv.save_word2vec_format(out_bin_path, binary=True)
    # save model
    model = KeyedVectors.load_word2vec_format(out_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    with open(out_path, 'wb') as f:
        pickle.dump(word_dict, f)
        print(f.name)


if __name__ == '__main__':
    train(config.train_seg_path,
          config.test_seg_path,
          config.sentence_path,
          config.sentence_w2v_path,
          out_bin_path=config.sentence_w2v_bin_path)
