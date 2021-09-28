# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

from pytextclassifier.utils.data_utils import read_lines, save_pkl


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
        if pos_sep in item:
            index = item.rindex(pos_sep)
            words.append(item[:index])
        else:
            words.append(item.strip())
    return word_sep.join(words)


def get_sentence_without_pos(sentence_tag, word_sep=' '):
    return sentence_tag.split(word_sep)


def extract_sentence(train_seg_path, test_seg_path, col_sep='\t'):
    ret = []
    lines = read_lines(train_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        if col_sep in line:
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


def build(train_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1, col_sep='\t'):
    sentences = extract_sentence(train_seg_path, test_seg_path, col_sep=col_sep)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    # sim = w2v.wv.similarity('大', '小')
    # print('大 vs 小 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    save_pkl(word_dict, out_path, overwrite=True)
