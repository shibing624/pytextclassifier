# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import time

import numpy as np
from sklearn.model_selection import KFold

import config
from generate_vocab import load_emb, load_vocab, load_train_data, load_test_data
from model import Model
from utils.io_util import clear_directory


def train():
    # generate vocab and load dict
    word_emb, pos_emb = load_emb(config.w2v_train_path, config.p2v_path)
    word_vocab, pos_vocab, label_vocab = load_vocab(config.word_vocab_path,
                                                    config.pos_vocab_path,
                                                    config.label_vocab_path)
    sentences, pos, labels = load_train_data(word_vocab, pos_vocab, label_vocab)
    # shuffle
    seed = 137
    np.random.seed(seed)
    np.random.shuffle(sentences)
    np.random.seed(seed)
    np.random.shuffle(pos)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # load data
    sentences_test, pos_test = load_test_data(word_vocab, pos_vocab, label_vocab)
    labels_test = None

    # clear
    clear_directory(config.model_save_dir)
    clear_directory(config.model_save_temp_dir)

    # 划分训练、开发、测试集，十折交叉验证
    kf = KFold(n_splits=config.kfold)
    train_indices, dev_indices = [], []
    for train_index, dev_index in kf.split(labels):
        train_indices.append(train_index)
        dev_indices.append(dev_index)
    for num in range(config.kfold):
        train_index, dev_index = train_indices[num], dev_indices[num]
        sentences_train, sentences_dev = sentences[train_index], sentences[dev_index]
        pos_train, pos_dev = pos[train_index], pos[dev_index]
        labels_train, labels_dev = labels[train_index], labels[dev_index]

        # init model
        model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
        # fit model
        model.fit(sentences_train, pos_train, labels_train,
                  sentences_dev, pos_dev, labels_dev,
                  sentences_test, pos_test, labels_test,
                  config.batch_size, config.nb_epoch, config.keep_prob,
                  config.word_keep_prob, config.pos_keep_prob)
        print(model.get_best_score())
        [p_test, r_test, f_test], nb_epoch = model.get_best_score()
        print(p_test, r_test, f_test, nb_epoch)
        # save best result
        cmd = 'cp %s/epoch_%d.csv %s/best_%d.csv' % \
              (config.model_save_temp_dir, nb_epoch + 1, config.model_save_dir, num)
        print(cmd)
        os.popen(cmd)

        # clear model
        model.clear_model()


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("spend time %ds." % (time.time() - start_time))
