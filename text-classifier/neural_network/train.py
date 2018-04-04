# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import time

from sklearn.model_selection import train_test_split

import config
from models.model import Model
from reader import build_pos_embedding
from reader import build_vocab
from reader import build_word_embedding
from reader import load_emb
from reader import load_vocab
from reader import test_reader
from reader import train_reader
from utils.io_utils import clear_directory


def train():
    # 1.build vocab for train data
    build_vocab(config.train_seg_path, config.word_vocab_path,
                config.pos_vocab_path, config.label_vocab_path)
    word_vocab, pos_vocab, label_vocab = load_vocab(config.word_vocab_path,
                                                    config.pos_vocab_path,
                                                    config.label_vocab_path)
    # 2.build embedding
    build_word_embedding(config.w2v_path, overwrite=True)
    build_pos_embedding(config.p2v_path, overwrite=True)
    word_emb, pos_emb = load_emb(config.w2v_path, config.p2v_path)

    # 3.data reader
    words, pos, labels = train_reader(config.train_seg_path, word_vocab, pos_vocab, label_vocab)
    word_test, pos_test = test_reader(config.test_seg_path, word_vocab, pos_vocab, label_vocab)
    labels_test = None

    # clear
    clear_directory(config.model_save_dir)

    # Division of training, development, and test set
    word_train, word_dev, pos_train, pos_dev, label_train, label_dev = train_test_split(
        words, pos, labels, test_size=0.2, random_state=42)

    # init model
    model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
    # fit model
    model.fit(word_train, pos_train, label_train,
              word_dev, pos_dev, label_dev,
              word_test, pos_test, labels_test,
              config.batch_size, config.nb_epoch, config.keep_prob,
              config.word_keep_prob, config.pos_keep_prob)
    [p_test, r_test, f_test], nb_epoch = model.get_best_score()
    print('P@test:%f, R@test:%f, F@test:%f, num_best_epoch:%d' % (p_test, r_test, f_test, nb_epoch + 1))
    # save best pred label
    cmd = 'cp %s/epoch_%d.csv %s/best.csv' % (config.model_save_temp_dir, nb_epoch + 1, config.model_save_dir)
    print(cmd)
    os.popen(cmd)
    # save best model
    cmd = 'cp %s/model_%d.* %s/' % (config.model_save_temp_dir, nb_epoch + 1, config.model_save_dir)
    print(cmd)
    os.popen(cmd)
    # model.save('%s/cnn_classification_model' % config.model_save_dir)

    # clear model
    model.clear_model()


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("spend time %ds." % (time.time() - start_time))
