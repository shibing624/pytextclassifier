# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import time

import config
from models.model import Model
from neural_network.reader import load_emb
from neural_network.reader import load_vocab
from neural_network.reader import test_reader
from utils.tensor_utils import get_ckpt_path


def infer(data_path, model_path,
          word_vocab_path, pos_vocab_path, label_vocab_path,
          word_emb_path, pos_emb_path, batch_size):
    word_vocab, pos_vocab, label_vocab = load_vocab(word_vocab_path,
                                                    pos_vocab_path,
                                                    label_vocab_path)
    word_emb, pos_emb = load_emb(word_emb_path, pos_emb_path)
    word_test, pos_test = test_reader(data_path, word_vocab, pos_vocab, label_vocab)

    model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
    ckpt_path = get_ckpt_path(model_path)
    if ckpt_path:
        print("Read model parameters from %s" % ckpt_path)
        model.saver.restore(model.sess, ckpt_path)
    else:
        print("Can't find the checkpoint.going to stop")
        return
    pred_lbls = model.predict(word_test, pos_test, batch_size)
    for i in pred_lbls:
        print(i)


if __name__ == '__main__':
    start_time = time.time()
    infer(config.test_seg_path,
          config.model_save_dir,
          config.word_vocab_path,
          config.pos_vocab_path,
          config.label_vocab_path,
          config.w2v_path,
          config.p2v_path,
          config.batch_size)
    print("spend time %ds." % (time.time() - start_time))
