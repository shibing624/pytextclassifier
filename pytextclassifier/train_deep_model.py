# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import time
from importlib import import_module
import numpy as np
import torch
from pytextclassifier.utils.log import logger
from pytextclassifier.utils.nn_utils import get_time_dif
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: textcnn, textrnn, fasttext, bilstm_att, bert_fc')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, FastText, TextRNN_Att
    if model_name == 'fasttext':
        #from pytextclassifier.utils.fasttext_utils import build_dataset, build_iterator

        embedding = 'random'
    else:
        # from pytextclassifier.utils.nn_utils import build_dataset, build_iterator
        pass

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logger.debug("Loading data...")
    vocab, train_data, dev_data, test_data = x.build_dataset(config, args.word)
    train_iter = x.build_iterator(train_data, config)
    dev_iter = x.build_iterator(dev_data, config)
    test_iter = x.build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:{}".format(time_dif))

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    x.init_network(model)
    logger.info(model.parameters)
    x.train(config, model, train_iter, dev_iter, test_iter)
    logger.info("finish train.")
