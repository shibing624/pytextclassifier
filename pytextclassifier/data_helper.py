# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import random

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def set_seed(seed):
    """
    Set seed for random number generators.
    """
    logger.info(f"Set seed for random, numpy and torch: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(contents, tokenizer, max_size, min_freq, unk_token, pad_token):
    vocab_dic = {}
    for line in tqdm(contents):
        line = line.strip()
        if not line:
            continue
        content = line.split('\t')[0]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({unk_token: len(vocab_dic), pad_token: len(vocab_dic) + 1})
    return vocab_dic


def load_vocab(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    with open(vocab_path, 'r', encoding='utf-8') as fr:
        vocab = json.load(fr)
    return vocab
