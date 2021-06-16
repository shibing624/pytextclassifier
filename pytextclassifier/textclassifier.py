# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from pytextclassifier.utils.tokenizer import Tokenizer


class TextClassifier(object):
    def __init__(self, model_name, tokenizer=Tokenizer()):
        self.model_name = model_name
        self.tokenizer = tokenizer

    def __repr__(self):
        return 'TextClassifier instance ({}, {})'.format(self.model_name, self.tokenizer)

