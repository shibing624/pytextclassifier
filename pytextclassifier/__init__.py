# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
__version__ = '1.3.4'

from pytextclassifier.classic_classifier import ClassicClassifier
from pytextclassifier.fasttext_classifier import FastTextClassifier
from pytextclassifier.textcnn_classifier import TextCNNClassifier
from pytextclassifier.textrnn_classifier import TextRNNClassifier
from pytextclassifier.bert_classifier import BertClassifier
from pytextclassifier.base_classifier import load_data
from pytextclassifier.textcluster import TextCluster
from pytextclassifier.bert_classification_model import BertClassificationModel, BertClassificationArgs
