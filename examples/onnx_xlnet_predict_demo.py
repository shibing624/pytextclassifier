# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import shutil
import sys
import time

import torch

sys.path.append('..')
from pytextclassifier import BertClassifier

if __name__ == '__main__':
    m = BertClassifier(output_dir='models/xlnet-chinese-v1', num_classes=2,
                       model_type='xlnet', model_name='hfl/chinese-xlnet-base', num_epochs=1)
    data = [
        ('education', '名师指导托福语法技巧：名词的复数形式'),
        ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事1？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事2？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事3？'),
        ('education', '公务员考虑越来越吃香，这是怎么回事4？'),
        ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
        ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与'),
        ('sports', '米兰客场8战不败国米10年连胜1'),
        ('sports', '米兰客场8战不败国米10年连胜2'),
        ('sports', '米兰客场8战不败国米10年连胜3'),
        ('sports', '米兰客场8战不败国米10年连胜4'),
        ('sports', '米兰客场8战不败国米10年连胜5'),
    ]
    m.train(data * 1)
    m.load_model()

    samples = ['名师指导托福语法技巧',
               '米兰客场8战不败',
               '恒生AH溢指收平 A股对H股折价1.95%'] * 10

    start_time = time.time()
    predict_label_bert, predict_proba_bert = m.predict(samples)
    print(f'predict_label_bert size: {len(predict_label_bert)}')
    end_time = time.time()
    elapsed_time_bert = end_time - start_time
    print(f'Standard xlnet model prediction time: {elapsed_time_bert} seconds')

    # convert to onnx, and load onnx model to predict, speed up 10x
    save_onnx_dir = 'models/xlnet-chinese-v1/onnx'
    m.model.convert_to_onnx(save_onnx_dir)
    # copy label_vocab.json to save_onnx_dir
    if os.path.exists(m.label_vocab_path):
        shutil.copy(m.label_vocab_path, save_onnx_dir)

    # Manually delete the model and clear CUDA cache
    del m
    torch.cuda.empty_cache()

    m = BertClassifier(output_dir=save_onnx_dir, num_classes=2, model_type='xlnet', model_name=save_onnx_dir,
                       args={"onnx": True})
    m.load_model()
    start_time = time.time()
    predict_label_bert, predict_proba_bert = m.predict(samples)
    print(f'predict_label_bert size: {len(predict_label_bert)}')
    end_time = time.time()
    elapsed_time_onnx = end_time - start_time
    print(f'ONNX model prediction time: {elapsed_time_onnx} seconds')
