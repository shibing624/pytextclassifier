#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@description:测试程序
"""

def demo_dict():
    """
    test classifier based on sentiment dict
    :return: 
    """
    from classifier import DictClassifier
    dc = DictClassifier()
    # test single sentence
    sentence1 = "土豆丝我觉得很好吃"
    result = dc.analyse_sentence(sentence1,"demo_result.out",True)
    print(result)

demo_dict()
