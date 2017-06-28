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
    # test single sentence1
    sentence1 = "土豆丝我觉得很好吃"
    result = dc.analyse_sentence(sentence1, "demo_result.out", True)
    print("result: ", result)

    # test single sentence2
    sentence1 = "啊啊啊，要难吃死了。这土豆丝非常烂！"
    result = dc.analyse_sentence(sentence1, "demo_result.out", True)
    print("result: ", result)


demo_dict()
