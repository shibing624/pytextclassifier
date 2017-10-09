# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 过滤停用词

import os
import re

stop_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")

stop_words_set = set()
with open(stop_words_path, "r", encoding="utf-8")as f:
    for word in f:
        stop_words_set.add(word.strip())


def filter_stop_words(words):
    return [w for w in words if w not in stop_words_set]
