# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 训练模型
import os
from analysis import sentiment

neg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neg.txt")
pos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pos.txt")
sentiment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.marshal")
sentiment.train(neg_path, pos_path)
sentiment.save(sentiment_path)
