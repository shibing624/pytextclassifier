# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief:
from sentiment import NLP
s = NLP("我喜欢这油画")
print(s.words)
print(s.pinyin)
print(s.sentiments)

s = NLP("我看了这油画半天，没看出什么有意思的内容")
print(s.words)
print(s.pinyin)
print(s.sentiments)

s = NLP("我看了这油画半天，真是讨厌的背景")
print(s.words)
print(s.pinyin)
print(s.sentiments)

s = NLP("我看了这油画半天，真是大爱的名作")
print(s.words)
print(s.pinyin)
print(s.sentiments)

print("*"*36)
s = NLP("啊啊啊，要难吃死了。这土豆丝非常烂！")
print(s.words)
print(s.pinyin)
print(s.sentiments)

print("*"*36)
s = NLP("这土豆丝非常美味！")
print(s.words)
print(s.pinyin)
print(s.sentiments)