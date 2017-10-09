# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 词频-概率工具


class BaseProb:
    def __init__(self):
        self.d = {}
        self.total = 0
        self.none = 0

    def exists(self, key):
        return key in self.d

    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    def freq(self, key):
        return float(self.get(key)[1]) / self.total

    def sample(self):
        return self.d.keys()


class AddOneProb(BaseProb):
    def __init__(self):
        self.d = {}
        self.total = 0
        self.none = 1

    def add(self, key, value):
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        else:
            self.d[key] += value
            self.total += value
