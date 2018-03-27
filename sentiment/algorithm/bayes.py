# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/9
# Brief: 贝叶斯
import sys
import gzip
import marshal
from math import log, exp
from utils.freq import AddOneProb


class Bayes:
    def __init__(self):
        self.d = {}
        self.total = 0

    def save(self, fname, iszip=True):
        d = {}
        d["total"] = self.total
        d['d'] = {}
        for k, v in self.d.items():
            d["d"][k] = v.__dict__
            if not iszip:
                marshal.dump(d.open(fname, "wb"))
            else:
                f = gzip.open(fname, "wb")
                f.write(marshal.dumps(d))
                f.close()

    def load(self, fname, iszip=True):
        if not iszip:
            d = marshal.load(open(fname, "rb"))
        else:
            try:
                f = gzip.open(fname, "rb")
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, "rb")
                d = marshal.loads(f.read())
            f.close()
        self.total = d["total"]
        self.d = {}
        for k, v in d['d'].items():
            self.d[k] = AddOneProb()
            self.d[k].__dict__ = v

    def train(self, data):
        for d in data:
            c = d[1]
            if c not in self.d:
                self.d[c] = AddOneProb()
            for word in d[0]:
                self.d[c].add(word, 1)
        for i in set(self.d.keys()):
            self.total += self.d[i].total

    def classify(self, x):
        tmp = {}
        for k in self.d:
            tmp[k] = log(self.d[k].total) - log(self.total)
            for word in x:
                tmp[k] += log(self.d[k].freq(word))
        ret, prob = 0, 0
        for k in self.d:
            now = 0
            try:
                for other_k in self.d:
                    now += exp(tmp[other_k] - tmp[k])
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = k, now
        return ret, prob
