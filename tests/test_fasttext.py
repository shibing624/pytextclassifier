# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import unittest

import sys

sys.path.append('..')
from pytextclassifier import TextClassifier
import torch

class SaveModelTestCase(unittest.TestCase):
    def test_classifier(self):
        m = TextClassifier(model_name='fasttext')
        data = [
            ('education', 'Student debt to cost Britain billions within decades'),
            ('education', 'Chinese education for TV experiment'),
            ('sports', 'Middle East and Asia boost investment in top level sports'),
            ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
        ]
        m.train(data)
        m.load_model()
        samples = ['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports']
        r, p = m.predict(samples)
        print(r, p)
        print('-' * 20)
        torch.save(m.model, 'model.pkl')
        model = torch.load('model.pkl')
        m.model = model
        r, p = m.predict(samples)
        print(r,p)

if __name__ == '__main__':
    unittest.main()
