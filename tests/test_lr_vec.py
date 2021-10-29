# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest

import sys

sys.path.append('..')
from pytextclassifier import TextClassifier


class VecTestCase(unittest.TestCase):
    def test_classifier(self):
        m = TextClassifier()
        data = [
            ('education', 'Student debt to cost Britain billions within decades'),
            ('education', 'Chinese education for TV experiment'),
            ('sports', 'Middle East and Asia boost investment in top level sports'),
            ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
        ]
        m.train(data)
        r, _ = m.predict(['Abbott government spends $8 million on higher education media blitz',
                          'Middle East and Asia boost investment in top level sports'])
        print(r)

    def test_vec(self):
        new_m = TextClassifier(model_name='lr')
        new_m.load_model()
        print(new_m.vectorizer.get_feature_names())
        print('feature name size:', len(new_m.vectorizer.get_feature_names()))

    def test_stopwords(self):
        new_m = TextClassifier(model_name='lr')
        new_m.load_model()
        from pytextclassifier.tools.lr_classification import stopwords
        print(len(stopwords))


if __name__ == '__main__':
    unittest.main()
