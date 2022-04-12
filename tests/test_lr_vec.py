# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest

import sys

sys.path.append('..')
from pytextclassifier import ClassicClassifier


class VecTestCase(unittest.TestCase):
    def setUp(self):
        m = ClassicClassifier('models/lr')
        data = [
            ('education', 'Student debt to cost Britain billions within decades'),
            ('education', 'Chinese education for TV experiment'),
            ('sports', 'Middle East and Asia boost investment in top level sports'),
            ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
        ]
        m.train(data)
        print('model trained:', m)

    def test_classifier(self):
        new_m = ClassicClassifier('models/lr')
        new_m.load_model()
        r, _ = new_m.predict(['Abbott government spends $8 million on higher education media blitz',
                              'Middle East and Asia boost investment in top level sports'])
        print(r)
        self.assertTrue(r[0] == 'education')

    def test_vec(self):
        new_m = ClassicClassifier('models/lr')
        new_m.load_model()
        print(new_m.feature.get_feature_names())
        print('feature name size:', len(new_m.feature.get_feature_names()))
        self.assertTrue(len(new_m.feature.get_feature_names()) > 0)

    def test_stopwords(self):
        new_m = ClassicClassifier('models/lr')
        new_m.load_model()
        stopwords = new_m.stopwords
        print(len(stopwords))
        self.assertTrue(len(stopwords) > 0)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree('models')
        print('remove dir: models')


if __name__ == '__main__':
    unittest.main()
