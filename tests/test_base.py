# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest

import sys

sys.path.append('..')
from pytextclassifier import TextClassifier


class BaseTestCase(unittest.TestCase):
    def test_classifier(self):
        m = TextClassifier()
        data = [
            ('education', 'Student debt to cost Britain billions within decades'),
            ('education', 'Chinese education for TV experiment'),
            ('sports', 'Middle East and Asia boost investment in top level sports'),
            ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
        ]
        m.train(data)
        m.load_model()
        r, _ = m.predict(['Abbott government spends $8 million on higher education media blitz',
                          'Middle East and Asia boost investment in top level sports'])
        print(r)
        self.assertEqual(r[0], 'education')
        self.assertEqual(r[1], 'sports')
        test_data = [
            ('education', 'Abbott government spends $8 million on higher education media blitz'),
            ('sports', 'Middle East and Asia boost investment in top level sports'),
        ]
        acc_score = m.evaluate(test_data)
        print(acc_score)  # 1.0
        self.assertEqual(acc_score, 1.0)
        import shutil
        shutil.rmtree('lr')


if __name__ == '__main__':
    unittest.main()
