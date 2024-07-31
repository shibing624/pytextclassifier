# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import shutil
import sys
import time
import unittest

sys.path.append('..')
from pytextclassifier import BertClassifier


class ModelSpeedTestCase(unittest.TestCase):
    def test_classifier(self):
        m = BertClassifier(output_dir='models/bert-chinese', num_classes=2,
                           model_type='bert', model_name='bert-base-chinese', num_epochs=2)
        data = [
            ('education', '名师指导托福语法技巧：名词的复数形式'),
            ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
            ('education', '公务员考虑越来越吃香，这是怎么回事？'),
            ('education', '公务员考虑越来越吃香，这是怎么回事1？'),
            ('education', '公务员考虑越来越吃香，这是怎么回事2？'),
            ('education', '公务员考虑越来越吃香，这是怎么回事3？'),
            ('education', '公务员考虑越来越吃香，这是怎么回事4？'),
            ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
            ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与'),
            ('sports', '米兰客场8战不败国米10年连胜1'),
            ('sports', '米兰客场8战不败国米10年连胜2'),
            ('sports', '米兰客场8战不败国米10年连胜3'),
            ('sports', '米兰客场8战不败国米10年连胜4'),
            ('sports', '米兰客场8战不败国米10年连胜5'),
        ]
        m.train(data * 10)
        m.load_model()

        samples = ['名师指导托福语法技巧',
                   '米兰客场8战不败',
                   '恒生AH溢指收平 A股对H股折价1.95%'] * 100

        start_time = time.time()
        predict_label_bert, predict_proba_bert = m.predict(samples)
        print(f'predict_label_bert size: {len(predict_label_bert)}')
        self.assertEqual(len(predict_label_bert), 300)
        end_time = time.time()
        elapsed_time_bert = end_time - start_time
        print(f'Standard BERT model prediction time: {elapsed_time_bert} seconds')

        # convert to onnx, and load onnx model to predict, speed up 10x
        save_onnx_dir = 'models/onnx'
        m.model.convert_to_onnx(save_onnx_dir)
        # copy label_vocab.json to save_onnx_dir
        shutil.copy('models/bert-chinese/label_vocab.json', save_onnx_dir)
        m = BertClassifier(output_dir=save_onnx_dir, num_classes=2, model_type='bert', model_name=save_onnx_dir,
                           args={"onnx": True})
        m.load_model()
        start_time = time.time()
        predict_label_bert, predict_proba_bert = m.predict(samples)
        print(f'predict_label_bert size: {len(predict_label_bert)}')
        end_time = time.time()
        elapsed_time_onnx = end_time - start_time
        print(f'ONNX model prediction time: {elapsed_time_onnx} seconds')

        self.assertEqual(len(predict_label_bert), 300)
        shutil.rmtree('models')


if __name__ == '__main__':
    unittest.main()
