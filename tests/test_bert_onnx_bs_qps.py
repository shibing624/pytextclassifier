# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import shutil
import sys
import time
import unittest

import numpy as np
import torch

sys.path.append('..')
from pytextclassifier import BertClassifier


class ModelSpeedTestCase(unittest.TestCase):

    def test_classifier_diff_batch_size(self):
        # Helper function to calculate QPS and 95th percentile latency
        def calculate_metrics(times):
            completion_times = np.array(times)
            total_requests = len(completion_times)
            average_qps = total_requests / np.sum(completion_times)
            sorted_times = np.sort(completion_times)
            p95_latency = sorted_times[int(0.95 * len(sorted_times))]
            return average_qps, p95_latency

        # Train the model once
        def train_model():
            m = BertClassifier(output_dir='models/bert-chinese-v1', num_classes=2,
                               model_type='bert', model_name='bert-base-chinese', num_epochs=1)
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
            return m

        # Evaluate performance for a given batch size
        def evaluate_performance(m, batch_size):
            samples = ['名师指导托福语法技巧',
                       '米兰客场8战不败',
                       '恒生AH溢指收平 A股对H股折价1.95%'] * 100

            batch_times = []

            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                start_time = time.time()
                m.predict(batch_samples)
                end_time = time.time()

                batch_times.append(end_time - start_time)

            avg_qps, p95_latency = calculate_metrics(batch_times)
            return avg_qps, p95_latency

        # Convert the model to ONNX format
        def convert_model_to_onnx(m):
            save_onnx_dir = 'models/bert-chinese-v1/onnx'
            m.model.convert_to_onnx(save_onnx_dir)
            shutil.copy(m.label_vocab_path, save_onnx_dir)
            return save_onnx_dir

        # Main function
        batch_sizes = [8, 16, 32, 64, 128, 256, 512]  # Modify these values as appropriate

        # Train the model once
        model = train_model()

        # Evaluate Standard BERT model performance
        for batch_size in batch_sizes:
            model.args['eval_batch_size'] = batch_size
            avg_qps, p95_latency = evaluate_performance(model, batch_size)
            print(
                f'Standard BERT model - Batch size: {batch_size}, Average QPS: {avg_qps:.2f}, P95 Latency: {p95_latency:.4f} seconds')

        # Convert to ONNX
        onnx_model_path = convert_model_to_onnx(model)
        del model
        torch.cuda.empty_cache()

        # Load and evaluate ONNX model performance
        for batch_size in batch_sizes:
            onnx_model = BertClassifier(output_dir=onnx_model_path, num_classes=2, model_type='bert',
                                        model_name=onnx_model_path,
                                        args={"eval_batch_size": batch_size, "onnx": True})
            onnx_model.load_model()
            avg_qps, p95_latency = evaluate_performance(onnx_model, batch_size)
            print(
                f'ONNX model - Batch size: {batch_size}, Average QPS: {avg_qps:.2f}, P95 Latency: {p95_latency:.4f} seconds')
            del onnx_model
            torch.cuda.empty_cache()

        # Clean up
        shutil.rmtree('models')


if __name__ == '__main__':
    unittest.main()
