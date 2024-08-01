# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
import shutil
import sys
import time
import unittest

import numpy as np
import torch
from loguru import logger

sys.path.append('..')
from pytextclassifier import BertClassifier


class ModelSpeedTestCase(unittest.TestCase):

    def test_classifier_diff_batch_size(self):
        # Helper function to calculate QPS and 95th percentile latency
        def calculate_metrics(times):
            completion_times = np.array(times)
            total_requests = 300

            # 平均每秒请求数（QPS），计算公式为总请求数除以总耗时
            average_qps = total_requests / np.sum(completion_times)
            latency = np.sum(completion_times) / total_requests

            # 返回所有计算结果
            return {
                'total_requests': total_requests,  # 总请求数
                'latency': latency,  # 每条请求的平均完成时间
                'average_qps': average_qps,  # 平均每秒请求数
            }

        # Train the model once
        def train_model(output_dir='models/bert-chinese-v1'):
            m = BertClassifier(output_dir=output_dir, num_classes=2,
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

            metrics = calculate_metrics(batch_times)
            return metrics

        # Convert the model to ONNX format
        def convert_model_to_onnx(m, model_dir='models/bert-chinese-v1'):
            save_onnx_dir = os.path.join(model_dir, 'onnx')
            if os.path.exists(save_onnx_dir):
                shutil.rmtree(save_onnx_dir)
            m.model.convert_to_onnx(save_onnx_dir)
            shutil.copy(m.label_vocab_path, save_onnx_dir)
            return save_onnx_dir

        # Main function
        batch_sizes = [1, 8, 16, 32, 64, 128]  # Modify these values as appropriate
        model_dir = 'models/bert-chinese-v1'
        # Train the model once
        model = train_model(model_dir)
        # Convert to ONNX
        onnx_model_path = convert_model_to_onnx(model)
        del model
        torch.cuda.empty_cache()

        # Evaluate Standard BERT model performance
        for batch_size in batch_sizes:
            model = BertClassifier(output_dir=model_dir, num_classes=2, model_type='bert',
                                   model_name=model_dir,
                                   args={"eval_batch_size": batch_size, "onnx": False})
            model.load_model()
            metrics = evaluate_performance(model, batch_size)
            logger.info(
                f'Standard BERT model - Batch size: {batch_size}, total_requests: {metrics["total_requests"]}, '
                f'Average QPS: {metrics["average_qps"]:.2f}, Average Latency: {metrics["latency"]:.4f}')
            del model
            torch.cuda.empty_cache()

        # Load and evaluate ONNX model performance
        for batch_size in batch_sizes:
            onnx_model = BertClassifier(output_dir=onnx_model_path, num_classes=2, model_type='bert',
                                        model_name=onnx_model_path,
                                        args={"eval_batch_size": batch_size, "onnx": True})
            onnx_model.load_model()
            metrics = evaluate_performance(onnx_model, batch_size)
            logger.info(
                f'ONNX model - Batch size: {batch_size}, total_requests: {metrics["total_requests"]}, '
                f'Average QPS: {metrics["average_qps"]:.2f}, Average Latency: {metrics["latency"]:.4f}')
            del onnx_model
            torch.cuda.empty_cache()

        # Clean up
        shutil.rmtree('models')


if __name__ == '__main__':
    unittest.main()
