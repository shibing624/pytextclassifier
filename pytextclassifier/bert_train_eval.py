# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import sys

sys.path.append('..')
from pytextclassifier.utils.nn_utils import get_time_dif


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    # LayerNorm,bias是不需要decay的
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "bert" in n], "lr": config.bert_learning_rate,
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if "bert" not in n], "lr": config.other_learning_rate,
         'weight_decay': 0.01},
    ]
    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.05,
                                                num_training_steps=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # empty list to save model predictions
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, label_ids) in enumerate(train_iter):
            data = [t.to(config.device) for t in trains]
            label_ids = label_ids.to(config.device)
            outputs = model(data)
            model.zero_grad()
            # F.cross_entropy combines `log_softmax` and `nll_loss`
            loss = F.cross_entropy(outputs, label_ids)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                y_true = label_ids.data.cpu().numpy()
                y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
                train_acc = metrics.accuracy_score(y_true, y_pred)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, (trains, label_ids) in enumerate(data_iter):
            data = [t.to(config.device) for t in trains]
            label_ids = label_ids.to(config.device)
            # (x, seq_len, mask), y
            outputs = model(data)
            loss = F.cross_entropy(outputs, label_ids)
            loss_total += loss
            labels = label_ids.cpu().numpy()
            preds = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, preds)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all,
                                               target_names=config.class_list,
                                               digits=len(config.class_list))
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
