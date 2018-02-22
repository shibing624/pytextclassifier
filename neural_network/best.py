# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time

import numpy as np

import config
from utils.io_util import read_lines


def read_model_data():
    labels = []
    for i in range(config.kfold):
        lines = read_lines(config.model_save_dir + '/best_%d.csv' % i)
        temp = []
        for line in lines:
            parts = line.split(',')
            if len(parts) < 2:
                continue
            temp.append(parts[1])
        labels.append(temp)
    return labels


def best():
    dataes = read_model_data()
    data_count = len(dataes[0])
    label_type_count = config.nb_labels
    labels = np.zeros((data_count, label_type_count))
    for data in dataes:
        for i, label in enumerate(data):
            label_id = int(label) - 1
            labels[i][label_id] += 1

    # get final label
    final_labels = []
    for item in labels:
        label = item.argmax() + 1
        final_labels.append(label)

    with open(config.infer_result_path, 'w', encoding='utf-8') as f:
        for i, label in enumerate(final_labels):
            f.write('%d,%d\n' % (i + 1, label))
    print('result: %s' % f.name)


if __name__ == '__main__':
    start_time = time.time()
    best()
    print("spend time %ds." % (time.time() - start_time))
