# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""


def trim_pos(line):
    data = ''
    try:
        parts = line.split('\t', 1)
        content = parts[1]
        content_split = [w.split('/')[0] for w in content.split(' ')]
        data = ' '.join(content_split)
    except Exception as e:
        print('err', e, line)
    return data


if __name__ == "__main__":
    seg_data = []
    with open('training_new_seg.txt', 'r', encoding='utf-8') as f:
        for line in f:
            seg_data.append(trim_pos(line))

    with open('testing_seg.txt', 'r', encoding='utf-8') as f:
        for line in f:
            seg_data.append(trim_pos(line))

    with open('all_segmented.txt', 'w', encoding='utf-8') as f:
        for line in seg_data:
            f.write(line + '\n')
