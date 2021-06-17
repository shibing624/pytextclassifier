# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier.textcluster import TextCluster

if __name__ == '__main__':
    m = TextCluster()
    print(m)
    data = [
        'Student debt to cost Britain billions within decades',
        'Chinese education for TV experiment',
        'Abbott government spends $8 million on higher education',
        'Middle East and Asia boost investment in top level sports',
        'Summit Series look launches HBO Canada sports doc series: Mudhar'
    ]
    model, X_vec, labels = m.train(data)
    r = m.predict(['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports'])
    print(r)
    m.show_clusters(X_vec, labels, image_file='cluster.png')
    m.save()
    del m

    new_m = TextCluster()
    new_m.load()
    r = new_m.predict(['Abbott government spends $8 million on higher education media blitz',
                       'Middle East and Asia boost investment in top level sports'])
    print(r)

    # load train data from file
    tc = TextCluster()
    data = tc.load_file_data('train_seg_sample.txt')
    _, X_vec, labels = tc.train(data)
    tc.show_clusters(X_vec, labels, 'cluster_train_seg_samples.png')
    r = tc.predict(data[:5])
    print(r)
