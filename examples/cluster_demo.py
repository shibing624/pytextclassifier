# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier.textcluster import TextCluster

if __name__ == '__main__':
    m = TextCluster(n_clusters=2)
    print(m)
    data = [
        'Student debt to cost Britain billions within decades',
        'Chinese education for TV experiment',
        'Abbott government spends $8 million on higher education',
        'Middle East and Asia boost investment in top level sports',
        'Summit Series look launches HBO Canada sports doc series: Mudhar'
    ]
    X_vec, labels = m.train(data)
    r = m.predict(['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports'])
    print(r)
    m.show_clusters(X_vec, labels, image_file='cluster.png')
    del m

    new_m = TextCluster(n_clusters=2)
    new_m.load_model()
    r = new_m.predict(['Abbott government spends $8 million on higher education media blitz',
                       'Middle East and Asia boost investment in top level sports'])
    print(r)

    ########### load chinese train data from 10w data file
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2))
    tcluster = TextCluster(vectorizer=vec, n_clusters=10)
    data = tcluster.load_file_data('thucnews_train_10w.txt', sep='\t', use_col=1)
    X_vec, labels = tcluster.train(data[:5000])
    tcluster.show_clusters(X_vec, labels, 'cluster_train_seg_samples.png')
    r = tcluster.predict(data[:30])
    print(r)
