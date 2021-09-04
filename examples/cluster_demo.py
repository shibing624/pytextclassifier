# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pytextclassifier.textcluster import TextCluster

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
m.save()
del m

new_m = TextCluster(n_clusters=2)
new_m.load()
r = new_m.predict(['Abbott government spends $8 million on higher education media blitz',
                   'Middle East and Asia boost investment in top level sports'])
print(r)

########### load chinese train data from file
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=0.1, sublinear_tf=True)
tcluster = TextCluster(vectorizer=vec, n_clusters=3)
data = tcluster.load_file_data('train_seg_sample.txt')
X_vec, labels = tcluster.train(data)
tcluster.show_clusters(X_vec, labels, 'cluster_train_seg_samples.png')
r = tcluster.predict(data[:5])
print(r)
