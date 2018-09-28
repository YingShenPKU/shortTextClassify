# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: AP.py
# @Time: 17/12/21 19:39
# @Desc:
# W2V = 'C:\code\\bishe\others_common\LSTM+Hierachy\data\GoogleNews-vectors-negative300.bin'
#####################################################################
#只对分级语料中出现的词做聚类
print('1.getting unique tokens...')
from keras.preprocessing.text import Tokenizer

texts = []  # list of text samples
with open('mergeData.txt','r',encoding='utf-8') as fr:
    for line in fr:
        texts.append(line.strip())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#####################################################################
#构建kmeans输入的词向量矩阵
print('2.getting input matrix...')
import gensim
import numpy as np

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('uniqueWords.vector', encoding='utf-8')
embedding_matrix = np.zeros((17052, 300))
not_in_model = 0
in_model = 0
inWords = []
i=0
#将待聚类的词和向量写入文件
fw1 = open('uniqueWords.vector','w',encoding='utf-8')
for word,a in word_index.items():
    fw1.write(word+' ')
    if word in w2v_model:
        in_model += 1
        inWords.append(word)
        vec = w2v_model[word].tolist()
        fw1.write(' '.join([str(ii) for ii in vec]))
        embedding_matrix[i] = np.asarray(vec, dtype='float32')
        i += 1
    else:
        # fw1.write(' '.join([0 for i in range(200)]))
        not_in_model += 1
    fw1.write('\n')
fw1.close()
print(str(in_model) + ' words in w2v model')
print (str(not_in_model)+' words not in w2v model')
print('input shape:',np.shape(embedding_matrix))
##########################################################################
#iter use kmeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

print('3.starting Hierachy cluster...')
print(embedding_matrix)
# 计算包不同类簇个数对应的聚类结果
prefs = [i for i in range(4100,5000,100)]
for pref in prefs:
    print('preference value:',pref)
    ap1 = AgglomerativeClustering(n_clusters=pref).fit_predict(embedding_matrix)
    ap = AgglomerativeClustering(n_clusters=pref).fit(embedding_matrix)
    # cluster_centers_indices = ap.core_sample_indices_ # 预测出的中心点的索引，如[123,23,34]
    labels = ap.labels_  # 预测出的每个数据的类别标签,labels是一个NumPy数组

    res = {}
    num = 0
    for label in labels:
        if label in res.keys():
            res[label].append(num)
        else:
            res[label] = [num]
        num += 1

    newDict = sorted(res.items(), key=lambda x: len(x[1]), reverse=True)

    print('wirte file:','syns_'+str(pref)+'.txt')
    fw = open('clusterResult2\syns_' + str(pref) + '.txt', 'w', encoding='utf-8')

    for key, value in dict(newDict).items():
            fw.write(inWords[key] + ':')
            fw.write(' '.join([inWords[i] for i in value]))
            fw.write('\n')
        # print('-------------------')
    fw.close()

    # cal_har_sco = metrics.calinski_harabaz_score(embedding_matrix, ap1)
    # print('cal_har_sco:', cal_har_sco)
    # silhouette = metrics.silhouette_score(embedding_matrix, labels, metric='euclidean')
    # print('silhouette:',silhouette)
    # print('-------------------')
