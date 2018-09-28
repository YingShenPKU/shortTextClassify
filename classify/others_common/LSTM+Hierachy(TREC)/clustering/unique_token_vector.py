# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: unique_token_vector.py
# @Time: 18/1/4 11:57
# @Desc:
W2V = 'C:\code\\bishe\others_common\LSTM+Hierachy\data\GoogleNews-vectors-negative300.bin'
#####################################################################
#只对分级语料中出现的词做聚类
print('2.getting unique tokens...')
from keras.preprocessing.text import Tokenizer
import os

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
print('3.getting input matrix...')
import gensim
import numpy as np

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(W2V, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, 300))
not_in_model = 0
in_model = 0
#将待聚类的词和向量写入文件
fw1 = open('uniqueWords.vector','w',encoding='utf-8')
for word, i in word_index.items():

    if word in w2v_model:
        fw1.write(word + ' ')
        in_model += 1
        vec = w2v_model[word].tolist()
        fw1.write(' '.join([str(i) for i in vec]))
        embedding_matrix[i] = np.asarray(vec, dtype='float32')
        fw1.write('\n')
    else:
        # fw1.write(' '.join([0 for i in range(200)]))
        not_in_model += 1

fw1.close()
print(str(in_model) + ' words in w2v model')
print (str(not_in_model)+' words not in w2v model')
print('input shape:',np.shape(embedding_matrix))