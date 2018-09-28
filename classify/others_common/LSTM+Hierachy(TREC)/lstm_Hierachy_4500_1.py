# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: lstm_AP1-450.py
# @Time: 17/12/25 9:36
# @Desc:

import os
import numpy as np
import gensim

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 600
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2

###################################################################
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')
embeddings_index = {}
f = open('uniqueWords.vector','r',encoding='utf-8')
for line in f:
    values = line.split()
    if len(values)<300:continue
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

###################################################################
# second, prepare text samples and their labels
print ('(1) load texts...')

texts = []  # list of text samples
#s1,s2,s3对应'轻度','中度','重度'
# labels_index={'s1':1,'s2':2,'s3':3}# labels_index = {}  # dictionary mapping label name to numeric id
labels_index = {"business":1, "computers":2, "culture-arts-entertainment":3,
                "sports":4,"education-science":5, "engineering":6,
                "health":7, "politics-society":8}

labels = []  # list of label ids
for file in os.listdir('corpus'):
    label = file.split('.')[0]
    with open('corpus/'+file,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
            texts.append(line)
            labels.append(labels_index[label])

print('Found %s texts.' % len(texts))

###################################################################
print ('(2) doc to var...')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

###################################################################
print ('(3) split data set...')
#shuffle the indice of data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# split the data into training set, validation set, and test set
p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print ('train docs: '+str(len(x_train)))
print ('val docs: '+str(len(x_val)))
print ('test docs: '+str(len(x_test)))
###################################################################

def lstm_fun(file):
    center_cluster = {}
    centers = []
    clusters = []
    f1 = open( file,'r',encoding='utf-8')
    for line in f1:
        values = line.split(':')
        centers.append(values[0])
        clusters.append(values[1])
    f1.close()

    #使用类簇所有词的均值向量表示聚类中心
    def findAP(word):
        vec = []
        for i in range(len(clusters)):
            if word in clusters[i]:
                # print("clusters[i]:",clusters[i].split())
                for wd in clusters[i].split():
                    vec.append(embeddings_index[wd].tolist())
                    # print(vec)
                # print('vec:',vec)
                vec = np.array(vec)
                vec = vec.mean(axis=0).tolist()  # mean value of columns
                # print('word:', word)
                # print('mean vector:', vec)
                # print('------------')
                break

        if len(vec)==0:
                vec = embeddings_index[word].tolist()

        return vec

    ###################################################################
    print ('(4) load word2vec as embedding...')
    from gensim.models.keyedvectors import KeyedVectors
    w2v_model0 = KeyedVectors.load_word2vec_format('uniqueWords.vector', encoding='utf-8')
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    not_in_model = 0
    in_model = 0

    #the origin word vector
    for word, i in word_index.items():
        if word in w2v_model0:
            in_model += 1
            vec0 = w2v_model0[word].tolist()
            vec1 = findAP(word)
            embedding_matrix[i] = np.asarray(vec0+vec1, dtype='float32')
        else:
            not_in_model += 1

    print (str(not_in_model)+' words not in w2v model')

    from keras.layers import Embedding
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    ###################################################################
    print ('(5) training model...')
    from keras.layers import Dense,  Dropout,BatchNormalization
    from keras.layers import LSTM
    from keras.models import Sequential

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    # exit(0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    print (model.metrics_names)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
    # model.save('word_vector_lstm.h5')
    ###################################################################
    print ('(6) testing model...')
    finalRes = model.evaluate(x_test, y_test)
    print (finalRes)

    return finalRes[1]

#run
res = lstm_fun('clusterResult0/syns_4500.txt')