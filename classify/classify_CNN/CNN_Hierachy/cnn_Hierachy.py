# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: cnn_ap3.py
# @Time: 17/12/25 16:14
# @Desc:

import os
import numpy as np
import gensim

MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2
###################################################################
center_cluster = {}
centers = []
clusters = []
f1 = open( 'syns_1200.txt','r',encoding='utf-8')
for line in f1:
    values = line.split(':')
    centers.append(values[0])
    clusters.append(values[1])
f1.close()

#使用聚类中心向量表示
def findAP(word):
    vect = []
    for i in range(len(clusters)):
        if word in clusters[i]:
            vect.append(embeddings_index[centers[i]].tolist())
    if len(vect)==0:
            vect.append(embeddings_index[word].tolist())
    # print('word:',word)
    # print('vector:',vect)
    return vect[0]


###################################################################
# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')
embeddings_index = {}
f = open( 'medCorpus.zh.vector','r',encoding='utf-8')
for line in f:
    values = line.split()
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
labels_index={'s1':1,'s2':2,'s3':3}# labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for file in os.listdir('data'):
    label = file.split('_')[0]
    with open('data/'+file,'r',encoding='utf-8') as fr:
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
print ('(4) load word2vec as embedding...')
from gensim.models.keyedvectors import KeyedVectors
w2v_model0 = KeyedVectors.load_word2vec_format('medCorpus.zh.vector', encoding='utf-8')
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
from keras.layers import Dense, Input, GlobalMaxPooling1D,Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,BatchNormalization
from keras.models import Model
print('(5) Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 4, input_shape = (10,400), activation='relu')(embedded_sequences)
# x = MaxPooling1D()(x)
x = BatchNormalization()(x)
x = Conv1D(128, 4, activation='relu')(x)
x = BatchNormalization()(x)
# x = MaxPooling1D(3)(x)
# x = Conv1D(64, 3,activation='relu')(x)
x = GlobalMaxPooling1D()(x)
# x = MaxPooling1D()(x)
# x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(64,  activation='relu')(x)
preds = Dense(4, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print (model.metrics_names)

model.fit(x_train, y_train,
          batch_size=128,
          epochs=100,
          validation_data=(x_val, y_val))

print ('(6) testing model...')
print (model.evaluate(x_test, y_test))