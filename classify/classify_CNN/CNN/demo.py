# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: demo.py
# @Time: 2017/12/12 18:53
# @Desc:

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


W2V_DIR = 'C:\code\\bishe\word_embedding1'#vword vector 目录
MAX_SEQUENCE_LENGTH = 100#每条短文本最大长度
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200#词向量空间维度
VALIDATION_SPLIT = 0.2# 验证集比例

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(W2V_DIR, 'medCorpus.zh.vector'),'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

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

# finally, vectorize the text samples into a 2D integer tensor
#Tokenizer将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 4,  activation='relu')(embedded_sequences)
x = MaxPooling1D(4)(x)
x = Conv1D(128, 4, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(128, 4, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))