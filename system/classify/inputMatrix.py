# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: inputMatrix.py
# @Time: 18/4/9 15:48
# @Desc:

###################################################################
# second, prepare text samples and their labels
# print ('(1) load texts...')
# import os
# texts = []  # list of text samples
# #s1,s2,s3对应'轻度','中度','重度'
# labels_index={'s1':1,'s2':2,'s3':3}# labels_index = {}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids
# for file in os.listdir('data'):
#     label = file.split('_')[0]
#     with open('data/'+file,'r',encoding='utf-8') as fr:
#         for line in fr.readlines():
#             texts.append(line)
#             labels.append(labels_index[label])
#
# print('Found %s texts.' % len(texts))
# ###################################################################
#对数据语句进行分词
def classify(inputtext):
    import jieba
    # inputtext = '鼻塞明显，需用口呼吸'
    
    text0 = jieba.cut(inputtext)

    stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '\n', '《', '》', ' ', '-', '！', '？', '.', '\'', '[', ']', '：',
                 '/', '.', '"', '\u3000', '’', '．', ',', '…', '?']
    text = []
    for i in text0:
        if i not in stop_word:
            text.append(i)
    # text = [' '.join(text)]
    # print(text)
    # ###################################################################
    # from keras.preprocessing.text import Tokenizer
    #
    # #获取训练语料中所有词及其索引
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    # sequences = tokenizer.texts_to_sequences(texts)
    # word_index = tokenizer.word_index
    #
    # #保存词索引
    # f = open('word_index.txt','w',encoding='utf-8')
    # f.write(str(word_index))
    # f.close()
    #读取词索引
    f = open('classify\word_index.txt','r',encoding='utf-8')
    a = f.read()
    word_index = eval(a)
    f.close()
    ###################################################################
    #加载模型
    from keras.models import load_model
    import numpy as np
    model = load_model('classify\word_vector_lstm.h5')
    # testArray = sequence.pad_sequences(embedding_matrix, maxlen=MAX_SEQUENCE_LENGTH)
    # print(testArray)
    if len(text)<10:
        t_data = [0 for i in range(10-len(text))]
        for t in text:
            #长度小于10，前面补0
            if t in word_index.keys():
                t_data.append(word_index[t])
            else:t_data.append(0)
    if len(text)>=10:
        for t in text:
            #长度小于10，前面补0
            if t in word_index.keys() :
                if len(t_data)<10: t_data.append(word_index[t])
                else: break
            else:
                if len(t_data)<10: t_data.append(0)
                else: break
    # print(t_data)
    test_data = [t_data]
    pred = model.predict(np.asarray(test_data))#词序号
    # print(pred)
    #得到对应类别的概率
    prediction = pred.tolist()[0]
    res = {}
    res['轻微'] = round(float(prediction[1]),2)
    res['中等'] = round(float(prediction[2]),2)
    res['严重'] = round(float(prediction[3]),2)
    print('语义量化：',str(res))
    return res