#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @Author: zhangqiang2015@pku.edu.cn
# @Date  : 2017/5/16
# @Desc  :
import jieba
import jieba.posseg as pseg

#标注词性
def speechOfTag(text):
    #词性标注
    words = pseg.cut(text)
    possegs = []
    for w in words:
        possegs.extend([w.flag for i in w.word])
    # print (possegs)

    return possegs


#标注边界,使用BIES标注
def boundary(text):
    #Tokenize：返回词语在原文的起始位置
    result = jieba.tokenize(text)
    boundarys = []
    for tk in result:
        # print     ("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
        if tk[1]+1 == tk[2]:#单个成词的字
            boundarys.append(u'S')
        elif tk[1]+2 == tk[2]:#单个成词的字
            boundarys.append(u'B')
            boundarys.append(u'E')
        else:
            boundarys.append(u'B')
            for i in range(tk[1]+1, tk[2]-1):
                boundarys.append(u'I')
            boundarys.append(u'E')
    # print (boundarys)

    return boundarys

#判断句子中的词汇是否在字典中，并进行标注
import pandas as pd
import numpy as np

def useDict(text):
    data = pd.read_excel('chpo.2016-10.xls',skiprows=0)['名称(中文)']
    train_data = np.array(data)  # np.ndarray()
    dicts = train_data.tolist()  # list
    # print(train_x_list)
    # print(data)

    dictTag = []
    result = jieba.tokenize(text)
    for tk in result:
        if tk[0] in dicts:
            dictTag.extend(['Y' for i in tk[0]])
        else:
            dictTag.extend(['N' for i in tk[0]])
    # print (dictTag)

    return dictTag

#DEBUG------------------------------------
# str1 = u"1个21岁未婚手指中指断了算几级伤残"
# useDict(str1)