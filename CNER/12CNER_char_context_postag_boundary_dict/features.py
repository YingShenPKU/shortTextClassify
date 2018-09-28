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
    #加载用户词典
    diseaseDict = [d.strip() for d in open('dict/diseaseDict1.txt','r',encoding='utf-8').readlines()]
    # print(len(diseaseDict))
    symptomDict = [s.strip() for s in open('dict/symptomDict1.txt', 'r', encoding='utf-8').readlines()]
    dictTag = []
    #带词典的分词模式
    jieba.load_userdict('dict\\allDict1.txt')
    result = jieba.cut(text)
    # print (result)
    for tk in result:
        # print (tk)
        if tk in diseaseDict:
            dictTag.extend(['DB'])
            dictTag.extend(['DI' for i in tk[1:]])
        elif tk in symptomDict:
            dictTag.extend(['SB'])
            dictTag.extend(['SI' for i in tk[1:]])
        else:
            dictTag.extend(['N' for i in tk])
    # print (dictTag)

    return dictTag

#DEBUG------------------------------------
# str1 = u"1个21岁未婚鼻塞，手指中指断了算几级伤残"
# useDict(str1)