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


#DEBUG------------------------------------
# str1 = u"1个21岁未婚手指中指断了算几级伤残"
# speechOfTag(str1)