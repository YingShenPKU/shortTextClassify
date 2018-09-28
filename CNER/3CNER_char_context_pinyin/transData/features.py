#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @Author: zhangqiang2015@pku.edu.cn
# @Date  : 2017/5/16
# @Desc  :
import jieba
import jieba.posseg as pseg

#标注词性
def speechOfTag(text):
    # #分词
    # seg_list = jieba.cut(text)
    # print "Segment:", "/ ".join(seg_list)
    #词性标注
    words = pseg.cut(text)
    possegs = []
    for w in words:
        # print w.word,w.flag
        # if w.word.isdigit():
        #     possegs.append(u"m")
        # else:
        possegs.extend([w.flag for i in w.word])
    print (possegs)

    return possegs

from xpinyin import Pinyin
def pinyin(text):
    p = Pinyin()
    pinyins = []
    for i in text:
        # print i
        pinyin = p.get_pinyin(i)
        pinyins.append(pinyin)
    print (pinyins)

    return pinyins

#标注边界
def boundary(text):
    #Tokenize：返回词语在原文的起始位置
    result = jieba.tokenize(text)
    for tk in result:
        print     ("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))

#DEBUG------------------------------------
str1 = u"1个21岁未婚手指中指断了算几级伤残"
pinyin(str1)