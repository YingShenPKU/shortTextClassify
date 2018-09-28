# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: transData.py
# @Time: 18/2/25 13:20
# @Desc:

def readSent(name):
    with open(name+'original.txt','r',encoding='utf-8') as fr1:
        sent = fr1.readlines()[0].strip()
        print(sent)
    return sent


#使用BIOES标注法，症状加S，如SB；疾病加D，如DB
#返回字符标签列表
def LabelSent(sent,name):
    #初始化字符标签chaLabel为O
    charLabel = ['O' for i in range(len(sent))]
    #对标签为'症状和体征'和'疾病和诊断'的实体进行标注
    with open(name,'r',encoding='utf-8') as fr:
        print(name)
        for line in fr:
            if len(line) == 0: break#有些句子无实体
            label = line.split()[-1]
            L = int(line.split()[-3])
            R = int(line.split()[-2])
            if label == '症状和体征':
                if L == R: charLabel[L] = 'SS'
                else:
                    charLabel[L] = 'SB'
                    charLabel[R] = 'SE'
                    if L < R-1:
                        for i in range(L+1,R):
                            charLabel[i] = 'SI'
            if label == '疾病和诊断':
                if L == R: charLabel[L] = 'DS'
                else:
                    charLabel[L] = 'DB'
                    charLabel[R] = 'DE'
                    if L < R-1:
                        for i in range(L+1,R):
                            charLabel[i] = 'DI'
    #返回字符串sent和字符串charLabel
    print(sent)
    print(charLabel)
    return charLabel

#所有句子和标签写入文件
def writeFile(sent,posLabel,pinyins,boundaryLabel,charLabel):
    with open('labeledSent.txt','a',encoding='utf-8') as fw:
        for i in range(len(sent)):
            fw.write(sent[i]+' '+posLabel[i]+' '+pinyins[i]+' '+boundaryLabel[i]+' '+charLabel[i])
            fw.write('\n')
        fw.write('\n')

#获取四个文件夹下，每个文件的文件名，如“病史特点-1.txt”
import os
from features import *

for file in os.listdir('data'):
    print(file)
    for i in range(1,101):
        name = 'data/'+file+'/'+file+'-%d.txt'%i
        #获取句子，字符标签列表
        sent = readSent(name)
        posLabel = speechOfTag(sent)
        boundaryLabel = boundary(sent)
        pinyins = pinyin(sent)
        # print(boundaryLabel)
        charLabel = LabelSent(sent,name)
        writeFile(sent,posLabel,pinyins,boundaryLabel,charLabel)
        # exit(0)
