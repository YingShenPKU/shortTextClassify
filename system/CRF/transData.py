# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: transData.py
# @Time: 18/2/25 13:20
# @Desc:
#使用BIOES标注法，症状加S，如SB；疾病加D，如DB
#返回字符标签列表


#所有句子和标签写入文件
def writeFile(sent,posLabel,boundaryLabel):
    with open('CRF\labeledSent.txt','a',encoding='utf-8') as fw:
        for i in range(len(sent)):
            fw.write(sent[i]+' '+posLabel[i]+' '+boundaryLabel[i])
            fw.write('\n')
        fw.write('\n')

#获取四个文件夹下，每个文件的文件名，如“病史特点-1.txt”
import os
from CRF.features import *

def transData(sent):
        posLabel = speechOfTag(sent)
        boundaryLabel = boundary(sent)
        # print(sent,posLabel,boundaryLabel)
        writeFile(sent,posLabel,boundaryLabel)
        # exit(0)


#test
# sent = '轻微腹泻'
# transData(sent)