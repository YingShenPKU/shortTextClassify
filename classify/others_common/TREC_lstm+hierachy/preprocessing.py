# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: preprocessing.py
# @Time: 18/1/9 20:07
# @Desc:

#按照标签拆分到不同的文件中
def splitDataByLabel():
    with open('training.txt','r',encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split(':')
            label = line[0]
            sent = line[1]
            # print('label:',label)
            # print('sent:',' '.join(sent))
            fw = open('corpus/'+label+'.txt','a',encoding='utf-8')
            fw.write(sent)
            fw.write('\n')

#run
splitDataByLabel()