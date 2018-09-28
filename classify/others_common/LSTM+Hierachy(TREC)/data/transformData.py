# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: transformData.py
# @Time: 18/1/4 10:06
# @Desc:将训练和测试数据整合到一个文件，按照标签分到不同的文件

def mergeData(file):
    fw = open('data-web-snippets/mergeData.txt','a',encoding='utf-8')
    with open('data-web-snippets/'+file,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
            fw.write(line)
    fw.close()


#run
# mergeData('train.txt')
# mergeData('test.txt')
# labels = ["business", "computers", "culture-arts-entertainment","sports",
#           "education-science", "engineering", "health", "politics-society"]



def splitDataByLabel():
    with open('data-web-snippets/mergeData.txt','r',encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split()
            label = line[-1]
            sent = line[0:-1]
            # print('label:',label)
            # print('sent:',' '.join(sent))
            fw = open('data-web-snippets/'+label+'.txt','a',encoding='utf-8')
            fw.write(' '.join(sent))
            fw.write('\n')

#run
splitDataByLabel()
