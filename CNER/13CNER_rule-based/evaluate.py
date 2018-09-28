# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: evaluate.py
# @Time: 18/2/25 20:44
# @Desc:

with open('result.txt','r',encoding='utf-8') as fr:
    realList = []
    predList = []
    for line in fr:
        if line == '\n':
            continue
        else:
            line = line.split()
            real,pred = line[1],line[-1]
            realList.append(real)
            predList.append(pred)
    # print(len(realList))
    # print(len(predList))

    tp=fn=fp=0
    startLabel = False
    # realStr = predStr = ''
    for i in range(len(realList)):
        if realList[i] == predList[i] == 'SS': tp += 1
        if realList[i] in ['SB','DB'] and realList[i] != predList[i]:
            fn += 1
        if realList[i] == 'O' and predList[i] in ['SB','DB','SS']:
            fp += 1
        if realList[i] in ['SB','DB'] and realList[i] == predList[i]:
            startLabel = True
        if startLabel :
            if predList[i] == realList[i]:
                if realList[i] in ['SE', 'DE'] :
                    tp += 1
                    startLabel = False
            # realStr = predStr = ''


    P = float(tp)/(tp+fp)
    R = float(tp) / (tp + fn)
    F1 = 2*P*R/(P+R)
    # print(tp,fn,fp)
    print('P:%s'%(P))
    print('R:%s' % (R))
    print('F1:%s'%F1)
