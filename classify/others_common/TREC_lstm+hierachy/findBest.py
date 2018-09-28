# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: findBest.py
# @Time: 17/12/26 20:01
# @Desc:

from lstm_Hierachy import lstm_fun
import os

with open('result.txt','w',encoding='utf-8') as fw:
    for file in os.listdir('clusterResult'):
        file = 'clusterResult/'+file
        res = lstm_fun(file)
        fw.write(file)
        fw.write(str(res))
        fw.write('\n')
