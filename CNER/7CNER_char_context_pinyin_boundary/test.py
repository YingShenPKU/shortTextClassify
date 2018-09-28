# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: test.py
# @Time: 18/3/22 20:04
# @Desc:

#输出最优结果
with open("paraResult.txt",'r',encoding='utf-8') as fr:
    f1s = 0
    pline = ''
    for line in fr:
        line1 = line.strip().split()
        # print(line1[-1])
        f1 = float(line1[-1])
        if f1 > f1s :
            f1s, pline = f1, line
        else:
            pass

    print("best result:"+pline)