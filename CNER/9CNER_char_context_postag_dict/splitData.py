# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: splitData.py
# @Time: 18/2/25 20:20
# @Desc:

with open('train.txt','a',encoding='utf-8') as fw1:
    with open('test.txt', 'a', encoding='utf-8') as fw2:
        with open('labeledSent.txt', 'r', encoding='utf-8') as fr:
            count = 1
            for line in fr:
                if line == '\n': count += 1
                print(count)
                if (count % 4) != 0:
                    fw1.write(line)
                else:
                    fw2.write(line)