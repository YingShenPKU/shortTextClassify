# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: ontologyInfo.py
# @Time: 18/4/10 8:55
# @Desc:


def ontologyInfo(entity):
    #提取所有疾病本体列表
    diseases = {}
    with open('diseasesOntology.txt','r',encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split(',')
            disease = line[0].split(':')[1]
            diseases[disease]=line
    # print(diseases)

    #查看实体是否在疾病本体列表中
    info = ''
    if entity in diseases:
        info = diseases[entity]

    # print(info)
    return info


ontologyInfo('心脏病')