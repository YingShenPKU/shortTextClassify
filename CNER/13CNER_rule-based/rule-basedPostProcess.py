# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: rule-basedPostProcess.py
# @Time: 18/4/18 9:25
# @Desc:


# 1.生理部位 + 症状 = 症状，如“心肌梗塞”；
# 2.生理部位 + 检查 = 症状，如“尿量正常”；
# 3.否定词 + 症状 = 症状，如“无恶心”；
# 4.否定词 + 生理部位 + 症状 = 症状，如“无扁桃体肿大”；
# 5.否定词 + 疾病 = 疾病，如“无肺炎”；

#将疾病症状列提取为列表ds,身体部位检查列提取为tb
ds,tb = [],[]
with open('result_merged.txt','r',encoding='utf-8') as fr:
        for line in fr:
                line = line.strip().split()
                if (len(line)) > 0:
                    tb.append(line[-1])
                    ds.append(line[-2])

#使用规则1后处理
print(len(tb),len(ds))
temp = []
finalLabel = []
flag = False
num = 0
for i in range(len(ds)):
    finalLabel.append(ds[i])
    if tb[i] == 'BB':
        temp.append('SB')
        num += 1
    if tb[i] == 'BI':
        temp.append('SI')
        num += 1
    if tb[i] == 'BE' and ds[i+1] =='SB':
        temp.append('SI')
        flag = True
        num += 1
    else:
        temp = []
        flag = False
    if flag and ds[i]=='SI':
        temp.append('SI')
        num += 1
    if flag and ds[i] == 'SE':
        temp.append('SE')
        num += 1
        flag = False
        finalLabel = finalLabel[:-num]
        finalLabel.extend(temp)
print(len(finalLabel))
print(finalLabel)

#将finalLable写入文件result.txt
with open('result.txt','w',encoding='utf-8') as fw:
    with open('result_merged.txt', 'r', encoding='utf-8') as fr:
        count = 0
        for line in fr:
            line = line.strip().split()
            line.append(finalLabel[count])
            fw.write('\t'.join(line))
            fw.write('\n')
            count += 1