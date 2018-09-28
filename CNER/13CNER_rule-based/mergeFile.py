# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: rule-basedPostProcess.py
# @Time: 18/4/18 8:59
# @Desc:


#将result_TB.txt最后一列加入result_DS.txt后面
#保存到文件result_merged.txt
predLabel = []
with open('result_TB.txt','r',encoding='utf-8') as fr:
    for line in fr:
            line = line.strip().split()
            if (len(line)) > 0:
                # print(line)
                predLabel.append(line[-1])
print(predLabel)

num = 0
with open('result_merged.txt','w',encoding='utf-8') as fw:
    with open('result_DS.txt','r',encoding='utf-8') as fr:
        for line in fr:
                line = line.strip().split()
                if (len(line)) > 0:
                    line.append(predLabel[num])
                    num += 1
                    print(line)
                    fw.write('\t'.join(line))
                    fw.write('\n')