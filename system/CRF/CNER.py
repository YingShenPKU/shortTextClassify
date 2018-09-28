# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: CNER.py
# @Time: 18/4/9 15:06
# @Desc:
import os,time
from CRF.transData import transData

def CNER(sent):
    #删除上一次测试生成的结果文件
    dir = "C:\code\\bishe\system\CRF"
    os.system("cd "+dir)
    os.system("del " + dir + "\\labeledSent.txt")
    os.system("del " + dir + "\\result.txt")

    #输入句子，提取特征，放到文件'labeledSent.txt'

    transData(sent)
    # time.sleep(10)
    #加载CRF模型，预测标签
    os.popen(dir + "\crf_test.exe -m %s\\model %s\\labeledSent.txt >> %s\\result.txt" % (dir, dir, dir))
    time.sleep(1)
    #根据标签提取实体
    diseases = []
    # symptoms = []
    #加载疾病和症状词典
    disDict = [i.strip() for i in open('CRF\\allDict1.txt','r',encoding='utf-8')]
    # symDict = [i.strip() for i in open('CRF\\symptomDict1.txt', 'r', encoding='utf-8')]
    # print(disDict)
    for i in disDict:
        if i in sent:diseases.append(i)
    # for i in symDict:
    #     if i in sent:symptoms.append(i)

    with open('CRF\\result.txt','r',encoding='utf-8') as fr:

        # symptom=''
        disease=''
        for line in fr:
            # print(line)
            cols = line.strip().split()
            if len(cols) == 0: break
            # print(cols)
            char = cols[0]
            label = cols[-1]

            if label in ['SB','SI','DB','DI']: disease += char
            elif label=='DE':
                disease += char
                diseases.append(disease)
                disease=''
        print('医学实体：',' '.join(set(diseases)))
        # print('symptoms:',' '.join(symptoms))
        return set(diseases)


#test
# sent = '摔伤后疼痛肿胀'
# diseases,symptoms = CNER(sent)