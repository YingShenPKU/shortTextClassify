# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: selectPara.py
# @Time: 18/3/21 23:27
# @Desc:

import os

#进入CRF目录
dir = "C:\code\\bishe\pre_NER\CCKS2017\CNER\\10CNER_char_context_postag_boundary_dict\CRF"
os.system("cd "+dir)

#从输出结果，匹配特征数,迭代次数，训练时间
import re
def extractInfo(res):
    mfea = re.findall("(?<=features:).*?(?=\n)",res)[0]
    # miters = re.find(,res)
    mtime = re.findall("(?<=Done!).*?(?= s)",res)[-1]
    # print(mfea,mtime)
    return mfea,mtime


#自动调参，训练模型，测试数据，评估结果
from evaluate1 import *
from time import sleep
# crf_learn.exe -f 3 -c 1.5 template train.txt model
# crf_test.exe -m model test.txt >> result.txt
with open("paraResult.txt",'w',encoding='utf-8') as fw:
    for f in range(1,5):
        for c10 in range(5,30,3):
            c = c10*0.1
            print("crf_learn.exe -f %d -c %f template train.txt model"%(f,c))
            # 删除原来的result.txt和model文件
            os.system("del " + dir + "\\result.txt")
            os.system("del " + dir + "\\model")
            #训练，要用绝对目录
            r = os.popen(dir+"\crf_learn.exe -f %d -c %f %s\\template %s\\train.txt %s\model"%(f,c,dir,dir,dir))
            res = r.readlines()
            mfea,mtime = extractInfo(''.join(res))
            #测试
            os.popen(dir + "\crf_test.exe -m %s\\model %s\\test.txt >> %s\\result.txt" % (dir, dir, dir))
            #调用evaluate1文件，评估性能
            sleep(2)
            p,r,f1 = eval(dir+"\\result.txt")
            #将过程和结果保存再parResult.txt文件
            #f,c,mfea,mtime,p,r,f1
            print(f,c,mfea,mtime,p,r,f1)
            #将结果写入文件
            fw.writelines([str(f),str(c),mfea,mtime,str(p),str(r),str(f1)])
            fw.write('\n')


#输出最优结果
with open("paraResult.txt",'r',encoding='utf-8') as fr:
    f1s = 0
    pline = ''
    for line in fr:
        if len(line)==0:continue
        line1 = line.strip().split()
        # print(line1[-1])
        f1 = float(line1[-1])
        if f1 > f1s :
            f1s, pline = f1, line
        else:
            pass

    print("best result:"+pline)