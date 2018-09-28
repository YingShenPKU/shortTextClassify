# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: calTFIDF.py
# @Time: 17/12/25 10:21
# @Desc:

from sklearn.feature_extraction.text import CountVectorizer
import os
# 语料
corpus = []
files = os.listdir('data')
for file in files:
    with open('data/'+file,'r',encoding='utf-8') as fr:
        for line in fr:
            corpus.extend(line.strip().split())
print(corpus)
print('corpus len:',len(corpus))
corpus = [' '.join(corpus)]
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)

# 查看词频结果
print(X.toarray())


# from sklearn.feature_extraction.text import TfidfTransformer
# # 将词频矩阵X统计成TF-IDF值
# tfidf = TfidfTransformer().fit_transform(X)
# # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
# print(tfidf.toarray())

def findTF(word1):
    try:
        i = word.index(word1)
        res = X.toarray()[0][i]
    except:
        res = 0

    return res
