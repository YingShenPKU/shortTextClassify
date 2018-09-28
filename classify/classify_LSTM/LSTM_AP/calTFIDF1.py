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
            corpus.append(line.strip())
print(corpus)
print('corpus len:',len(corpus))
# corpus = [' '.join(corpus)]


# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# # 获取词袋中所有文本关键词
# word = vectorizer.get_feature_names()
# print(word)
# # 查看词频结果
# print(X.toarray())


from sklearn.feature_extraction.text import TfidfTransformer
# 将词频矩阵X统计成TF-IDF值
tfidf = TfidfTransformer().fit_transform(X)
# 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
word=vectorizer.get_feature_names()
weight=tfidf.toarray()
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print ("-------这里输出第",i,u"类文本的词语tf-idf权重------" )
        for j in range(len(word)):
            if weight[i][j] != 0.0:
                print (word[j],weight[i][j])

#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

# for words in corpus:
#     for word in words:


