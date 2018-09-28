# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: test.py
# @Time: 18/1/4 15:18
# @Desc:

from textblob import TextBlob

blob = TextBlob('I was working at home')
print (blob.words)
for word in blob.words:
    print (word.lemmatize())
