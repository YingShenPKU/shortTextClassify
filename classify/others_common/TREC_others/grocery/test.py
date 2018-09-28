# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: test.py
# @Time: 18/1/3 10:22
# @Desc:


from tgrocery import Grocery

grocery = Grocery('CommonTest')
grocery.train('training.txt',delimiter=':')

res = grocery.test('testing.txt',delimiter=':')

print res