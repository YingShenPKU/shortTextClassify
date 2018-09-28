# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Author: zhangqiang1104@pku.edu.cn
# @File: test.py
# @Time: 17/12/14 20:53
# @Desc:多线程同步


import numpy as np
import threading

class myThread(threading.Thread):
    def __init__(self,threadID,i):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.i = i

    def run(self):
        print ("start thread:",self.threadID)
        threadLock.acquire()
        for t in range(10):
            a[self.i + t*5] = (self.i) * np.ones(10)
        threadLock.release()

a = np.zeros((50,10))
threadLock = threading.Lock()
threads = []
# 创建新线程
for i in range(5):
    thread_i = myThread(i,i)
    # 开启新线程
    thread_i.start()
    # 添加线程到线程列表
    threads.append(thread_i)

# 等待所有线程完成
for t in threads:
    t.join()
print ("Exiting Main Thread")
print (a)
