# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:57:19 2019

@author: Wang
"""

from annoy import AnnoyIndex
import random
import pandas as pd
import re
import numpy as np

file = open('D:\\NewRefer_.txt','r')    
f = 100 # Length of item vector that will be indexed
chunk = int(10406793/10)
print("start")
for i in range(10):
    t = AnnoyIndex(f, 'angular') 
    for g in range(chunk):
            tem = file.readline()
            vec = np.array(re.split(', |]',tem[1:])[:-1]).astype('float')
#    print(i)
    t.build(50) # 50 trees
    t.save('D:/distance__'+str(i))
print('success')