 # -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:18:41 2019

@author: Wang
"""
import pandas as pd
import numpy as np
from time import *
from Scorer import *
import nltk

# a sample of how to used the function
text = pd.read_table('D:\Simplified.txt')
text = np.array(text)
begin_time = time()
sen = nltk.word_tokenize("little blurred vision.")
s = dymatic_planing(sen,dic,text)
print(s)
print(time()-begin_time)
