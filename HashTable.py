# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:28:08 2019

@author: Wang
"""

import numpy as np
import pandas as pd
import re
from time import *
#transform the txt into a dictionary
dic = open('D:\Medhelp_100d.txt','r',encoding='UTF-8').read()
dic = dic.split('\n')
for i in range(len(dic)):
    dic[i] = dic[i].split(' ')
name = [x[0] for x in dic]
vec = [x[1:101] for x in dic]
dic = dict(zip(name,vec))