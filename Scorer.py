# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:42:40 2019

@author: Wang
"""

import numpy as np
import nltk
from time import *
import re
import pandas as pd
from annoy import AnnoyIndex
import random
#from HashTable import *


def centroid(tokens,dic):
#   calculate the centroid of each phrase,return a vector
    vec = []
    fin_vec = []
    tokens = nltk.word_tokenize(str(tokens))
    for words in tokens:
#        print(words)
        try:
#            print(words)
#            word = nltk.word_tokenize(words)
#            print(words)
#            for w in words:
#                print(w)
            vec.append(dic[words.lower()])
        except:
            vec.append([0 for i in range(100)])
    for g in range(len(vec[0])):
        cen = 0
        for i in range(len(vec)):
            cen += float(vec[i][g])
        fin_vec.append(round(cen/len(vec),8))
    return fin_vec
def cos_sim(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return 0 ;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*float(y[i])   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=float(y[i])**2     #sum(Y*Y)
    if result1 == 0:
        return 0 
    else:
        return result1/((result2*result3)**0.5) #结果
#def score(x,threshold,text):
#    file = open('D:\Refer.txt','r')
#    i = 0
#    g = 0
#    res = []
#    tem = file.readline()
#    vec = re.split(', |]',tem[1:])[:-1]
#    while tem:
##        print(x,vec)
##        print(cos_sim(x,vec))
#        grade = cos_sim(x,vec) 
#        if grade>threshold:
#            g+=1
#            res.append((text[i],grade))
#            print(g)
#        i+=1
#        print(i)
#        tem = file.readline()
#        vec = re.split(', |]',tem[1:])[:-1]
##        vec = list(map(float,tem[1:499].split(', ')))
#    file.close()
#    return res
#def score(x,threshold,text):
#    chunktable = pd.read_table('D:\Refer.txt',chunksize = 5000)
#    i = 0
#    g = 0
#    res = []
#    for chunk in chunktable:
#        arr = np.array(chunk)
#        for line in arr:
#            vec = re.split(', |]',line[0][1:])[:-1]
#            grade = cos_sim(x,vec) 
#            if grade>threshold:
#                res.append((text[i],grade))
#        i+=1
#        print(i)
#    return res
def score(x,concept,dic,refer):
    i = 0
    res = []
    try:
        mark = dic["['"+x+"']"]
    except:
        return 0 
    return cos_sim(centroid(x,refer),centroid(concept,refer))
def segment(tokens,start,end):
    res = str("")
    for i in range(start,end):
        res = res+tokens[i]+" "
    res = res+tokens[end]
    fin = (tokens[:start])
    fin.append(res)
    fin = fin+tokens[end+1:]
    return res,fin
#def dymatic_planing(que,dic,text):
#    res = []
#    for start in range(len(que)):
#        for end in range(start+1,len(que)):
#            seg = segment(que,start,end)
#            res.append(score(centroid(seg,dic),0.9,text))
#    return res
def dymatic_planing(que,dic,refer):
    res = []
    for start in range(len(que)):
        for end in range(start+1,len(que)):
            cand,seg = segment(que,start,end)
            rank = score(cand,seg,dic,refer)
            if rank!=0:
                res.append((cand,rank))
    return res
def mapping(vec,dic):
    res = []
    chunk = int(10406793/10)
    for i in range(10):
        print(i)
        u = AnnoyIndex(100, 'angular')
        u.load('D:/distance__'+str(i)) # mmap the file
        print(u.get_nns_by_vector(vec, 2))
        for g in range(2):
            res.append(dic[int(u.get_nns_by_vector(vec, 2)[g])+i*chunk]) # will find the 1000 nearest neighbors
    return res
