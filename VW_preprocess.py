# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:41 2019

@author: Wang
"""

import os
import re
import string
import nltk
import numpy as np
from vowpalwabbit import pyvw
import sys
global past
class document():
    def __init__(self,text,origin,dic):
        self.annotation = re.split("\n",open(origin).read())
        self.ori = open(text).read()
        self.name = text
        self.text = nltk.word_tokenize(open(text).read())        
        i = 0
        annotation_copy = re.split("\n",open(origin).read())
        self.gram = nltk.pos_tag(self.text,tagset = 'universal')
        self.context = str(open(text).read())
        while i < len(self.annotation):
            if self.annotation[i] == "":
                del self.annotation[i]
                del annotation_copy[i]
            else:
                self.annotation[i] = re.split("\t| ",self.annotation[i])
                annotation_copy[i] = re.split("\t",annotation_copy[i])[2]
                if self.annotation[i][0].find('#') != -1:
                              self.annotation.pop(i)
                              annotation_copy.pop(i)
                              i-=1
            i+=1
        length = len(self.annotation)
        self.tag = [[] for g in range(length)]
        self.span = [[] for g in range(length)]
        for i in range(length):
            self.tag[i] = self.annotation[i][1]
            self.span[i] = re.split(' ',''.join(c for c in annotation_copy[i] if c not in string.punctuation))
        self.ori_text =[[] for i in range(len(self.text))]
        for i in range(len(self.text)):
            self.ori_text[i] = self.text[i] 
        for g in range(len(self.text)):
            flag = 0
            res = [0 for i in range(100)]
            for i in range(len(dic)):
                if self.text[g].lower() == dic[i][0] and len(dic[i]) == 102:
                    res = list(map(float,dic[i][1:101]))
                    flag = 1
                    break
            self.text[g] = res
def features(obj,index,dic):
    maxsum = maxnum = minnum = minsum = maxaver = minaver = 0
    past = [0 for i in range(100)] if index == 0 else obj.text[index-1]
    next_ = [0 for i in range(100)] if index == len(obj.text)-1 else obj.text[index+1]
    voc =  np.append(np.append(obj.text[index],past),next_)
    caps = float(0.1*(obj.ori_text[index][0].upper() == obj.ori_text[index][0]) and (index!=0 and obj.ori_text[index-1] != '.'))
    return list(np.append(np.append(voc,caps),float(0.01*len(obj.ori_text[index]))))
def tag(obj,index,pa):
    global past
    flag =0
    loc = 0
    na = "tag"
    for i in range(len(obj.span)):
        if (obj.ori_text[index] in obj.span[i]) and  ((len(obj.span[i]) ==1) or (obj.ori_text[index-1] in obj.span[i] or obj.ori_text[index+1] in obj.span[i])):
            flag =1
            loc = i
    if flag == 1:
        if obj.tag[loc] == "ADR":
                    past = 1
        else:
            past = -1
    else:
        past = -1
    return past
def location(obj,index):
    global before
    start = obj.context.find(obj.ori_text[index],before)
    end = obj.context.find(obj.ori_text[index],before)+len(obj.ori_text[index])
    before = end
    return(obj.ori_text[index],start,end)
def transform(objects,dic):
    x,y,loca = [],[],[]
    global past
    global before
    past = 0
    for i in range(len(objects)):
        print(i)
        before = 0
        for g in range (len(objects[i].text)):
                x.append(features(objects[i],g,dic))
                y.append(tag(objects[i],g,past))
                loca.append(location(objects[i],g))
                
    return x,y,loca
def clean(x,y):
    i = 0
    l = len(x)
    while i < l:
        if x[i] == None:
            del x[i]
            del y[i]
            l-=1
            i-=1
        i+=1
    return x,y
def clean_(x,y,z):
    i = 0
    l = len(x)
    while i < l:
        if x[i] == None:
            del x[i]
            del y[i]
            del z[i]
            l-=1
            i-=1
        i+=1
    return x,y,z
