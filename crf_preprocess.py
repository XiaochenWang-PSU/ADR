# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:06:10 2019

@author: Wang
"""

import os
import re
import string
import nltk
import numpy as np
global past
class document():
    def __init__(self,text,origin,dic):
        self.annotation = re.split("\n",open(origin).read())
        self.name = text
        self.text = nltk.word_tokenize(open(text).read())        
        i = 0
        annotation_copy = re.split("\n",open(origin).read())
        self.gram = nltk.pos_tag(self.text,tagset = 'universal')
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
    name = [ str(i) for i in range(300)]
    past_ = [0 for i in range(100)] if index == 0 else obj.text[index-1]
    next_ = [0 for i in range(100)] if index == len(obj.text)-1 else obj.text[index+1]
    voc =  np.append(np.append(obj.text[index],past_),next_)
    caps = float(0.1*(obj.ori_text[index][0].upper() == obj.ori_text[index][0]) and (index!=0 and obj.ori_text[index-1] != '.'))
    fin = np.append(np.append(voc,caps),float(0.1*len(obj.ori_text[index])))
    return dict(zip(name,fin))
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
        if index != 0 and pa in ("B_"+obj.tag[loc],"I_"+obj.tag[loc]):
                    past = "I_"+obj.tag[loc]
        else:
                    past = "B_"+obj.tag[loc]
    else:
        past = "Other"
    return past
def transform(objects,dic):
    x,y = [],[]    
    global past
    past = 0
    for i in range(len(objects)):
                x.append([features(objects[i],g,dic) for g in range(len(objects[i].text))])
                y.append([tag(objects[i],g,past) for g in range(len(objects[i].text))])
    return x,y
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