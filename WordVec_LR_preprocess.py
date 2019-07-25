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
            res = [0 for i in range(50)]
            for i in range(len(dic)):
                if self.text[g].lower() == dic[i][0] and len(dic[i]) == 52:
                    res = list(map(float,dic[i][1:51]))
                    flag = 1
                    break
            self.text[g] = res
def features(obj,index,dic):
    past = [0 for i in range(50)] if index == 0 else obj.text[index-1]
    next_ = [0 for i in range(50)] if index == len(obj.text)-1 else obj.text[index+1]
    voc =  np.append(np.append(obj.text[index],past),next_)
    caps = float(0.01*(obj.ori_text[index][0].upper() == obj.ori_text[index][0]) and (index!=0 and obj.ori_text[index-1] != '.'))
#    return list(voc)
    return list(np.append(np.append(voc,caps),float(0.01*len(obj.ori_text[index]))))
#     return obj.text[index]
#    return list(np.append(voc,len(obj.ori_text[index])))
#                "length":len(obj.text[index]),
#                "past":  '' if index == 0 else obj.text[index-1],
#                "next":  '' if index == len(obj.text)-1 else obj.text[index+1],
#                "caps":  (ori.upper() == ori[0]) and (index!=0 and obj.text[index-1] != '.'),
#                "us_num":ori.isalpha(),
#                't':obj.gram[index][1],
#                't_p':'' if index == 0 else obj.gram[index-1][1],
#                't_n':'' if index == len(obj.text)-1 else obj.gram[index+1][1]
                
def tag(obj,index):
    flag =0
    loc = 0
    for i in range(len(obj.span)):
        if obj.ori_text[index] not in obj.span[i]: 
            pass
        else:
            flag =1
            loc = i
    if flag == 1:
        return  {
                "tag": obj.tag[i]
                }
    else:
        return  {
                    "tag":"Other"
                    }
def transform(objects,dic):
    x,y = [],[]
    
    for i in range(len(objects)):
        for g in range (len(objects[i].text)):
                x.append(features(objects[i],g,dic))
                y.append(tag(objects[i],g))
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