# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:06:10 2019

@author: Wang
"""

import os
import re
import string
import nltk
class document():
    def __init__(self,text,origin):
        self.annotation = re.split("\n",open(origin).read())
        self.name = text
        self.text = nltk.word_tokenize(open(text).read())
        i = 0
        annotation_copy = re.split("\n",open(origin).read())
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

def features(obj,index):
        return {
                "words" :obj.text[index],
                "length":len(obj.text[index]),
                "past":  '' if index == 0 else obj.text[index-1],
                "next":  '' if index == len(obj.text)-1 else obj.text[index+1],
                "caps":  obj.text[index][0].upper() == obj.text[index][0],
                "us_num":obj.text[index].isalpha(),
                }
def tag(obj,index):
    flag =0
    loc = 0
    for i in range(len(obj.span)):
        if obj.text[index] not in obj.span[i]: 
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
def transform(objects):
    x,y = [],[]
    for i in range(len(objects)):
        for g in range (len(objects[i].text)):
                x.append(features(objects[i],g))
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