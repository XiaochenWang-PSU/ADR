# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 00:03:24 2019

@author: Wang
"""
from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pylab as plt
import nltk
import sklearn
import preprocess
import os
import numpy as np
import gc
dic = open('D:\MedHelp_50d.txt','r',encoding='UTF-8').read()
dic = dic.split('\n')
for i in range(len(dic)):
    dic[i] = dic[i].split(' ')
objects = []
for i in range(int(len(os.listdir('D:/cadec/text')))):
    text_filename = os.listdir('D:/cadec/text')[i]
    Ori_filename = os.listdir('D:/cadec/Ori')[i]
    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename,dic))
    print(i)
sel = (len(objects))
x,y = transform(objects[:sel],dic)
train_cut = int(0.75*(len(objects)))
test_cut = (len(objects))
x_train,y_train = transform(objects[:train_cut],dic)
x_test,y_test = transform(objects[train_cut:test_cut],dic)
del objects
gc.collect()
x,y = clean(x,y)
x_train,y_train = clean(x_train,y_train)
x_test,y_test = clean(x_test,y_test)
print("completed")
vecx =   DictVectorizer(sparse=True)
vecy =   DictVectorizer(sparse=False)
train_len = len(x_train)
test_len = len(x_test)
y_train = vecx.fit_transform(y)[:(train_len)]
y_train = np.argmax(y_train,axis = 1)
clf = LogisticRegression()
clf.fit(x_train,y_train)
y_test = vecx.fit_transform(y)[(train_len):(train_len)+(test_len)]
y_test = np.argmax(y_test,axis = 1)
answer = clf.predict(x_test)
recall_str=classification_report(y_test,answer)
print(recall_str)    