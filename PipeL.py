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
import matplotlib.pylab as plt
import nltk
import sklearn
import ADR
import os
import numpy as np
import gc
objects = []
for i in range(len(os.listdir('D:/cadec/text'))):
    text_filename = os.listdir('D:/cadec/text')[i]
    Ori_filename = os.listdir('D:/cadec/Ori')[i]
    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename)) 
sel = int(0.3*(len(objects)))
x,y = transform(objects[:sel])
train_cut = int(0.25*(len(objects)))
test_cut = int(0.3*(len(objects)))
x_train,y_train = transform(objects[:train_cut])
x_test,y_test = transform(objects[train_cut:test_cut])
del objects
gc.collect()
x,y = clean(x,y)
x_train,y_train = clean(x_train,y_train)
x_test,y_test = clean(x_test,y_test)
vec =   DictVectorizer(sparse=False)
train_len = len(x_train)
test_len = len(x_test)
x_train = vec.fit_transform(x)[:(train_len)]
y_train = vec.fit_transform(y)[:(train_len)]
#clf = DecisionTreeClassifier(splitter = "best",criterion = "entropy")
clf.fit(x_train,y_train)
del x_train
del y_train
gc.collect()
x_test = vec.fit_transform(x)[(train_len):(train_len)+(test_len)]
y_test = vec.fit_transform(y)[(train_len):(train_len)+(test_len)]
print(clf.score(x_test,y_test))
answer = clf.predict(x_test)
recall_str=classification_report(y_test,answer)
print(recall_str)    
        
        
        
        
        
        
        
        
        
        
        