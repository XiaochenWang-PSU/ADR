# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:22:16 2019

@author: Wang
"""
from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from sklearn_crfsuite import scorers
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import compute_precision_recall_wrapper
from ner_evaluation.ner_eval import Evaluator
from collections import defaultdict
from sklearn.feature_selection import RFE
import matplotlib.pylab as plt
import nltk
import sklearn
import VW_preprocess
import os
import numpy as np
import gc
from vowpalwabbit.sklearn_vw import VWClassifier
#
dic = open('D:\Medhelp_100d.txt','r',encoding='UTF-8').read()
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
x,y,l = transform(objects[:sel],dic)
train_cut = int(0.75*(len(objects)))
test_cut = (len(objects))
x_train,y_train,l_train = transform(objects[:train_cut],dic)
x_test,y_test,l_test = transform(objects[train_cut:test_cut],dic)
x,y = clean(x,y)
x_train,y_train = clean(x_train,y_train)
x_test,y_test = clean(x_test,y_test)
print(x_test)
print(y_test)
print("completed")
vecx =   DictVectorizer(sparse=True)
vecy =   DictVectorizer(sparse=False)
train_len = (x_train)
test_len = (x_test)
clf = VWClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(len(y_pred))
print((y_pred))
print((y_test))
recall_str=classification_report(y_test,y_pred)
print(recall_str)


