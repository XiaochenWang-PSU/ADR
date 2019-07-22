# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 00:03:24 2019

@author: Wang
"""
from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score,roc_curve
from sklearn_crfsuite import metrics
from sklearn_crfsuite import CRF
import matplotlib.pylab as plt
from sklearn_crfsuite import scorers
import nltk
import sklearn
import preprocess
import os
import numpy as np
objects = []
for i in range(len(os.listdir('D:/cadec/text'))):
    text_filename = os.listdir('D:/cadec/text')[i]
    Ori_filename = os.listdir('D:/cadec/Ori')[i]
    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename)) 
train_cut = int(0.75*(len(objects)))
test_cut = int((len(objects)))
x_train,y_train = transform(objects[:train_cut])
x_test,y_test = transform(objects[train_cut:test_cut])
x_train,y_train = clean(x_train,y_train)
x_test,y_test = clean(x_test,y_test)
clf =CRF( 
     algorithm='lbfgs', 
     c1=0.8, 
     c2=0.1, 
     all_possible_transitions=True 
 ) 
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
y_pred_train=clf.predict(x_train)
metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=clf.classes_)
print(metrics.flat_classification_report(y_test, y_pred, labels=clf.classes_, digits=3 )) 

        
        
        
        
        
        
        
        
        
        