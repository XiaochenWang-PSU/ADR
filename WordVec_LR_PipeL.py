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
import preprocess
import os
import numpy as np
import gc

#dic = open('D:\Medhelp_200d.txt','r',encoding='UTF-8').read()
#dic = dic.split('\n')
#for i in range(len(dic)):
#    dic[i] = dic[i].split(' ')
#
#objects = []
#for i in range(int(len(os.listdir('D:/cadec/text')))):
#    text_filename = os.listdir('D:/cadec/text')[i]
#    Ori_filename = os.listdir('D:/cadec/Ori')[i]
#    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename,dic))
#    print(i)
#sel = (len(objects))
#x,y,l = transform(objects[:sel],dic)
#train_cut = int(0.75*(len(objects)))
#test_cut = (len(objects))
#x_train,y_train,l_train = transform(objects[:train_cut],dic)
#x_test,y_test,l_test = transform(objects[train_cut:test_cut],dic)
#x,y = clean(x,y)
#x_train,y_train = clean(x_train,y_train)
#x_test,y_test = clean(x_test,y_test)
#print(y_test)
#print("completed")
#vecx =   DictVectorizer(sparse=True)
#vecy =   DictVectorizer(sparse=False)
#train_len = len(x_train)
#test_len = len(x_test)
#y_train = (y)[:(train_len)]
#clf = LogisticRegression()
#print("start training")
#clf.fit(x_train,y_train)
#y_test = (y)[(train_len):(train_len)+(test_len)]
#y_pred = [[i] for i in clf.predict(x_test)]
#recall_str=classification_report(y_test,y_pred)
#print(recall_str)
indicator = 0    
y_test_,y_pred_ = [],[]
l_perdocu = []
trans = objects[train_cut:test_cut]
for i in range(len(trans)):
    y_pred_.append([y_pred[g][0] for g in range(indicator,indicator+len(trans[i].ori_text))])
    y_test_.append([y_test[g][0] for g in range(indicator,indicator+len(trans[i].ori_text))])
    l_perdocu.append([l_test[g] for g in range(indicator,indicator+len(trans[i].ori_text))])
    indicator += len(trans[i].ori_text)
#te = []
#for i in range(len(y_test_)):
#    f=open('D:/LR_true/true'+str(i)+'.ann','a')
#    t = open('D:/LR_true/true'+str(i)+'.txt','a')
#    te.append(objects[train_cut:test_cut][i].ori_text)
#    t.write(objects[train_cut:test_cut][i].ori.replace('\n',' '))
#    for g in range(len(y_test_[i])):
#        if y_test_[i][g]!="Other":
#            info = "T"+str(g)+"\t"+y_test_[i][g]+" "+str(l_perdocu[i][g][1])+" "+str(l_perdocu[i][g][2])+"\t"+l_perdocu[i][g][0]+"\n"
#            f.write(info)
#    f.close()
#    t.close()

for i in range(len(y_pred_)):
    p_pred = 0
    p_test = 0
    for g in range(len(y_pred_[i])):
        if y_pred_[i][g] != p_pred:
            p_pred = y_pred_[i][g]
            y_pred_[i][g] = "B-"+y_pred_[i][g]
        else:
            p_pred = y_pred_[i][g]
            y_pred_[i][g] = "I-"+y_pred_[i][g]
        if y_test_[i][g] != p_test:
            p_test = y_test_[i][g]
            y_test_[i][g] = "B-"+y_test_[i][g]
        else:
            p_test = y_test_[i][g]
            y_test_[i][g] = "I-"+y_test_[i][g]
metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                   'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0}

# overall results
results = {'strict': deepcopy(metrics_results),
           'ent_type': deepcopy(metrics_results),
           'partial':deepcopy(metrics_results),
           'exact':deepcopy(metrics_results)
          }


# results aggregated by entity type
evaluation_agg_entities_type = {e: deepcopy(results) for e in ['Symptom']}

for true_ents, pred_ents in zip(y_test_, y_pred_):
    tmp_results, tmp_agg_results = compute_metrics(
        collect_named_entities(true_ents), collect_named_entities(pred_ents),  ['Symptom']
    )
    
    for eval_schema in results.keys():
        for metric in metrics_results.keys():
            results[eval_schema][metric] += tmp_results[eval_schema][metric]
            
    # Calculate global precision and recall
        
    results = compute_precision_recall_wrapper(results)


    # aggregate results by entity type
 
    for e_type in ['Symptom']:

        for eval_schema in tmp_agg_results[e_type]:

            for metric in tmp_agg_results[e_type][eval_schema]:
                
                evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]
                
        # Calculate precision recall at the individual entity level
                
        evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])
print(results)
        
        