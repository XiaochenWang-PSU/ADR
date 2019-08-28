# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 00:03:24 2019

@author: Wang
"""
from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score,roc_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from sklearn_crfsuite import scorers
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import compute_precision_recall_wrapper
from ner_evaluation.ner_eval import Evaluator
from collections import defaultdict
import urllib.request
import nltk
import sklearn
import WordVec_LR_preprocess
import os
import numpy as np
import gc
proxy_support = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)
sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql",agent = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3722.400 QQBrowser/10.5.3738.400")

#preprocess work

dic = open('D:\Medhelp_50d.txt','r',encoding='UTF-8').read()
dic = dic.split('\n')
for i in range(len(dic)):
    dic[i] = dic[i].split(' ')

#import the MedHelp text data, in order to transform each word into wordvector

#objects = []
#for i in range(int(len(os.listdir('D:/cadec/text')))):
#    text_filename = os.listdir('D:/cadec/text')[i]
#    Ori_filename = os.listdir('D:/cadec/Ori')[i]
#    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename,dic))
#    print(i)


#deliver data into the classes 


x,y,mark = transform(objects)
vecy =   DictVectorizer(sparse=True)
d = [[] for i in range(len(x))]
for i in range(len(x)):
    d[i] = {'word':x[i][-1]}
onto = vecy.fit_transform(d)
onto = np.argmax(onto,axis = 1)
print("finish")
x_ = [s[0:151] for s in x]
x_ = np.append(x_,onto,1)
x_train,y_train = x_[:mark],y[:mark]
x_test,y_test = x_[mark:],y[mark:]
x_train,y_train = clean(x_train,y_train)
x_test,y_test = clean(x_test,y_test)
print("completed")

#make all the data saved into list or array

clf = LogisticRegression()
print("start training")
clf.fit(x_train,y_train)
print(len(y))
y_pred = clf.predict(x_test.astype(float))
recall_str=classification_report(y_test,y_pred)
print(recall_str)


#train,and show the performance of predicting

#indicator = 0    
#y_test_,y_pred_ = [],[]
#l_perdocu = []
#trans = objects[train_cut:test_cut]
#for i in range(len(trans)):
#    y_pred_.append([y_pred[g][0] for g in range(indicator,indicator+len(trans[i].ori_text))])
#    y_test_.append([y_test[g][0] for g in range(indicator,indicator+len(trans[i].ori_text))])
#    l_perdocu.append([l_test[g] for g in range(indicator,indicator+len(trans[i].ori_text))])
#    indicator += len(trans[i].ori_text)

# transform the input of classifaction into format as "list of list"


#te = []
#for i in range(len(y_test_)):
#    f = open('D:/LR_pred/'+trans[i].name.split('/')[-1].split('.')[0]+'.'+trans[i].name.split('/')[-1].split('.')[1]+'.ann','a')
#    t = open('D:/LR_pred/'+trans[i].name.split('/')[-1].split('.')[0]+'.'+trans[i].name.split('/')[-1].split('.')[1]+'.txt','a')
#    te.append(objects[train_cut:test_cut][i].ori_text)
#    t.write(objects[train_cut:test_cut][i].ori.replace('\n',' '))
#    for g in range(len(y_test_[i])):
#        if y_test_[i][g]!="Other":
#            info = "T"+str(g)+"\t"+y_test_[i][g]+" "+str(l_perdocu[i][g][1])+" "+str(l_perdocu[i][g][2])+"\t"+l_perdocu[i][g][0]+"\n"
#            f.write(info)
#    f.close()
#    t.close()


#write into text files in right format in order to vitualize the result of predication


#for i in range(len(y_pred_)):
#    p_pred = 0
#    p_test = 0
#    for g in range(len(y_pred_[i])):
#        if y_pred_[i][g] != p_pred:
#            p_pred = y_pred_[i][g]
#            y_pred_[i][g] = "B-"+y_pred_[i][g]
#        else:
#            p_pred = y_pred_[i][g]
#            y_pred_[i][g] = "I-"+y_pred_[i][g]
#        if y_test_[i][g] != p_test:
#            p_test = y_test_[i][g]
#            y_test_[i][g] = "B-"+y_test_[i][g]
#        else:
#            p_test = y_test_[i][g]
#            y_test_[i][g] = "I-"+y_test_[i][g]


# add B-I label, in order to have a NER evaluation



#metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
#                   'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0}
#
## overall results
#results = {'strict': deepcopy(metrics_results),
#           'ent_type': deepcopy(metrics_results),
#           'partial':deepcopy(metrics_results),
#           'exact':deepcopy(metrics_results)
#          }
#
#
## results aggregated by entity type
#evaluation_agg_entities_type = {e: deepcopy(results) for e in ['Symptom']}
#
#for true_ents, pred_ents in zip(y_test_, y_pred_):
#    tmp_results, tmp_agg_results = compute_metrics(
#        collect_named_entities(true_ents), collect_named_entities(pred_ents),  ['Symptom']
#    )
#    
#    for eval_schema in results.keys():
#        for metric in metrics_results.keys():
#            results[eval_schema][metric] += tmp_results[eval_schema][metric]
#            
#    # Calculate global precision and recall
#        
#    results = compute_precision_recall_wrapper(results)
#
#
#    # aggregate results by entity type
# 
#    for e_type in ['Symptom']:
#
#        for eval_schema in tmp_agg_results[e_type]:
#
#            for metric in tmp_agg_results[e_type][eval_schema]:
#                
#                evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]
#                
#        # Calculate precision recall at the individual entity level
#                
#        evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])
#print(results)
        
#used for NER evaluation