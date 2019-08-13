# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 00:03:24 2019

@author: Wang
"""
from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score,roc_curve
from sklearn_crfsuite import metrics
from sklearn_crfsuite import CRF
import matplotlib.pylab as plt
from copy import deepcopy
from sklearn_crfsuite import scorers
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import compute_precision_recall_wrapper
from ner_evaluation.ner_eval import Evaluator
from collections import defaultdict
import nltk
import sklearn
import crf_preprocess
import os
import numpy as np
#nltk.download('averaged_perceptron_tagger')
#dic = open('D:\MedHelp_100d.txt','r',encoding='UTF-8').read()
#dic = dic.split('\n')
#for i in range(len(dic)):
#    dic[i] = dic[i].split(' ')
#objects = []
#for i in range(int(len(os.listdir('D:/cadec/text')))):
#    text_filename = os.listdir('D:/cadec/text')[i]
#    Ori_filename = os.listdir('D:/cadec/Ori')[i]
#    objects.append(document('D:/cadec/text/'+text_filename,'D:/cadec/Ori/'+Ori_filename,dic))
#    print(i)
#sel = (len(objects))
#train_cut = int(0.75*(len(objects)))
#test_cut = int((len(objects)))
#x_train,y_train,l_train = transform(objects[:train_cut],dic)
#x_test,y_test,l_test = transform(objects[train_cut:test_cut],dic)
#x_train,y_train,l_train = clean_(x_train,y_train,l_train)
#print(len(x_train),(x_train[0]))
#print(len(y_train),(y_train))
#x_test,y_test,l_test = clean_(x_test,y_test,l_test)
#clf =CRF( 
#     algorithm='lbfgs',
#     c1=0.8, 
#     c2=0.12, 
#     all_possible_transitions=True 
# ) 
#clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
y_pred_train=clf.predict(x_train)
metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=clf.classes_)
print(metrics.flat_classification_report(y_test, y_pred, labels=clf.classes_, digits=3 )) 
for i in range(len(y_pred)):
    f=open('D:/CRF_pred/pred'+str(i)+'.ann','a')
    t = open('D:/CRF_pred/pred'+str(i)+'.txt','a')
    t.write(objects[train_cut:test_cut][i].ori.replace('\n',' '))
    for g in range(len(y_pred[i])):
        if y_pred[i][g].split('_')[0] == 'B':
            sentence = l_test[i][g][0]
            end = str(l_test[i][g][2])
            for l in range(1,len(y_pred[i])-g):
                if y_pred[i][g+l].split('_')[0] == 'I':
                    end = str(l_test[i][g+l][2])
                    sentence+=' '+(l_test[i][g+l][0])
                else:
                    break
#            y_test[i][g] = y_test[i][g].split('_')[1]
            info = "T"+str(g)+"\t"+y_pred[i][g].split('_')[1]+" "+str(l_test[i][g][1])+" "+end+"\t"+sentence+"\n"
            f.write(info)
            print(info)
    f.close()
    t.close()
#        
#
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
#for true_ents, pred_ents in zip(y_test, y_pred):
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
        
        
        
        
        