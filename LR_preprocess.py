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
import xml.etree.ElementTree as ET
import urllib.request
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.request, urllib.error, urllib.parse
import json
import os
from pprint import pprint


REST_URL = "http://data.bioontology.org"
API_KEY = "f6a71f57-a4ef-4389-9273-71c1faaf1605"

def get_json(url):
#    get class from api
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

def label_obtain(annotations, get_class=True):
    #input the text you want to annotate into the web, and receive the label of words
    answer = []
    label = []
    for result in annotations:
        class_details = result["annotatedClass"]
        if get_class:
            try:
                class_details = get_json(result["annotatedClass"]["links"]["self"])
            except urllib.error.HTTPError:
                print(f"Error retrieving {result['annotatedClass']['@id']}")
                continue
        g = get_json(class_details["links"]["ancestors"])
#        label.append(link[-1]["prefLabel"])
        for annotation in result["annotations"]:
            from_to = (int(annotation["from"]),int(annotation["to"]))
            if g!=[]:
                if len(g)>1:
                    label.append(g[-2]['prefLabel'])
                else:
                    label.append(g[-1]['prefLabel'])
            else:
                label.append("Not Found")
            
            answer.append(list(from_to))
    return answer,label
def getMeshIDFromLabel(disease,sparql):
#    used to find whether it is a disease
    disease = disease.lower()
    disease = """'""" + disease + """'"""
    # first we need to get the label
    query = """PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?s ?p ?o
                WHERE { ?s rdfs:label""" + disease + """@en}"""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    result = results["results"]["bindings"]
    if len(result) > 0:
        # retrieve the result from Wikidata
        label = result[0]["s"]["value"]
        # make it as a subject URI
        subject = """<""" + label + """>"""
        
        # then use the label as subject to retrieve
        # wdt:P486 is the predicate for mesh id
        query = """PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?o WHERE { """ + subject + """ wdt:P486 ?o.}"""
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        result = results["results"]["bindings"]
        if len(result) > 0:
            # get mesh id
            meshid = result[0]["o"]["value"]
        else:
            # 1 indicates no result because there is no predicate of "meshid"
            meshid = "1"
    else:
        # 2 indicates no result because this term does not exist in Wikidata
        # we can try aliases or dbpedia
        meshid = "2"
    return meshid
class document():
#    fundamental class, used to store all the information got from the txt and ann.
    def __init__(self,text,origin,dic):
        self.ori = open(text).read()
        self.annotation = re.split("\n",open(origin).read())
        self.name = text
        self.text = nltk.word_tokenize(open(text).read())        
        self.context = str(open(text).read())
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
def features(obj,index,sparql):
#    used to get the feature of each word
    past = [0 for i in range(50)] if index == 0 else obj.text[index-1]
    next_ = [0 for i in range(50)] if index == len(obj.text)-1 else obj.text[index+1]
    voc =  np.append(np.append(obj.text[index],past),next_)
    caps = float(0.1*(obj.ori_text[index][0].upper() == obj.ori_text[index][0]) and (index!=0 and obj.ori_text[index-1] != '.'))
#    if obj.ori_text[index].isalpha() is True:
#        if getMeshIDFromLabel(obj.ori_text[index],sparql)!=1 or 2:
#            dis_exist = 1
#        else:
#            dis_exist = 0
#    else:
#        dis_exist = 0
    past_tag = (np.append(np.append(voc,caps),float(0.01*len(obj.ori_text[index]))))
#    return list(np.append(past_tag,dis_exist))
    return past_tag
                
def tag(obj,index,pa):
#    used to get tag of each word
    global past
    flag =0
    loc = 0
    for i in range(len(obj.span)):
        if (obj.ori_text[index] in obj.span[i]) and  ((len(obj.span[i]) ==1) or (obj.ori_text[index-1] in obj.span[i] or obj.ori_text[index+1] in obj.span[i])):
            flag =1
            loc = i
    if flag == 1:
        past = obj.tag[loc]

    else:
        past = "Other"
    return [past]
def location(obj,index):
#    used to get where one word locate in order to further step: achieve the vitualization goal
    global before
    start = obj.context.find(obj.ori_text[index],before)
    end = obj.context.find(obj.ori_text[index],before)+len(obj.ori_text[index])
    before = end
    return(obj.ori_text[index],start,end)
def transform(objects):
#    input the data stored in the classes, transform it into list/dictionary.
    x,y,loca = [],[],[]
    div,mark = 0,0
#    mark is used to find where the data can be divided into the train set and test set
    global past
    global before
    past = 0
    vecx =   DictVectorizer(sparse=True)
    vecy =   DictVectorizer(sparse=False)
    for i in range(len(objects)):
        if i == int(0.75*len(objects)):
            print(i)
            div = mark
        print(i)
        before = 0
        if objects[i].context!='':
            text_to_annotate = objects[i].context
            annotations = get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(text_to_annotate))
            reference,label = label_obtain(annotations)
        for g in range (len(objects[i].ori_text)):
                loc =  location(objects[i],g)
                exist = "None"
                if reference:
                    if len(reference[0]) > 1:
                        for n in range(len(reference)):
                            if loc[1]+1>=reference[n][0] and loc[2]<=reference[n][1]:
                                exist = label[n]
                    else:
                            if loc[1]+1>=reference[0] and loc[2]<=reference[1]:
                                exist = label
                x.append(list(np.append(features(objects[i],g,sparql),exist)))
                y.append(tag(objects[i],g,past))
                mark += 1
#                loca.append(loc)
#    d = [[] for i in range(len(x))]
#    for i in range(len(x)):
#        d[i] = {'word':x[i][302]}
#        print(i)
#    onto = vecy.fit_transform(d)
#    x = np.delete(x,302,1)
#    x = np.append(x,onto,1)
    return x,y,div
def clean(x,y):
#    clean the empty list
    i = 0
    l = len(x)
    while i < l:
        if x[i] == []:
            del x[i]
            del y[i]
            l-=1
            i-=1
        i+=1
    return x,y
def clean_(x,y,z):
#    the same to above, with 3 input, specially designed for diliver the location information
    i = 0
    l = len(x)
    while i < l:
        if x[i] == []:
            del x[i]
            del y[i]
            del z[i]
            l-=1
            i-=1
        i+=1
    return x,y,z
