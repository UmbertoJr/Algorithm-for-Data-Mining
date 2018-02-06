# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:35:17 2016

@author: Umbertojunior
"""

#inverted index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import operator 
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import re
import csv
import requests
from bs4 import BeautifulSoup
import time

def extractBasicInfo(repSoup,itemtype):
    result=""
    for tag in repSoup.find_all(itemprop=itemtype):
        result=tag.contents[0]
    return result
    
def extractDietaryInfo(repSoup):
    result=""
    for tag in repSoup.find_all('p'):
        if tag.get('class')==['recipe-metadata__dietary-vegetarian-text']:
            result=re.sub("[\n| ]*","",tag.contents[0])
    return result
    
def extractMethodInfo(repSoup):
    result=[]
    for tag in repSoup.find_all(itemprop="recipeInstructions"):
        result.append(str(*tag.contents[1].contents))
    return result

def extractIngreInfo(repSoup):
    result=[]   
    for tag in repSoup.find_all(itemprop="ingredients"):  
        L=len(tag.contents)
        stringa=""
        for i in range(L):
            if str(type(tag.contents[i]))=="<class 'bs4.element.Tag'>":
                stringa+=str(*tag.contents[i].contents)
                
            elif str(type(tag.contents[i]))=="<class 'bs4.element.NavigableString'>":
                stringa+=str(tag.contents[i])    
                
        result.append(stringa)
    return result 
      
def extractAllinfo(recipe):
    cnt= requests.get("http://www.bbc.co.uk"+ recipe)
    count=0
    while str(cnt)!='<Response [200]>' and count<10:
        time.sleep(1)
        count+=1
        cnt = requests.get("http://www.bbc.co.uk"+ recipe)
    pSoup=BeautifulSoup(cnt.text, "lxml")
    contents={}
    contents["name"]=pSoup.title.contents[0][21:]
    contents["prepTime"]=extractBasicInfo(pSoup,"prepTime")
    contents["cookTime"]=extractBasicInfo(pSoup,"cookTime")
    contents["author"]=extractBasicInfo(pSoup,"author")
    contents["recipeYield"]=extractBasicInfo(pSoup,"recipeYield")
    contents["ingredients"]=extractIngreInfo(pSoup)
    contents["instructions"]=extractMethodInfo(pSoup)
    contents["dietaryInfo"]=extractDietaryInfo(pSoup)
    return contents 
    
def All_in_CSV(allrecipes):
    with open('ricette.csv', 'a',encoding='utf8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        cont=0
        for k in allrecipes:
            a=extractAllinfo(k)
            spamwriter.writerow([a[i] for i in ['name',"author","dietaryInfo",'prepTime',"cookTime","recipeYield","ingredients","instructions"]])
            for i in range(1,11200,15):
                if(cont==i):
                    print('sono arrivato a',cont)
            cont+=1
        csvfile.close()
    return print('Ho fatto')           
#=============================================================
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
st = LancasterStemmer() 
stop=stopwords.words('english')
stop_ita=stopwords.words('italian')

#=========================
def create_index(tokens):
    inverted_index = {}
    i=0
    for k, v in tokens.items():
        for q in range(0,11200,100):
            if i==q:
                print('ricetta numero',q)
        i+=1
        for word in v.split():
            if inverted_index.get(word,False):
                if k not in inverted_index[word]:
                    inverted_index[word].append(k)
            else:
                inverted_index[word] = [k]
    return inverted_index
    
def create_doc_query(q,listwords):
    pos=[]
    for i in q:
        pos.append(listwords.index(i))        
    Vquery=np.zeros((len(listwords)))
    Vquery[pos]=1.0
    return Vquery

def inter(query,inv):
    n=len(query)
    if n>=2:
        val=interwhithskip(inv[query[0]],inv[query[1]])
        for i in range(2,n):
            val=interwhithskip(val,inv[query[i]])
        return val
    else:
        return inv[query[0]]
   
def search():
    print('Find something below')
    print('Hint:If you put \'VV\' in the beginning you will see only vegetarian recipes')
    print('Hint:If you put \'IL\' in the beginning you will see only recipes for lactose intolerant')    
    frase=input()
    Veg=False
    IL=False
    if bool(re.match(r"VV",frase))==True:
        Veg=True
        frase=frase[3:].lower()
    elif bool(re.match(r"IL",frase))==True:
        IL=True
        frase=frase[3:].lower()    
    text=tokenizer.tokenize(frase)
    text=[i for i in text if i not in stop]
    text=[i for i in text if i not in stop_ita ]
    queryFormat=[]
    for w in text:
        queryFormat.append(st.stem(w))
    return queryFormat ,Veg,IL 

def fix_wrt_q(cos,query,ListIntersQuery,MatOrig,Veg,IL):
    if Veg:
        print('Are you in the vegetarian search engine')
        count=0
        for i in range(len(ListIntersQuery)):
            if MatOrig.loc[ListIntersQuery[i]][2]=='Vegetarian':
                count+=1
                for j in query:
                    if j in MatOrig.loc[ListIntersQuery[i]][0]:
                       cos[0][i]*= 10
                    if j in MatOrig.loc[ListIntersQuery[i]][6]:
                       cos[0][i]*= 5
            else:         
               cos[0][i]*= 0
        return cos,count
    elif IL:
            print('Are you in the search engine for lactose intolerant')
            count=0
            for i in range(len(ListIntersQuery)):
                if MatOrig.loc[ListIntersQuery[i]][8]=='Lactose Intolerant':
                    count+=1
                    for j in query:
                        if j in MatOrig.loc[ListIntersQuery[i]][0]:
                           cos[0][i]*= 10
                        if j in MatOrig.loc[ListIntersQuery[i]][6]:
                           cos[0][i]*= 5
                else:         
                   cos[0][i]*= 0
            return cos,count           
    else:
            print('Are you in the search engine')
            count=0
            for i in range(len(ListIntersQuery)):
                count+=1
                for j in query:
                    if j in MatOrig.loc[ListIntersQuery[i]][0]:
                       cos[0][i]*= 10
                    if j in MatOrig.loc[ListIntersQuery[i]][6]:
                       cos[0][i]*= 5
            return cos,count
        
    
    
   
def ranklist(inv,listwords,matrix,MatOrig) :
    query=[]
    while not query:
        queryFormat,Veg,IL=search()
        prima=set(queryFormat)
        query=[i for i in queryFormat if i in listwords]
        dopo=set(query)
        if not query:
            print('No Result found')
        if prima!=dopo:
            print('These words don\'t will search',prima-dopo)  
    ListIntersQuery=inter(query,inv)
    Vquery=create_doc_query(query,listwords)
    cos=cosine_similarity(Vquery.reshape(1, -1),matrix.toarray()[ListIntersQuery])  
    cosfixed,count=fix_wrt_q(cos,query,ListIntersQuery,MatOrig,Veg,IL)
    dizio={}
    for i in range(len(cosfixed[0])):
        dizio[ListIntersQuery[i]]=cosfixed[0][i]
    rank_q=sorted(dizio.items(), key=operator.itemgetter(1),reverse=True)[:count]
    print('Ci sono',len(rank_q),'ricette')
    return rank_q
  
def interwhithskip(p1,p2):
    answer=[]
    i=0
    j=0
    while i<(len(p1)) and j<(len(p2)):
        if p1[i]==p2[j]:
            answer.append(p1[i])
            i+=1
            j+=1
        else:
            if p1[i]<p2[j]:
                var,i_new=hasSkip(i,p1)
                if var and p1[i_new]<=p2[j]:
                    
                    while var and p1[i_new]<=p2[j]:
                        i=i_new
                        var,i_new=hasSkip(i,p1)                        
                else:
                    i+=1
            else:
                var,j_new=hasSkip(j,p2)
                if var and p2[j_new]<=p1[i]:
                    while var and p2[j_new]<=p1[i]:
                        j=j_new
                        var,j_new=hasSkip(j,p2)
                else:
                    j+=1
    return answer

def hasSkip(numberpos,p):
  n=int(round(np.sqrt(len(p))))
  var=False
  for pos_skip in range(0,len(p),n):
      if pos_skip == numberpos:
          var=True
          newpos=numberpos+n
          break
  if var==False:
      newpos=numberpos
  if newpos>=len(p):
      var=False
      return var,len(p)-1
  else:
      return var,newpos
                         
def matrixTfIdf(listadoc,listword):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=listword,norm='l2',smooth_idf=False)
    return tfidf_vectorizer.fit_transform(list(listadoc))

#%% per app   
    
    
def search_app(inp):
    #print('Find something below')
    #print('Hint:If you put \'VV\' in the beginning you will see only vegetarian recipes')
    frase=inp
    Veg=False
    li=False
    if bool(re.match(r"VV",frase))==True:
        Veg=True
        frase=frase[3:].lower()
    if bool(re.match(r"IL",frase))==True:
        li=True
        frase=frase[3:].lower()
    if bool(re.match(r"VV IL",frase))==True:
        li=True
        Veg=True
        frase=frase[5:].lower()
    text=tokenizer.tokenize(frase) 
    text=[i for i in text if i not in stop]
    text=[i for i in text if i not in stop_ita]
       
    queryFormat=[]
    for w in text:
        queryFormat.append(st.stem(w))
    return queryFormat , Veg, li, text
    
    
def wordsnotin(inp,inv,listwords,matrix,MatOrig) :
    queryFormat,Veg,li,s=search_app(inp)
    prima=set(queryFormat)
    query=[i for i in queryFormat if i in listwords]
    dopo=set(query)
    ogg=[]
    if prima!=dopo:
        for el in prima-dopo:
            for parol in s:
                if bool(re.match(el,parol)):
                    ogg.append(parol)
        if len(ogg)>1:
            ogg=["these words aren't found"]+[' , '.join(ogg)]
            q=' : '.join(ogg)
            return q
        elif len(ogg)==1 :
            ogg=["this word isn't found"]+[' , '.join(ogg)]
            q=' : '.join(ogg)
            return q
    if len(dopo)==0:
        return None 

    
def fix_wrt_q_app(cos,query,ListIntersQuery,MatOrig,Veg,IL):
    if Veg:
        #print('Are you in the vegetarian search engine')
        count=0
        for i in range(len(ListIntersQuery)):
            if MatOrig.loc[ListIntersQuery[i]][2]=='Vegetarian':
                count+=1
                for j in query:
                    if j in MatOrig.loc[ListIntersQuery[i]][0]:
                       cos[0][i]*= 10
                    if j in MatOrig.loc[ListIntersQuery[i]][6]:
                       cos[0][i]*= 5
            else:         
               cos[0][i]*= 0
        return cos,count
    elif IL:
        #print('Are you in the search engine for lactose intolerant')
        count=0
        for i in range(len(ListIntersQuery)):
            if MatOrig.loc[ListIntersQuery[i]][8]=='Lactose Intolerant':
                count+=1
                for j in query:
                    if j in MatOrig.loc[ListIntersQuery[i]][0]:
                       cos[0][i]*= 10
                    if j in MatOrig.loc[ListIntersQuery[i]][6]:
                       cos[0][i]*= 5
            else:         
               cos[0][i]*= 0
        return cos,count           
    else:
        #print('Are you in the search engine')
        count=0
        for i in range(len(ListIntersQuery)):
            count+=1
            for j in query:
                if j in MatOrig.loc[ListIntersQuery[i]][0]:
                   cos[0][i]*= 10
                if j in MatOrig.loc[ListIntersQuery[i]][6]:
                   cos[0][i]*= 5
        return cos,count
        
def ranklist_app(inp,inv,listwords,matrix,MatOrig) :
    queryFormat,Veg,li,s=search_app(inp)
    prima=set(queryFormat)
    query=[i for i in queryFormat if i in listwords]
    dopo=set(query)
    ogg=[]
    q=''
    if prima!=dopo:
        for el in prima-dopo:
            for parol in s:
                if bool(re.match(el,parol)):
                    ogg.append(parol)
        if len(ogg)>1:
            ogg=["these words aren't found"]+[' , '.join(ogg)]
            q=' : '.join(ogg)
        elif len(ogg)==1 :
            ogg=["this word isn't founded"]+[' , '.join(ogg)]
            q=' : '.join(ogg)
    if len(dopo)==0:
        return None    
    ListIntersQuery=list(inter(query,inv))
    Vquery=create_doc_query(query,listwords)
    cos=cosine_similarity(Vquery.reshape(1, -1),matrix.toarray()[ListIntersQuery])
    cosfixed,count=fix_wrt_q_app(cos,query,ListIntersQuery,MatOrig,Veg,li)
    dizio={}
    for i in range(len(cosfixed[0])):
        dizio[ListIntersQuery[i]]=cosfixed[0][i]
    rank_q=sorted(dizio.items(), key=operator.itemgetter(1),reverse=True)[:count]
    return rank_q

def num_app(rank_q):
    h='Ci sono',len(rank_q),'ricette'
    return h
