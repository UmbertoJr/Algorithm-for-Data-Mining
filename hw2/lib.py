#inverted index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import operator 
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import re

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
st = LancasterStemmer() 
stop=stopwords.words('english')
stop_ita=stopwords.words('italian')

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
    
#==============================================================================
# def termFrequency(term, document):
#     normalizeDocument = document.split()
#     return normalizeDocument.count(term) / float(len(normalizeDocument))
# 
# def inverseDocumentFrequency(term, allDocuments):
#     numDocWithThisTerm = 0
#     for doc in allDocuments:
#         if term in doc.split():
#             numDocWithThisTerm = numDocWithThisTerm + 1
#     if numDocWithThisTerm > 0:
#         return 1.0 + np.log(float(len(allDocuments)) / numDocWithThisTerm)
#     else:
#         return 1.0
#==============================================================================

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
    frase=input()
    Veg=False
    if bool(re.match(r"VV",frase))==True:
        Veg=True
        frase=frase[3:].lower()
    text=tokenizer.tokenize(frase)
    text=[i for i in text if i not in stop]
    text=[i for i in text if i not in stop_ita ]
    queryFormat=[]
    for w in text:
        queryFormat.append(st.stem(w))
    return queryFormat ,Veg 

def fix_wrt_q(cos,query,ListIntersQuery,MatOrig,Veg):
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
    else:
        print('Are you in the search engine')
        for i in range(len(ListIntersQuery)):
            for j in query:
                if j in MatOrig.loc[ListIntersQuery[i]][0]:
                   cos[0][i]*= 10
                if j in MatOrig.loc[ListIntersQuery[i]][6]:
                   cos[0][i]*= 5
        count=len(cos[0])
        return cos,count
    
    
   
def ranklist(inv,listwords,matrix,MatOrig) :
    queryFormat,Veg=search()
    prima=set(queryFormat)
    query=[i for i in queryFormat if i in listwords]
    dopo=set(query)
    if prima!=dopo:
        print('questa/e parola/e non appaiono nella query',prima-dopo)
    ListIntersQuery=inter(query,inv)
    Vquery=create_doc_query(query,listwords)
    cos=cosine_similarity(Vquery.reshape(1, -1),matrix.toarray()[ListIntersQuery])  
    cosfixed,count=fix_wrt_q(cos,query,ListIntersQuery,MatOrig,Veg)
#==============================================================================
#     for i in range(len(ListIntersQuery)):
#         for j in query:
#             if j in MatOrig.loc[ListIntersQuery[i]][0]:
#                cos[0][i]*= 10
#             if j in MatOrig.loc[ListIntersQuery[i]][6]:
#                cos[0][i]*= 5        
#==============================================================================
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
    while i<len(p1) and j<len(p2):
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
  for i in range(0,len(p),n):
      if i== numberpos:
          var=True
          break
  newpos=numberpos+n
  if newpos>=len(p):
      return var,len(p)-1
  else:
      return var,newpos

    
def matrixTfIdf(listadoc,listword):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=listword,norm='l2',smooth_idf=False)
    return tfidf_vectorizer.fit_transform(list(listadoc))
    
         
#==============================================================================
# def ranklistVeg(inv,listwords,matrix,MatOrig): 
#     queryFormat=search()
#     Veg=False
#     if bool(re.match(r"VV",queryFormat))==True:
#         print('Are you in the vegetarian search engine')
#         Veg=True
#     prima=set(queryFormat)
#     query=[i for i in queryFormat if i in listwords]
#     dopo=set(query)
#     if prima!=dopo:
#         print('questa/e parola/e non appaiono nella query',prima-dopo)
#     ListIntersQuery=inter(query,inv)
#     Vquery=create_doc_query(query,listwords)
#     cos=cosine_similarity(Vquery.reshape(1, -1),matrix.toarray()[ListIntersQuery])  
#     count=0
#     for i in range(len(ListIntersQuery)):
#         if MatOrig.loc[ListIntersQuery[i]][2]=='Vegetarian':
#             count+=1
#             print(MatOrig.loc[ListIntersQuery[i]][2])
#             for j in query:
#                 if j in MatOrig.loc[ListIntersQuery[i]][0]:
#                    cos[0][i]*= 10
#                 if j in MatOrig.loc[ListIntersQuery[i]][6]:
#                    cos[0][i]*= 5
#         else:         
#            cos[0][i]*= 0    
#     dizio={}
#     for i in range(len(cos[0])):
#         dizio[ListIntersQuery[i]]=cos[0][i]
#     return sorted(dizio.items(), key=operator.itemgetter(1),reverse=True)[:count]
# 
#==============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    