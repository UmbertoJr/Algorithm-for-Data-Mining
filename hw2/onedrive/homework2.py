
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from operator import itemgetter
import pandas as pd
import re
import csv
import numpy as np
import lib as lb
import importlib
np.set_printoptions(precision=2)
importlib.reload(lb)
#%%
recipes=pd.read_csv("ricettecontitolo.csv",sep='\t')

f=open("ricette.csv","r",encoding='utf-8-sig')
ricette=[]
for row in csv.reader(f, delimiter='\t'):
    if row:
        a=[]
        a.extend(row[:3])
        a.extend(row[6:])
        ricette.append(a)
f.close()

#%%
# Creo alcune Funzioni da utilizzare
stop=stopwords.words('english') 
stop_ita=stopwords.words('italian')   
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
st = LancasterStemmer() 
#%%
pos=[0,3,4]
Num_Ricetta={}
ListaDocs={}
for i in range(len(ricette)):
    for q in range(0,11200,200):
        if i==q:
            print('ricetta numero',q)
    #Associo numero a ricetta
    Num_Ricetta[i]=ricette[i][0] 
    # Trasformo ricetta in testo
    text_i=" ".join(list(itemgetter (*pos)(ricette[i]))).lower()
    tit_i=" ".join(ricette[i][1:3]).lower()
    # Rimuovo segni di punteggiatura   
    for p in punctuation:
        text_i=text_i.replace(p,' ')
        tit_i=tit_i.replace(p,' ')
    # Creo lista di tokens   
    tokens_i=tokenizer.tokenize(text_i)     
    # Rimuovo stopwords
    doc_i=[i for i in tokens_i if i not in stop]
    # Pulisco da imperfezioni tipo 1/2 3/4    
    words_i=[]
    for word in doc_i:
        if bool(re.search(r"[^a-zA-Z0-9àåãåâûùöôòìîèéêñç]",word))!=True:
            words_i.append(word)
    # Stemming delle parole        
    stem_i=[]
    for w in words_i:  
        stem_i.append(st.stem(w))
        
    stem_i.extend(tit_i.split())
    # Creo dizionario con numero ricetta come key e doc come value
    ListaDocs[i]=" ".join(stem_i)
    
#%%
#Creo inverted Index
inverted=lb.create_index(ListaDocs)
wordF=list(inverted.keys())
#%%
#==============================================================================
# freq=pd.DataFrame(np.zeros((len(ricette),len(wordF))))
# freq.shape
# 
# for i,j in zip(range(freq.shape[0]),range(2)):
#     print(j)
#     for q in range(len(wordF)):
#         if wordF[q] in dict[i]:
#             tf=lb.termFrequency(wordF[q],dict[i])
#             if tf!=0:
#                 idf=lb.inverseDocumentFrequency(wordF[q], dict.values())
#                 freq.iloc[i,q]=tf*idf
#                 print('parola',wordF[q],'numero',q,'tf-idf',idf*tf)
#             else:
#                 pass
#==============================================================================

#%%
# Creo matrice di tfidf 
tfidf_matrix=lb.matrixTfIdf(ListaDocs.values(),wordF)
print('Questa è una matrice di dimensioni',tfidf_matrix.shape)
#%%
# Aumento size del titolo  DA RISOLVEREEEEEee
n,m=tfidf_matrix.shape
text_i=" ".join(list(Num_Ricetta.values())).lower()
# Rimuovo segni di punteggiatura   
for p in punctuation:
    text_i=text_i.replace(p,' ')
# Creo lista di tokens   
tokens_i=tokenizer.tokenize(text_i)     
# Rimuovo stopwords
doc_i=[i for i in tokens_i if i not in stop]
# Pulisco da imperfezioni tipo 1/2 3/4    
words_i=[]
for word in doc_i:
    if bool(re.search(r"[^a-zA-Z0-9àåãåâûùöôòìîèéêñç]",word))!=True:
        words_i.append(word)
# Stemming delle parole        
paroleNelTit=[]
for w in words_i:
    paroleNelTit.append(st.stem(w))

paroleNelTit=set(paroleNelTit)

'''for i in range(n):
    print(i)
    for j in range(m):
        if tfidf_matrix[i,j]!=0.0 and wordF[j] in paroleNelTit:
            tfidf_matrix[i,j]=tfidf_matrix[i,j]*10'''

#%%            
#FormatQuery=lb.search()
#parmigiana di melanzane
rank_q=lb.ranklist(inverted,wordF,tfidf_matrix,recipes)
#[(Num_Ricetta[rank_q[i][0]],'#',rank_q[i][0]) for i in range(len(rank_q))][:10]
#recipes.loc[6304]

#%%

