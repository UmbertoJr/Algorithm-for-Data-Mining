# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 10:51:35 2016

@author: Umbertojunior
"""

import lib2 as lb
import pandas as pd
import re
import csv
import numpy as np
import importlib
np.set_printoptions(precision=2)
importlib.reload(lb)

#%%
l=open('listdoc.txt','r', encoding='utf-8-sig')
a=l.read()
ListaDocs=eval(a)
l.close()

h=open('words.txt', 'r')
b=h.read()
wordF=eval(b)
h.close()
#%%

tfidf_matrix=lb.matrixTfIdf(ListaDocs.values(),wordF)
#print('Questa è una matrice di dimensioni',tfidf_matrix.shape)

#%%
g= open('invertedInd.txt', 'r')
lec=g.read()
inverted=eval(lec)
g.close()

recipes=pd.read_csv("ricettecontitolo.csv",sep='\t')

f= open('Namerecip.txt', 'r', encoding='utf-8-sig')
lect=f.read()
Num_Ricetta=eval(lect)
f.close()
#%%
rank_q=lb.ranklist(input(),inverted,wordF,tfidf_matrix,recipes)
#[(Num_Ricetta[rank_q[i][0]],'#',rank_q[i][0]) for i in range(len(rank_q))][:10]
#recipes.loc[6304]
