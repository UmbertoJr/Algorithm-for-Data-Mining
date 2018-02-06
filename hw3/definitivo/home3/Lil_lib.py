# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt
from scipy import sparse

np.set_printoptions(precision=3)

def create_pivot_UB(X,y,Nu,Nb):
    rat=pd.concat([X,y],axis=1)
    pivot =sparse.lil_matrix((Nu, Nb))
    for line in rat.itertuples():
        pivot[line[3], line[4]] = line[5]
    return pivot

def norm_mean(pivot):
    piv=pivot.copy()
    for i in range(piv.shape[0]):
        m=mean(i,piv)+0.001
        for el in piv[i,:].nonzero()[1]:
            piv[i,el]-=m
    return piv

def create_pivot_IB(X,y,Nu,Nb):
    #Create two user-item matrices
    #n_users = X['User_Lab'].unique().shape[0]
    #n_items = X['Book_Lab'].unique().shape[0]
    #print('us',n_users,'it',n_items)
    rat=pd.concat([X,y],axis=1)
    pivot =sparse.lil_matrix((Nb, Nu))
    for line in rat.itertuples():
        pivot[line[4], line[3]] = line[5]
    return pivot

def neighbors(elem, cos, n):
    l = pd.Series(cos[elem,:].toarray()[0]).sort_values(ascending=0)[1:n+1]
    return l

def mean(elem, pivot):
    res = pivot[elem,:].sum()/pivot[elem,:].count_nonzero()
    return res

def mean_c(elem, pivot):
    res = pivot[:,elem].sum()/pivot[:,elem].count_nonzero()
    return res


def pred_rating_UB(user, item, closer, pivot):
    m = mean(user, pivot)
    if m!=m:m=0
    somma=0
    #lista={}
    for us in closer.index:
        somma+=closer[us]*(pivot[us,item]-mean(us,pivot))
        if somma<0:somma=0
    z=somma/closer.sum()
    if z!=z:z=0
    return round(float(m+z),3) #lista
        

def pred_rating_IB(item, user, closer, pivot):
    m = mean(item, pivot)
    if m!=m:m=0
    somma=0
    #lista={}
    for it in closer.index:
        somma+=closer[it]*(pivot[it,user]-mean(it,pivot))
        if somma<0:somma=0
    z=somma/closer.sum()
    if z!=z:z=0
    return round(float(m+z),3) #lista

def pred_row_UB(user,closer, pivot):
    userp={}
    for i in range(pivot.shape[1]):
        print('I am predicting the',i,'element of',pivot.shape[1])
        if pivot[user,i]==0:
            userp[i]=pred_rating_UB(user, i, closer, pivot)
    return pd.Series(userp)

#==============================================================================
# def pred_row_IB(item,closer, pivot):
#     itemp={}
#     for i in range(pivot.shape[1]):
#         print('I am predicting the',i,'element of',pivot.shape[1])
#         if pivot[item,i]==0:
#             itemp[i]=pred_rating_IB(item, i, closer, pivot)
#     return pd.Series(itemp)
#==============================================================================

def rmse(y,res):
    l=y.shape[0]
    return sqrt(((y.values-res.values)**2).sum()/l)

    
    
    
    
    
    
    
    
