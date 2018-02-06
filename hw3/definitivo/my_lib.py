import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def take_most_commun(piv,user,number):
    dic={}
    h = piv[user,:].nonzero()[1]
    for i in range(piv.shape[0]):
        z=0
        dist=0
        if i != user:
            o=piv[i,:].nonzero()[1]
            for col in np.intersect1d(h,o):
                z+=1
                dist+= abs(piv[user,col]-piv[i,col])
            if z!=0:
                dic[i]=[z,dist/z ,z**2/(dist+1)]
    if len(dic)==0:
        return None
    data=pd.DataFrame.from_dict(dic, orient='index')
    data.columns=['comune_columns','distance','my_index']
    if data.shape[0]>number:
        return data.sort_values(['comune_columns','distance'],axis=0, ascending=[0,1])[:number]
    else:
        return data.sort_values(['comune_columns','distance'],axis=0, ascending=[0,1])
    '''if len(dic)>number:
        return [(i,dic[i]) for i in sorted(dic, key=dic.__getitem__, reverse=1)[:number]]
    else:
        return [(i,dic[i]) for i in sorted(dic, key=dic.__getitem__, reverse=1)]'''
 
def pred_rating_mean_user_based(user,item,piv,neighbors):
    closer=take_most_commun(piv,user,neighbors)
    rat=0
    if str(closer)=='None':
        return round(piv[:,item].sum()/piv[:,item].count_nonzero(),3)
    else:
        for i in closer.index:
            z=0
            if piv[i,item]!=0:
                rat+=piv[i,item]
                z+=1
        if z==0:
            return round(piv[user,:].sum()/piv[user,:].count_nonzero(),3)
        else:
            return round(rat/z ,3)

def pred_rating_mean_user_based2(user,item,piv,closer):
    rat=0
    if str(closer)=='None':
        return  round(piv[:,item].sum()/piv[:,item].count_nonzero(),3)
    else:
        for i in closer.index:
            z=0
            if piv[i,item]!=0:
                rat+=piv[i,item]
                z+=1
        if z==0:
            return round(piv[user,:].sum()/piv[user,:].count_nonzero(),3)
        else:
            return round(rat/z ,3)

def rmse(y,res):
    l=y.shape[0]
    return sqrt(((y.values-res.values)**2).sum()/l)    

def create_pivot_UB(X,y,Nu,Nb):
    #Create two user-item matrices
    #n_users = X['User_Lab'].unique().shape[0]
    #n_items = X['Book_Lab'].unique().shape[0]
    #print('us',n_users,'it',n_items)
    rat=pd.concat([X,y],axis=1)
    pivot =sparse.lil_matrix((Nu, Nb))
    for line in rat.itertuples():
        pivot[line[1], line[2]] = line[3]+1e-5
    return pivot

def take_item_similarity(item,piv, n):
    cos= cosine_similarity(piv[:,item].T,piv.T).flatten()
    l=pd.Series(cos).sort_values(ascending=0)[:n]
    return l


def pred_item_based_cosine_deviation(user,item,piv,closer):
        m = mean_item(item, piv)
        somma=0
        for it in closer.index:
            somma+= closer[it]*(piv[user,it]-mean_item(it,piv))
        z=somma/closer.sum()
        if z!=z: z=0
        return round(float(m+z),3)
    
def mean_item(elem, pivot):
    print(elem)
    res = pivot[:,elem].sum()/pivot[:,elem].count_nonzero()
    return res

    
def find_similar_item_rated(user,item,piv,n=10):
    col = piv[user,:].nonzero()[1]
    sim={}
    for i in col:
        sim[i]= float(cosine_similarity(piv[:,i].T,piv[:,item].T))
    return [(i,sim[i]) for i in sorted(sim, key=sim.__getitem__, reverse=1)[:n]]

def find_rating_ITBMean(user,similar,piv):
    so=0
    c=0
    for el in similar: 
        if round(el[1],2)!=0.00:
            c+=1
            so+= piv[user,el[0]]
    if c==0:
        return 0
    else:
        return round(so/c, 2)
    
    