import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import numpy as np
import Lil_lib as lb
import importlib
np.set_printoptions(precision=3)
importlib.reload(lb)
#%%
# We load the datasets
ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1' )
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')
#%%      OFFLINE PART
# Discard from dataset 'users' the users that don't give a rating and
# from dataset 'books' the books not rated
users=users.loc[users['User-ID'].isin(ratings['User-ID'])]
# We take only the users that give a rating greater or egual 100
#==============================================================================
# i=0
# lu=[]
# for user in users['User-ID']:
#     i+=1
#     print(i)
#     if ratings[ratings['User-ID']==user].shape[0]>100:
#         lu.append(user)
#         #ratings.drop(ratings[ratings['User-ID']==user].index)
# lis=pd.Series(lu)   
# lis.to_csv('userover5.csv')     
#==============================================================================
userover=pd.read_csv('userover5.csv',index_col=0,header=None)
ratings=ratings.loc[ratings['User-ID'].isin(userover.values)] 
# We take only the books that have a number of  rating greater or egual 2
books=books.loc[books['ISBN'].isin(ratings['ISBN'])]
#==============================================================================
# i=0
# lb=[]
# for book in books['ISBN']:
#     i+=1
#     print(i)
#     if ratings[ratings['ISBN']==book].shape[0]>1:
#         lb.append(book)
# lis=pd.Series(lb)   
# lis.to_csv('bookover5.csv')     
#==============================================================================
bookover=pd.read_csv('bookover5.csv',index_col=0,header=None)
l=[]
for i in bookover.values:
    l.append(i[0])
#bookover=books.sample(25000,replace=False).ISBN                    
ratings=ratings.loc[ratings['ISBN'].isin(l)]

#%%
#Final Ratings
n_users = ratings['User-ID'].unique().shape[0]
n_items = ratings.ISBN.unique().shape[0]
print( 'Number of users = ' + str(n_users) + ' | Number of books = ' + str(n_items)  )

#piv1=ratings.pivot(index='User-ID',columns='ISBN',values='Book-Rating')
#%%
from sklearn import preprocessing
leB = preprocessing.LabelEncoder()
ratings.insert(2, 'Book_Lab',leB.fit_transform(ratings.ISBN)) 
leU = preprocessing.LabelEncoder()
ratings.insert(2, 'User_Lab',leU.fit_transform(ratings['User-ID'])) 
ratings.replace({0 : 1 ,1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 4, 8 : 4, 9 : 5, 10 : 5},inplace=True)
#ratings.replace({0:1,1:1,2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 2, 7 : 2, 8 : 2, 9 : 2, 10 : 2},inplace=True)
#%%
# Split dataset in X and y(target variable)
importlib.reload(lb)
X=ratings[ratings.columns[:-1]]
X.set_index([np.arange(X.shape[0])],inplace=True)
y=ratings[ratings.columns[-1]]
y.index=np.arange(X.shape[0])

#%%      
# We use K-fold cross validation (5 folds)
kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
print(kf)
#%%     USER-BASED
listaRMSE=[]  
numtest=0
from sklearn.preprocessing import normalize

for train_index, test_index in kf.split(X):
    y_pred={}
    print("TRAIN:", train_index, "TEST:", test_index)
    print("lunghezza train",len(train_index))
    print("lunghezza test",len(test_index))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    pivot_UB=lb.create_pivot_UB(X_train,y_train,n_users,n_items)
    
    #piv=lb.tolgo_media(pivot_UB)    
#==============================================================================
#     rat=pd.concat([X_train[X_train.columns[2:4]],y_train],axis=1)
#     pivot=rat.pivot(index=rat.index,columns='Book_Lab',values='Book-Rating')
#==============================================================================

#==============================================================================
#     mat=pd.DataFrame(pivot_UB.todense())
#     mat1=mat.sub(mat.mean(axis=1), axis=0)
#     piv=sparse.csr_matrix(mat.values)
#==============================================================================
    
    cos_UB=cosine_similarity(pivot_UB,pivot_UB,dense_output=False)
    i=0
    numtest+=1
    tot=len(X_test['User_Lab'].unique())
    for user in X_test['User_Lab'].unique():
        print('User number',i,'on a total of',tot,'in the',numtest,'test-set')
        i+=1
        closer=lb.neighboers(user,cos_UB,10)
        for index,item in X_test[X_test['User_Lab']==user]['Book_Lab'].iteritems():
            y_pred[index]=lb.pred_rating_UB(user,item,closer,pivot_UB)
    res=pd.Series(y_pred)
    listaRMSE.append(lb.rmse(y_test,res))
    print(listaRMSE)
np.mean(listaRMSE)
#%%      ITEM-BASED
listaRMSE=[] 
testnum=0 
for train_index, test_index in kf.split(X):
    y_pred={}
    print("TRAIN:", train_index, "TEST:", test_index)
    print("lunghezza train",len(train_index))
    print("lunghezza test",len(test_index))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    pivot_IB=lb.create_pivot_IB(X_train,y_train,n_users,n_items)    
    cos_IB=cosine_similarity(pivot_IB,pivot_IB,dense_output=False)
    i=0
    testnum+=1
    tot=len(X_test['Book_Lab'].unique())
    for item in X_test['Book_Lab'].unique():
        print('book numero',i,'su un totale di',tot,'in the',testnum,'test-set')
        i+=1
        closer=lb.neighboers(item,cos_IB,10)
        for index,user in X_test[X_test['Book_Lab']==item]['User_Lab'].iteritems():
            y_pred[index]=lb.pred_rating_IB(item,user,closer,pivot_IB)
    res=pd.Series(y_pred)
    listaRMSE.append(lb.rmse(y_test,res))
np.mean(listaRMSE)
#%%      MODEL-BASED

from sklearn import cross_validation as cv
train_data,test_data=cv.train_test_split(ratings,test_size=0.20)


import scipy.sparse as sp
from scipy.sparse.linalg import svds
#==============================================================================
# pivot =sparse.lil_matrix((n_users, n_items))
# for line in train_data.itertuples():
#     pivot[line[3], line[4]] = line[5]
#==============================================================================
pivot_UB=lb.create_pivot_UB(X_train,y_train,n_users,n_items)
test_data=lb.create_pivot_UB(X_test,y_test,n_users,n_items)

#get SVD
k=30
u,s,vt=svds(pivot_UB,k=30)
s=np.diag(s)
s=s[0:k,0:k]
u=u[:,0:k]
vt=vt[0:k,:]

X_pred=np.dot(np.dot(u,s),vt)
piv=sparse.csr_matrix(X_pred)

cos_IB=cosine_similarity(piv,piv,dense_output=False)
i=0
testnum+=1
tot=len(X_test['Book_Lab'].unique())
for item in X_test['Book_Lab'].unique():
    print('book numero',i,'su un totale di',tot,'in the',testnum,'test-set')
    i+=1
    closer=lb.neighboers(item,cos_IB,10)
    for index,user in X_test[X_test['Book_Lab']==item]['User_Lab'].iteritems():
        y_pred[index]=lb.pred_rating_IB(item,user,closer,piv)
res=pd.Series(y_pred)
listaRMSE.append(lb.rmse(y_test,res))


