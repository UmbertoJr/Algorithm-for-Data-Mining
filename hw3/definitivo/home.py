import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import Lil_lib as lb
import importlib
np.set_printoptions(precision=3)
importlib.reload(lb)
#%%
ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1' )
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')
#%%
users=users.loc[users['User-ID'].isin(ratings['User-ID'])]
books=books.loc[books['ISBN'].isin(ratings['ISBN'])]
#%%
#==============================================================================
# i=0
# lu=[]
# for user in users['User-ID']:
#     i+=1
#     print(i)
#     if ratings[ratings['User-ID']==user].shape[0]>1000:
#         lu.append(user)
#         #ratings.drop(ratings[ratings['User-ID']==user].index)
# lis=pd.Series(lu)   
# lis.to_csv('userover5.csv')     
#==============================================================================
userover=pd.read_csv('userover5.csv',index_col=0,header=None)
#==============================================================================
# i=0
# lb=[]
# for book in books['ISBN']:
#     i+=1
#     print(i)
#     if ratings[ratings['ISBN']==book].shape[0]>1000:
#         lb.append(book)
#         #ratings.drop(ratings[ratings['User-ID']==user].index)
# 
# lis=pd.Series(lb)   
# lis.to_csv('bookover5.csv')     
# bookover=pd.read_csv('bookover5.csv',index_col=0,header=None)
#==============================================================================
bookover=books.sample(30000,replace=False).ISBN
#%%
ratings=ratings.loc[ratings['User-ID'].isin(userover.values)] 
ratings=ratings.loc[ratings['ISBN'].isin(bookover.values)]
#Final Ratings
n_users = ratings['User-ID'].unique().shape[0]
n_items = ratings.ISBN.unique().shape[0]
print( 'Number of users = ' + str(n_users) + ' | Number of books = ' + str(n_items)  )

#%%
from sklearn import preprocessing
leB = preprocessing.LabelEncoder()
ratings.insert(2, 'Book_Lab',leB.fit_transform(ratings.ISBN)) 
leU = preprocessing.LabelEncoder()
ratings.insert(2, 'User_Lab',leU.fit_transform(ratings['User-ID'])) 

#%%      USER-BASED
#Create two user-item matrices
pivot =sparse.lil_matrix((n_users, n_items))
for line in ratings.itertuples():
    pivot[line[3], line[4]] = line[5]+1e-5
pivot
#%%
cos_UserBased=cosine_similarity(pivot,pivot,dense_output=False)
cos_UserBased
#cos_UserBasedDense=cos_UserBased.todense()
 #%%      
closer=lb.neighboers(0,cos_UserBased,10)
closer
lb.mean(0,pivot)
z=lb.pred_rating_UB(0,1,closer,pivot)
z
zp=lb.pred_row_UB(0,closer,pivot)
zp.sort(ascending=False)
book=zp[:10].index
isbn=leB.inverse_transform(book)
predic=books[books.ISBN.isin(isbn)]
predic['Book-Title']
#%%
importlib.reload(lb)
from sklearn.model_selection import KFold
X=ratings[ratings.columns[:-1]]
X.set_index([np.arange(X.shape[0])],inplace=True)
y=ratings[ratings.columns[-1]]
y.index=np.arange(X.shape[0])
kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
print(kf)
listaRMSE=[]  
for train_index, test_index in kf.split(X):
    y_pred={}
    print("TRAIN:", train_index, "TEST:", test_index)
    print("lunghezza train",len(train_index))
    print("lunghezza test",len(test_index))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    piv=lb.create_pivot_UB(X_train,y_train,n_users,n_items)
    cos_UB=cosine_similarity(piv,piv,dense_output=False)
    i=0
    for user in X_test['User_Lab'].unique():
        print('user numero',i,'su un totale di',len(X_test['User_Lab'].unique()))
        i+=1
        closer=lb.neighboers(user,cos_UB,10)
        for index,item in X_test[X_test['User_Lab']==user]['Book_Lab'].iteritems():
            y_pred[index]=lb.pred_rating_UB(user,item,closer,piv)
    res=pd.Series(y_pred)
    listaRMSE.append(lb.rmse(y_test,res))
np.mean(listaRMSE)



#%%      ITEM-BASED
pivot1 =sparse.lil_matrix((n_items,n_users))
for line in ratings.itertuples():
    pivot1[line[4], line[3]] = line[5]#non sommiamo perrchè troppi elementi modificare 
#%%
cos_ItemBased=cosine_similarity(pivot1,pivot1,dense_output=False)  
#%%
closer=lb.neighboers(56,cos_ItemBased,10)
closer
lb.mean(56,pivot1)
z=lb.pred_rating_IB(56,0,closer,pivot1)
z
zp=lb.pred_row(0,closer,pivot)
zp.sort(ascending=False)
book=zp[:10].index
isbn=leB.inverse_transform(book)
predic=books[books.ISBN.isin(isbn)]
predic['Book-Title']
