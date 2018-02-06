pimport pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import Lil_lib as L
import importlib 
importlib.reload(L)
importlib.reload(sparse)
#%%

ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1' )
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')
#%%
n_users = users['User-ID'].unique().shape[0]
n_items = ratings.ISBN.unique().shape[0]
print( 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)  )

#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ratings['book_ID']=le.fit_transform(ratings.ISBN)
#%%
#Create two user-item matrices, one for training and another for testing
pivot =sparse.lil_matrix((n_users, n_items))
for line in ratings.itertuples():
    pivot[line[1], line[4]] = line[3]+1.0   
#%%
cos_UserBased=cosine_similarity(pivot,pivot,dense_output=False)

#%%
pivot1 =sparse.lil_matrix((n_items,n_users))
for line in ratings.itertuples():
    pivot1[line[4], line[1]] = line[3]
#%%
cos_ItemBased=cosine_similarity(pivot1,pivot1,dense_output=False)
#%%
for i in pivot[10,:]: print(getcol(i))
