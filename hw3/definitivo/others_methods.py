import pandas as pd
import numpy as np
from scipy import sparse
from importlib import reload
import my_lib as lb
reload(lb)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
#%% 
ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1')
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')

n_user,n_item=(map(len, [ratings['User-ID'].unique(),ratings['ISBN'].unique()]))
#%% 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ratings['book_ID']=le.fit_transform(ratings.ISBN)
ratings['user_ID']=le.fit_transform(ratings['User-ID'])

'''piv=sparse.lil_matrix((n_user,n_item))
for row in ratings.itertuples():
    piv[row[5],row[4]]=row[3]+ 1e-9'''
       
X,y=ratings[['user_ID','book_ID']],ratings['Book-Rating']
x_train , x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
#%%   
piv= lb.create_pivot_UB(x_train,y_train,n_user,n_item)
#%%     USER BASED MEAN
#res=lb.pred_rating_mean_user_based(100,100,piv,100)
i=1
y_pred={}
gr = x_test.groupby('user_ID')
for user in gr.groups:
        print('user numero',i,'su un totale di',len(x_test['user_ID'].unique()))
        i+=1
        closer=lb.take_most_commun(piv,user,10)
        for index, item in gr.get_group(user)['book_ID'].iteritems():
            y_pred[index]=lb.pred_rating_mean_user_based2(user,item,piv,closer)
res=pd.Series(y_pred)
lb.rmse(y_test,res)

#%%     ITEM BASED COSINE METHOD SLOW

i=1
y_pred={}
gri = x_test.groupby('book_ID')
for book in gri.groups:
        print('user numero',i,'su un totale di',len(gri)
        i+=1
        closer=lb.take_item_similarity(book, piv, 10)
        for index, user in gri.get_group(book)['user_ID'].iteritems():
            y_pred[index]=lb.pred_item_based_cosine_deviation(user,item,piv,closer)
res=pd.Series(y_pred)
lb.rmse(y_test,res)

#%%     ITEM BASED MEAN
i=1
y_pred={}
gri = x_test.groupby('book_ID')
for book in gri.groups:
        print('user numero',i,'su un totale di',len(gri))
        i+=1
        for index, user in gri.get_group(book)['user_ID'].iteritems():
            similar_items = lb.find_similar_item_rated(user,book,piv)
            y_pred[index]=lb.find_rating_ITBMean(user, similar_items, piv)
            
res=pd.Series(y_pred)
lb.rmse(y_test,res)
            