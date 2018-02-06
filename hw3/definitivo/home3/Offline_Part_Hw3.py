import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import numpy as np
import Lil_lib as lb
import importlib
from scipy.sparse.linalg import svds
np.set_printoptions(precision=3)
importlib.reload(lb)
#%%
# We load the datasets
ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1' )
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')
#%%      OFFLINE PART
# Discard from dataset 'users' the users that don't give a rating and
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
users=users.loc[users['User-ID'].isin(userover[1])].copy() 
# We take only the ratings of the user's sample
ratings=ratings.loc[ratings['User-ID'].isin(users['User-ID'])]
#  Discard from dataset 'books' the books not rated
books=books.loc[books['ISBN'].isin(ratings['ISBN'])]
# We take only the books that have a number of rating greater or egual 2
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
books=books.loc[books['ISBN'].isin(list(bookover[1]))]    
# We take only the ratings of the book's sample     
ratings=ratings.loc[ratings['ISBN'].isin(books.ISBN)]
#%%
#Final dataset Ratings
n_users = users['User-ID'].unique().shape[0]
n_items = books.ISBN.unique().shape[0]
print( 'Number of users = ' + str(n_users) + ' | Number of books = ' + str(n_items)  )
#%%
# With LabelEncoder we label all the Books
leB = preprocessing.LabelEncoder().fit(books.ISBN)
ratings.insert(2, 'Book_Lab',leB.transform(ratings.ISBN)) 
# With LabelEncoder we label all the Users
leU = preprocessing.LabelEncoder().fit(users['User-ID'])
ratings.insert(2, 'User_Lab',leU.transform(ratings['User-ID'])) 
# We reduce the rating's range from 0:10 to 1:5 (like NETFLIX)
ratings.replace({0 : 1,1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 4, 8 : 4, 9 : 5, 10 : 5},inplace=True)
# We reduce the rating's range from 0:10 to 1:2 (like or dislike)
# ratings.replace({0 : 1,1 : 1,2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 2, 7 : 2, 8 : 2, 9 : 2, 10 : 2},inplace=True)
#%%     EVALUATION'S PART
# Split dataset in X (train) and y(target variable)
# Reset the index for the cross validation
X=ratings[ratings.columns[:-1]]
X.reset_index(drop=True,inplace=True)
y=ratings[ratings.columns[-1]]
y.reset_index(drop=True,inplace=True)
#%%      
# We use K-fold cross validation (5 folds)
kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
print(kf)
#%%     USER-BASED
# We have a list that will contain the 5 rmse
listaRMSE=[]  
numtest=0
for train_index, test_index in kf.split(X):
    # We create a dict with the predictions
    y_pred={}
    print("TRAIN:", train_index, "TEST:", test_index)
    print("length rows train",len(train_index))
    print("length rows test",len(test_index))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    # We create a sparse matrix for the pivot
    pivot_UB=lb.create_pivot_UB(X_train,y_train,n_users,n_items)
    # We have tried to normalize the data without improvment
    # piv=lb.norm_mean(pivot_UB)    
    # We have chosen, for calculating the distance, the cosine similarity 
    # and we have created the cosine matrix
    cos_UB=cosine_similarity(pivot_UB,pivot_UB,dense_output=False)
    i=0
    numtest+=1
    user_test=X_test['User_Lab'].unique()
    tot=len(user_test)
    for user in user_test:
        print('User number',i,'on a total of',tot,'in the',numtest,'test-set')
        i+=1
        # We find the neighbors of the considerated user 
        closer=lb.neighbors(user,cos_UB,10)
        item_test=X_test[X_test['User_Lab']==user]['Book_Lab']
        for index,item in item_test.iteritems():
            y_pred[index]=lb.pred_rating_UB(user,item,closer,pivot_UB)
    res=pd.Series(y_pred)
    # Calcolation of rmse and append
    listaRMSE.append(lb.rmse(y_test,res))
    print(listaRMSE)
np.mean(listaRMSE)
#%%      ITEM-BASED
# We repeat the same process
listaRMSE=[] 
numtest=0 
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
    numtest+=1
    item_test=X_test['Book_Lab'].unique()
    tot=len(item_test)
    for item in item_test:
        print('Book number',i,'on a total of',tot,'in the',numtest,'test-set')
        i+=1
        closer=lb.neighbors(item,cos_IB,10)
        user_test=X_test[X_test['Book_Lab']==item]['User_Lab']
        for index,user in user_test.iteritems():
            y_pred[index]=lb.pred_rating_IB(item,user,closer,pivot_IB)
    res=pd.Series(y_pred)
    listaRMSE.append(lb.rmse(y_test,res))
np.mean(listaRMSE)
#%%      MODEL-BASED
listaRMSE=[] 
numtest=0 
for train_index, test_index in kf.split(X):
    y_pred={}
    print("TRAIN:", train_index, "TEST:", test_index)
    print("lunghezza train",len(train_index))
    print("lunghezza test",len(test_index))
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    # We create for this test the same pivot of User-Based Model
    pivot_MB=lb.create_pivot_UB(X_train,y_train,n_users,n_items)
    #get SVD, after several tries we have chosen k=100
    u,s,vt=svds(pivot_MB,k=100)
    s=np.diag(s)
    X_pred=np.dot(np.dot(u,s),vt)
    i=0
    numtest+=1
    item_test=X_test['Book_Lab'].unique()
    tot=len(item_test)
    for item in item_test:
        print('Book number',i,'on a total of',tot,'in the',numtest,'test-set')
        i+=1
        user_test=X_test[X_test['Book_Lab']==item]['User_Lab']
        for index,user in user_test.iteritems():
            y_pred[index]=X_pred[user,item]
    res=pd.Series(y_pred)
    listaRMSE.append(lb.rmse(y_test,res))
    print(listaRMSE)
np.mean(listaRMSE)


