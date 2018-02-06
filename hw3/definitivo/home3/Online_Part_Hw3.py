import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import Lil_lib as lb
import importlib
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import preprocessing
import re
from PIL import Image
import requests
from io import BytesIO

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
userover=pd.read_csv('userover5.csv',index_col=0,header=None)
users=users.loc[users['User-ID'].isin(userover[1])].copy() 
# We take only the ratings of the user's sample
ratings=ratings.loc[ratings['User-ID'].isin(users['User-ID'])] 
#  Discard from dataset 'books' the books not rated
books=books.loc[books['ISBN'].isin(ratings['ISBN'])]
# We take only the books that have a number of rating greater or egual 2
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
#%%      CLUSTER
 #rivedere gli altri file .copy velocizza tutto
# We want to build a cluster in according to countries (new variable)
# Create a features Country
users["Country"]=''
# Insert the label for user 
users.insert(0, 'User_Lab',leU.transform(users['User-ID']))
for i in users.index:
    users.loc[i,"Country"]=users.Location[i].split(',')[-1]
# We create a sparse matrix users x countries where on each row we have 1 or 0 
cv_Country=CountVectorizer(max_features=43)
Country = cv_Country.fit_transform(users["Country"])
cluster_data = Country.todense()
# We have clustered the users in 7 groups
kmeans = KMeans(n_clusters=7).fit(cluster_data)
# and add the labels of cluster on users's database
users['cluster']=kmeans.labels_

#%%
#       ONLINE-PART
search=books[books['Book-Title'].apply(lambda x :bool(re.search('the lord of the rings',x,re.I)))] 
# We create a hypothecial user that love fantasy and mistery his name is Adam
sample_book=books[books.ISBN.isin(['0618002227','0618129030','0553573403','0553212419','1853260339','0786808012','0786808551','0679445358','0439064864','0590353403','0439136350','0345285549','0345314255'])]
print(sample_book['Book-Title'])

#Create two user-item matrices with additional row for Adam
X=ratings[ratings.columns[:-1]]
y=ratings[ratings.columns[-1]]
pivot=lb.create_pivot_UB(X,y,n_users+1, n_items)
# I rated the Adam's books with the mean of the columns
for it in sample_book.ISBN:
    item=leB.transform([it])
    pivot[n_users,item]=int(lb.mean_c(item,pivot[:n_users,:]))  # secondo me bisogna dare voti diversi come input
    print(pivot[n_users,item])

pivot[n_users,]
#%%
# User-Based Recommendation
cos_UB=cosine_similarity(pivot[n_users,],pivot,dense_output=False)
closer_UB=lb.neighbors(0,cos_UB,10)
closer_UB
prev_UB=lb.pred_row_UB(n_users,closer_UB,pivot)

prev_UB=prev_UB.sort_values(ascending=False)
isbn_UB=leB.inverse_transform(prev_UB[:10].index)
sample_book_rac_UB=books[books.ISBN.isin(isbn_UB)]
#%%
# User-Based Recommendation with cluster
users_clu_i=users[users.cluster==4]['User_Lab'] #vuol dire che Adam appartiene al 2 cluster
users_clu_i=np.append(users_clu_i,n_users)
# We have created a reducted pivot with less users and books
book_clu=ratings[ratings['User_Lab'].isin(users_clu_i)].Book_Lab.unique()


pivot_clu=pivot[users_clu_i,:].copy()
pivot_clu=pivot_clu[:,book_clu].copy()

cos_UB_clu=cosine_similarity(pivot[n_users,book_clu],pivot_clu,dense_output=False)
closer_clu=lb.neighbors(0,cos_UB_clu,10)
closer_clu

prev_clu=lb.pred_row_UB(closer_clu.shape[0],closer_clu,pivot_clu)

prev_clu=prev_clu.sort_values(ascending=False)
isbn_clu=leB.inverse_transform(prev_clu[:10].index)
sample_book_rac_clu=books[books.ISBN.isin(isbn_clu)]
#%%
# Model-based Recommendation
# Get SVD for the pivot matrix
u,s,vt=svds(pivot,k=100)
s=np.diag(s)
X_pred=np.dot(np.dot(u,s),vt)
prev_MB=pd.Series(X_pred[-1,:])

prev_MB=prev_MB.sort_values(ascending=False)
isbn_MB=leB.inverse_transform(prev_MB[:10].index)
sample_book_rac_MB=books[books.ISBN.isin(isbn_MB)]
#%%
#item-Based
pivot_IB=pivot.T
pivot_IB[:,n_users]

items=leB.transform(sample_book.ISBN)
cos_mat=pd.Series()
for b in items:
    cos_IB=cosine_similarity(pivot_IB[b,:],pivot_IB,dense_output=False)
    negh=lb.neighbors(0,cos_IB,1000)# possiamo creare una fun che taglia i vettori appena vede degli zeri
    cos_mat=pd.concat([cos_mat,negh])
# We remove duplicates index
cos_mat=cos_mat.groupby(cos_mat.index).first()    
prev_IB=cos_mat.sort_values(ascending=False)
isbn_IB=leB.inverse_transform(prev_IB[:10].index)
sample_book_rac_IB=books[books.ISBN.isin(isbn_IB)]
#%%
i=0
for b in isbn_IB:
    book=books[books.ISBN==b]
    if i<=3:
        resp = requests.get(book.iloc[0,7])
        img = Image.open(BytesIO(resp.content))
        print('Book image',img)
        i+=1
    elif i>3 & i<=6:
        resp = requests.get(book.iloc[0,6])
        img = Image.open(BytesIO(resp.content))
        print('Book image',img)
    else:
        resp = requests.get(book.iloc[0,5])
        img = Image.open(BytesIO(resp.content))
        print('Book image',img)
    i+=1

