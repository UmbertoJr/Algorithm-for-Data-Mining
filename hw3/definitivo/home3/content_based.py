import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import Lil_lib as lb
import importlib
from operator import itemgetter 
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
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
n_users = ratings['User-ID'].unique().shape[0]
n_items = ratings.ISBN.unique().shape[0]
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
#%%
# We use the column of label books in way that we can make more simpler the identification of the books
books.insert(0, 'Book_Lab',leB.transform(books.ISBN))
# We sort them by Book_Lab
books=books.sort_values(by='Book_Lab')
books['Year-Of-Publication']=books['Year-Of-Publication'].astype(str)
#%%
# We tokenize taking only the word that .......
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
# We take for reaching the final list of tokens only the author publisher and year of publication
pos=[3,4,5]
list_books={}
wordF=[]
for i in books.Book_Lab:
    print(i)
    text=" ".join(list(itemgetter (*pos)(books.iloc[i,:]))).lower()
    # Remove the punctuation's sign   
    for p in punctuation:
        text=text.replace(p,' ')   
    tokens=tokenizer.tokenize(text)
    # Every time we add the tokens at the list
    wordF.extend(tokens)
    list_books[i]=" ".join(tokens)
#%%
print('Write here a briefly desciption about you')
text=input().lower()
# Remove the punctuation's sign  
for p in punctuation:
    text=text.replace(p,' ')
# We add the tokens of the user's description at the final list  
tokens=tokenizer.tokenize(text)
wordF.extend(tokens)
list_books[40224]=" ".join(tokens)
# Remove duplicates
wordF=list(set(wordF))
#%%
# We build the tdfif matrix 
tfidf_vectorizer = TfidfVectorizer(vocabulary=wordF,norm='l2',smooth_idf=False)
tfidf_matrix=tfidf_vectorizer.fit_transform(list(list_books.values()))
#%%   
# We make a prediction
cos_vec=cosine_similarity(tfidf_matrix[-1,:],tfidf_matrix,dense_output=False)
negh=lb.neighbors(0,cos_vec,cos_vec.shape[1])
prev_CB=negh.sort_values(ascending=False)
book_CB=isbn_CB=leB.inverse_transform(prev_CB[:10].index)
sample_book_rac_CB=books[books.ISBN.isin(isbn_CB)]
     
