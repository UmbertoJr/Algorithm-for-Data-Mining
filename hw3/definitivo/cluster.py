import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
#%%
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1', error_bad_lines=0, index_col=0 )
books['Year-Of-Publication']=books['Year-Of-Publication'].astype(str)
clean_data = books.dropna(axis = 0)
clean_data=clean_data.head(10000)
#%%
def token(text):
    return(text.split("|"))

cv_author=CountVectorizer(max_features=75000,tokenizer=token )
author = cv_author.fit_transform(clean_data["Book-Author"])
author_list = ["Book-Author_"+ i for i in cv_author.get_feature_names()]

cv_publisher=CountVectorizer(max_features=10000,tokenizer=token )
publisher = cv_publisher.fit_transform(clean_data["Publisher"])
publisher_list = ["Publisher_"+ i for i in cv_publisher.get_feature_names()]

cv_year=CountVectorizer(max_features=75000,tokenizer=token )
year = cv_year.fit_transform(clean_data["Year-Of-Publication"])
year_list = ["Year_"+ i for i in cv_year.get_feature_names()]

cluster_data = np.hstack([author.todense(),publisher.todense(),year.todense()])
criterion_list = author_list + publisher_list + year_list
#%%
mod = KMeans(n_clusters=7)
category = mod.fit_predict(cluster_data)
category_dataframe = pd.DataFrame({"category":category},index = clean_data['Book-Title'])

for num in range(7):    
    print(clean_data.ix[list(category_dataframe['category'] == num),['Book-Title','Book-Author','Publisher']].shape[0])