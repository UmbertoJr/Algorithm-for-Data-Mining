
# coding: utf-8

# In[1]:

import pandas as pd
import operator

ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
books = pd.read_csv('BX-Books_mod.csv',sep=';', encoding='latin-1',index_col=0 )
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1',index_col=0)


rat_sort= ratings.sort(['User-ID'])

val=rat_sort.set_index('User-ID').index.value_counts().to_dict()
# maneggio val dict

dic= {}
s=0
i=0
dic[i]=[]
for k,v in sorted(val.items(), key=operator.itemgetter(0)):
    if s < 70000:
        dic[i].append(k)
        s+=v
    else:
        s=0
        i+=1
        dic[i]=[]

'''
s=0
for i in range(15):
    s+=len(dic[i])
    print(len(dic[i]))
'''
#da lavorare
ratings1 = rat_sort[rat_sort['User-ID']<=dic[0][-1]]

piv= ratings1.pivot(index='User-ID', columns='ISBN', values='Book-Rating')


from sklearn.metrics.pairwise import cosine_similarity

