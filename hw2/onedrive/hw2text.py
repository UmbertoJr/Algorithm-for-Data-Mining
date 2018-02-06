#adm
#%%

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import re
import csv
import numpy as np
import lib as lb
import importlib
importlib.reload(lb)
#nltk.download()
#%%
ric=pd.read_csv("ricettecontitolo.csv",sep='\t')
ric1= ric.drop(ric.columns[[3,4,5]],axis=1)

f=open("ricette.csv","r",encoding='utf-8-sig')
ricette=[]
for row in csv.reader(f, delimiter='\t'):
    if row:
        ricette.append(row[:])
f.close()    
    
with open("ricette.csv","r",encoding='utf-8-sig') as shakes:
    for line in shakes:
        text = shakes.read()
    shakes.close()
    
text = text.lower() #17881536 
len(text)
#%%

from string import punctuation
pun=list(punctuation)
pun.extend(['’','‘','“','”'])
for p in pun:
    text=text.replace(p,' ')
len(text)
text=text.replace('[àåãåâ]','a')
text=text.replace('[ûù]','u')
text=text.replace('[öò]','o')
text=text.replace('[ì]','i')
text=text.replace('[èéê]','e')
text=text.replace('[ñ]','n')
text=text.replace('[ç]','c')
#%%
# # Tokenizzazione
tokens=word_tokenize(text)

#tokenizer = RegexpTokenizer(r'\w+')
#tokens=tokenizer.tokenize(text)

tokens=set(tokens)
len(tokens)  #17868
             
# Pulisco dalle stopword
stop=stopwords.words('english')
stop.extend(['ingredients','instructions'])
doc=[i for i in tokens if i not in stop]
len(doc)  #17739

# Pulisco il testo
words=set()
for word in doc:
    if bool(re.search(r"[^a-zA-Zàòèùìé]",word))!=True:
        words.add(word)
len(words) #16214   

a=set(doc)
b=words
c=sorted(a-b,reverse=True)
 

# Faccio lo stemming   
ps = PorterStemmer()
stem=set()
for w in words:
    stem.add(ps.stem(w)) 
    
#stemmer = PorterStemmer()
#stemmed = stem_tokens(words, stemmer)
a=words
b=stem
c=sorted(b)

len(stem)  #13033
#len(stemmed)

#%%
numeri={}
dict={}
for i,j in zip(range(ric1.shape[0]),range(11224)):
    for q in range(0,11200,50):
        if j==q:
            print('ricetta numero',q)
    numeri[i]=ric1.iloc[i,0] 
    text_i="".join(ricette[i])
    
    for p in pun:
        text_i=text_i.replace(p,'')
        
    tokens_i=word_tokenize(text_i)
    
    doc_i=[i for i in tokens_i if i not in stop]
    
#==============================================================================
#     words_i=[]
#     for word in doc_i:
#         if bool(re.search(r"[^a-zA-Z0-9àòèùìé]",word))!=True:
#             words_i.append(word)
#==============================================================================
    words_i=[]
    for word in doc_i:
        if bool(re.search(r"[^a-zA-Zàòèùìé]",word))!=True:
            words_i.append(word)
    stem_i=[]
    for w in words_i:
        stem_i.append(ps.stem(w))
    dict[i]=" ".join(stem_i)
#%%
#==============================================================================
# dict1={}
# for i in range(ric1.shape[0]): 
#     text_i=" ".join(ricette[i])
#     dict1[i]="".join(text_i)
#==============================================================================
#%%
#==============================================================================
# numeri1={}
# for i in range(ric1.shape[0]):
#     numeri1[i]={}
#     text_i="".join(ricette[i])
#     tokens_i=tokenizer.tokenize(text_i.lower())
#     tokens_i=set(tokens_i)
#     doc_i=[i for i in tokens_i if i not in stop]
#     words_i=set()
#     for word in doc_i:
#         a=re.sub("[^a-zA-Z]","",word)
#         if a!="":
#             words_i.add(a)
#             
#     numeri1[i][ric1.iloc[i,0]]=" ".join(words_i)
#==============================================================================
   
inv,wc=lb.create_index(dict)

#%%
len(motore(['egg','water','salt','oil','cake','bread','onion']),inv)
print(motore(['egg','water','salt','oil','cake','bread','onion']),inv)

#%%
freq=pd.DataFrame(np.zeros((len(ricette),len(stem))))
freq.shape

stemming=list(stem)

for i,j in zip(range(freq.shape[0]),range(5)):
    print(j)
    for q in range(len(stemming)):
        tf=termFrequency(stemming[q],dict[i])
        if tf!=0:
            idf=inverseDocumentFrequency(stemming[q], dict.values)
            print('parola',q,'tf',tf,'idf',idf)
        else:
            idf=1
        freq.iloc[i,q]=tf*idf
        
        
#    Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||

#%%
a='Roast apricot basil apricot pistachio biscotti apricot ice creamJam MartinVegetarian12 hours30 min 1 hourServ 4 568ml1 pint doubl cream 1 vanilla pod split lengthway seed scrape 6 medium freerang egg yolk caster sugar apricot puré 250g9oz plain flour 250g9oz caster sugar tsp bake powder 3 medium freerang egg lightli beaten dri apricot chop medjool date stone remov chop pistachio nut shell weight hazelnut 1 lemon zest caster sugar oz doubl cream 1 tbsp chop basil leav 2 tbsp apricot liqueur apricot brandi 6 apricot halv stone removedFor apricot ice cream place cream vanilla pod seed larg saucepan heat start simmer gentli Set asid ten minut fish vanilla pod Thi rins store jar sugar make vanilla sugar Meanwhil whisk egg yolk sugar larg bowl thick pale Gradual add hot cream mixtur egg whisk time Onc combin return mixtur pan Set low heat cook low heat stir continu thicken Remov heat transfer mixtur bowl set asid cool complet Stir puré chill mixtur fridg complet cold Churn mixtur ice cream machin follow manufactur Store ice cream freezer readi serv For apricot pistachio biscotti preheat oven 180C350FGa 4 line bake tray bake parchment In larg bowl mix flour sugar bake powder Add half beaten egg mix well add half what left mix Add remain egg littl time dough take shape isnt wet You may need egg Work fruit nut lemon zest well combin Divid dough equal six piec Wet hand roll piec sausag shape 5cm2in wide place well apart bake sheet Lightli flatten sausag bake 20 minut goldenbrown Remov oven leav cool harden 10 minut Use serrat knife cut sausag angl slice lay cutsid bake tray You may need anoth bake tray next step batch Return biscuit oven bake eight minut turn slice cook 1015 minut pale golden colour Remov oven cool wire rack When complet cold biscotti store airtight jar week For roast apricot basil increas oven temperatur 200C400FGa 6 Heat ovenproof fri pan medium heat add sugar Cook gentli sugar melt complet turn goldenbrown caramel Care stir cream basil apricot liqueur cook coupl minut smooth Care add apricot cutsid place oven 1530 minut apricot soft caramelis Serv ice cream biscotti roast apricot shallow bowl'
b="apricot pistacchio"
type(b)
termFrequency(b,a)



str(stem)

a=freq.iloc[0:1,:300]


for q in stem:
        stem[q]









