import requests
from bs4 import BeautifulSoup
import re
import csv
import time

# rlttere = {} # ricette
#dopo la 180 pagina di n si trova l'ultima ricetta
#lettere necessarie 's','n' tuscan pest
ingre=['s','n']
start=time.time()
for rt in ingre:
    print("ingre",rt)
    cnt = requests.get("http://www.bbc.co.uk/food/recipes/search?keywords="+rt)
    count=0
    while str(cnt)!='<Response [200]>' and count<10:
            time.sleep(1)
            count+=1
            cnt = requests.get("http://www.bbc.co.uk/food/recipes/search?keywords="+rt)
    soup = BeautifulSoup(cnt.text , "lxml")
    lst=[]
    for link in soup.find_all('a'):
        if link.get("href").startswith('/food/recipes/search?'):
            lst.append(link.contents[0])
    if lst:
        maxpage=int(lst[-2])
    else:
        maxpage=1
    print('ha',maxpage,"pagine di ricette")
    i=0
    for page in range(1,maxpage+1):
        for i in range(0,748,45):
            if page==i:
                print('pagina numero',page)      
        cnt=requests.get('http://www.bbc.co.uk/food/recipes/search?page='+str(page)+'&keywords='+rt)
        count=0
        while str(cnt)!='<Response [200]>' and count<10:
            time.sleep(1)
            count+=1
            cnt=requests.get('http://www.bbc.co.uk/food/recipes/search?page='+str(page)+'&keywords='+rt)
        soup = BeautifulSoup(cnt.text , "lxml")
        for link in soup.find_all("a"):
            if(link.get("href").startswith('/food/recipes/')
               and not link.get('href')=='/food/recipes/'
               and "search" not in link.get('href')):
                rlettere[link.get('href')] = ""
                i+=1
                tot+=1
    print('questo ingrediente di nome',rt,'ha',i,"ricette")
    print('totale fino ad adesso nel dizionario',len(rlettere))
    print('totale ricette scansionata',tot)
end=time.time()
print('ci sono voluti in tutto',(end-start),'secondi')

f=open('ricelett.txt','w')
for k in rlettere.keys():
    f.write(k+'\n')
f.close()
#=====================================================================


f=open("ricelett.txt")
lista=[]
for row in csv.reader(f, delimiter='\t'):
    lista.append(row[0])

        
def extractBasicInfo(repSoup,itemtype):
    result=""
    for tag in repSoup.find_all(itemprop=itemtype):
        result=tag.contents[0]
    return result
    
def extractDietaryInfo(repSoup):
    result=""
    for tag in repSoup.find_all('p'):
        if tag.get('class')==['recipe-metadata__dietary-vegetarian-text']:
            result=re.sub("[\n| ]*","",tag.contents[0])
    return result
    
def extractMethodInfo(repSoup):
    result=[]
    for tag in repSoup.find_all(itemprop="recipeInstructions"):
        result.append(str(*tag.contents[1].contents))
    return result

def extractIngreInfo(repSoup):
    result=[]   
    for tag in repSoup.find_all(itemprop="ingredients"):  
        L=len(tag.contents)
        stringa=""
        for i in range(L):
            if str(type(tag.contents[i]))=="<class 'bs4.element.Tag'>":
                stringa+=str(*tag.contents[i].contents)
                
            elif str(type(tag.contents[i]))=="<class 'bs4.element.NavigableString'>":
                stringa+=str(tag.contents[i])    
                
        result.append(stringa)
    return result       

def extractAllinfo(rece):
    cnt= requests.get("http://www.bbc.co.uk"+ rece)
    count=0
    while str(cnt)!='<Response [200]>' and count<10:
        time.sleep(1)
        count+=1
        cnt = requests.get("http://www.bbc.co.uk"+ rece)
    pSoup=BeautifulSoup(cnt.text, "lxml")
    contents={}
    contents["name"]=pSoup.title.contents[0][21:]
    contents["prepTime"]=extractBasicInfo(pSoup,"prepTime")
    contents["cookTime"]=extractBasicInfo(pSoup,"cookTime")
    contents["author"]=extractBasicInfo(pSoup,"author")
    contents["recipeYield"]=extractBasicInfo(pSoup,"recipeYield")
    contents["ingredients"]=extractIngreInfo(pSoup)
    contents["instructions"]=extractMethodInfo(pSoup)
    contents["dietaryInfo"]=extractDietaryInfo(pSoup)
    return contents        

def tutto(d):
#==============================================================================
#     with open('ricette.csv', 'w',encoding='utf8', newline='') as csvfile:
#==============================================================================
    with open('ricette.csv', 'a',encoding='utf8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        cont=0
        for k in d:
            a=extractAllinfo(k)
            spamwriter.writerow([a[i] for i in ['name',"author","dietaryInfo",'prepTime',"cookTime","recipeYield","ingredients","instructions"]])
            for i in range(1,11200,15):
                if(cont==i):
                    print('sono arrivato a',cont)
            cont+=1
        csvfile.close()
    return print('Ho fatto')

tutto(lista[8000:])
#==============================================================================
# a=extractAllinfo('coqauvin_83680')
# b=extractAllinfo('bechamelsauce_70004')  
#==============================================================================

    
    
    
    