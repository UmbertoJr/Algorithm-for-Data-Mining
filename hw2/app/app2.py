# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:46:41 2016

@author: Umbertojunior
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:10:34 2016

@author: Umbertojunior
"""

import tkinter as tk
#from PIL import ImageTk,Image as i
#from alg import *

import final_lib as lb
import pandas as pd
import numpy as np
import importlib
np.set_printoptions(precision=2)
importlib.reload(lb)

#%%
l=open('listdoc.txt','r', encoding='utf-8-sig')
a=l.read()
ListaDocs=eval(a)
l.close()

h=open('words.txt', 'r')
b=h.read()
wordF=eval(b)
h.close()
#%%

tfidf_matrix=lb.matrixTfIdf(ListaDocs.values(),wordF)
#print('Questa è una matrice di dimensioni',tfidf_matrix.shape)

#%%
g= open('invertedInd.txt', 'r')
lec=g.read()
inverted=eval(lec)
g.close()

recipes=pd.read_csv("recipesfinal.csv",sep='\t')

f= open('Namerecip.txt', 'r', encoding='utf-8-sig')
lect=f.read()
Num_Ricetta=eval(lect)
f.close()
#%%
#rank_q=lb.ranklist(input(),inverted,wordF,tfidf_matrix,recipes)
#[(Num_Ricetta[rank_q[i][0]],'#',rank_q[i][0]) for i in range(len(rank_q))][:10]
#recipes.loc[6304]





state = ''
buttons = []
tp=[ 'I eat everything','Vegetarian','lactose intolerant']
#im = i.open('C:/Users/Umbertojunior/Desktop/sapienza.png')

l=list()

def CurSelet(evt):
    value = int(str((list1.get(list1.curselection())))[:4])
    global l
    l=[]
    #print(value)
    l.append(value)

def op(event):
    answ= tk.Tk()
    answ.geometry('1000x450+300+50')
    answ.title(recipes.loc[l[0]][0])
    test= tk.Text(answ,width=100,height=100,font=('times',13))    
    m=recipes.loc[int(l[0])]
    for i in range(1,9):
        test.insert(tk.END,str(m[i])+'\n')
    test.pack()
                
    answ.mainloop()


def callback(event):
    veg=''
    il=''
    if var.get()==1:
        veg='VV '
    if var.get()==2:
        il='IL '
    s=v.get()    
    new = tk.Tk()
    new.title('results')
    new.geometry()
    tit= tk.Label(new, text='please choose what recipies do you want to visualize')
    tit.pack()
    queryFormat,Veg,Il,t=lb.search_app(veg+il+s)
    if Veg:
        searc=tk.Label(new, text='You are in the Vegeterian search engine',fg='blue')
    elif Il:
        searc=tk.Label(new, text='You are in the lactose intorelant search engine',fg='blue')
    elif Il and Veg:
        searc=tk.Label(new, text='You are really in trouble man...this is your recipies search engine!!!',fg='blue')
    else:
        searc=tk.Label(new, text='Welcome to our recipes search engine!!!',fg='blue')
    searc.pack()
    global list1
    list1= tk.Listbox(new,width=60,height=10,font=('times',13))
    response=lb.ranklist_app(veg+il+s,inverted,wordF,tfidf_matrix,recipes)
    for el in response:
        list1.insert(tk.END, str(el[0])+'          '+str(Num_Ricetta[el[0]]))
    list1.bind('<<ListboxSelect>>',CurSelet)
    list1.pack()
    but = tk.Button(new, text="look")
    but.pack()
    but.bind('<Button-1>', op) 
    q=lb.wordsnotin(veg+il+s,inverted,wordF,tfidf_matrix,recipes)
    ovv= tk.Label(new, text=q)
    ovv.pack()
    new.mainloop()
        
def onPress(i):
    global state
    state = i
    for btn in buttons:
        btn.deselect()
    buttons[i].select()
    

app = tk.Tk()
app.geometry('800x300+300+200')
app.title(string='search engine for recipes')



l= tk.Label(app, text='Hello this is the search engine for recipes created by Emmanuele Conti, Valerio Rossini ed Umberto Junior Mele',font='Helvetica',fg="blue", bd= 5,cursor='mouse').pack()

var=tk.IntVar()
for i in range(len(tp)):
    rad = tk.Radiobutton(app, text=str(tp[i]), variable=var, 
                            value=str(i), command=(lambda i=i: onPress(i)) )
    rad.pack()
    buttons.append(rad)

v = tk.StringVar()
contents= tk.Entry(app, textvariable=v, bd =5)
contents.pack()

v.set("")
contents.bind('<Return>', callback) 

b = tk.Button(app, text="find", width=10)
b.pack()
b.bind('<Button-1>', callback)

app.mainloop()

