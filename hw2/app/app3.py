
import tkinter as tk
#from PIL import ImageTk,Image as i
#from alg import *
import re
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
a=recipes.set_index("\ufeffname")

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

#==============================================================================
# def op(event):
#     answ= tk.Tk()
#     answ.geometry('1000x450+300+50')
#     answ.title(l[0])
#     test= tk.Text(answ,width=100,height=100,font=('times',13))
#     test.insert(tk.END,l[0]+'\n')
#     for i in range(8):
#         test.insert(tk.END,str(a.loc[l[0]][i])+'\n')
#     test.pack()
#                 
#     answ.mainloop()
#==============================================================================
re.sub("[\[' |'\]]",'',str(recipes.loc[8827][6]))


def op(event):
    answ= tk.Tk()
    answ.geometry('1000x450+300+50')
    ric=recipes.loc[int(l[0])]
    answ.title(ric[0])
    test= tk.Text(answ,width=100,height=100,font=('times',13))
    test.insert(tk.END,'Author : '+ric[1]+'\n')
    test.insert(tk.END,'\n'+'Dietary : '+str(ric[2])+'\n')
    test.insert(tk.END,'Lactose Intolerant : '+str(ric[8])+'\n')
    test.insert(tk.END,'Preparation Time : '+str(ric[3])+'\n')
    test.insert(tk.END,'Cooking Time : '+str(ric[4])+'\n')
    test.insert(tk.END,'Serves : '+str(ric[5])+'\n')
    a=re.sub("',' ",' \n',str(ric[6]))
    #a=re.sub("[\[' |'\]]",'',str(ric[6]))
    test.insert(tk.END,'\n'+'Ingredients : '+'\n'+a+'\n')
    b=re.sub("',' ",' \n',str(ric[7]))
    #b=re.sub("[\[' |'\]]",'',str(ric[7]))
    test.insert(tk.END,'\n'+'Method : '+'\n'+b+'\n')
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
        list1.insert(tk.END,str(el[0])+"    "+str(Num_Ricetta[el[0]]))
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

