# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:46:35 2016

@author: Umbertojunior
"""
from tkinter import *

master = Tk()

e = Entry(master)
e.pack()

e.focus_set()

def callback():
    new = Tk()
    new.title(str(e.get()))    

b = Button(master, text="get", width=10, command=callback)
b.pack()

mainloop()