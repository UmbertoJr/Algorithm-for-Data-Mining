# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:51:51 2016

@author: Umbertojunior
"""

from tkinter import *      
state = ''
buttons = []
     
def onPress(i):
    global state
    state = i
    for btn in buttons:
        btn.deselect()
    buttons[i].select()
    
root = Tk()
for i in range(10):
    rad = Radiobutton(root, text=str(i), 
                            value=str(i), command=(lambda i=i: onPress(i)) )
    rad.pack(side=LEFT)
    buttons.append(rad)

root.mainloop()

print(state)
