# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:22:01 2016

@author: Umbertojunior
"""

from tkinter import *
fenetre = Tk()
portee = IntVar()
def effacer():
    rb1.deselect()
    rb2.deselect()
rb1 = Radiobutton(fenetre, text="HOT 1/2", font=("Times",-15,'bold'),
    variable=portee, value=4000)
rb1.grid(row=4, column=1, sticky = E)
rb2 = Radiobutton(fenetre, text="HOT 3", font=("Times",-15,'bold'),
    variable=portee, value=3850)
rb2.grid(row=4, column=2, sticky = E)
# preset rb2
portee.set(3850)
bt1 = Button(fenetre, width=10, height=-10, fg="black",
    font=("Times",-15,'bold'), text="Reset", command=effacer)
bt1.grid(row=6, column=2, pady=10)
fenetre.mainloop()