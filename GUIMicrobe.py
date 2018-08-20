#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 20:06:38 2018

@author: parkershankin-clarke
"""
import sys
from tkinter import *
from tkinter import Tk
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import *

root = tk.Tk()

topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)


def getfile():
    """ This function when called opens a file """
    filename = askopenfilename() 
    print(filename)
    
def displayimage():
    """ This function when called displays an image named 'Figure.png' """
    #https://stackoverflow.com/questions/15999661/image-in-tkinter-window-by-clicking-on-button
    novi = Toplevel()
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = 'Figure.png')
    #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1




button1 = tk.Button(bottomFrame,text ='Display attractor network',command = displayimage, height=5, width=20) 
button = tk.Button(topFrame, text="Choose file", command=getfile)

button.pack()
button1.pack()

root.mainloop()
  