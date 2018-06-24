#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:43:21 2018

@author: parkershankin-clarke
"""
import numpy as np
import math


xa = np.array([1,2,3])
xb = np.array([4,5,6])
p = .3

def get_line_equation(xa,xb,p):
    if p == 0:
        return xa
    if p ==1:
        return xb
    else :
        return p * xa+xb-p*xb


point = get_line_equation(xa,xb,p)

    
def goes_to_xa(xa,xb,point) :
    '''This function returns true if the distance from
    point p to xa is less than a given value'''
    da=0
    for i in range(len(xa)):
        da = math.pow(math.pow(xa[i]-point[i],2),.5) + da

    if da < 1e-6:
        return True 

    
def goes_to_xb(xa,xb,point) :
    '''This function returns true if the distance from
    point p to xb is less than a given value'''
    db=0
    for i in range(len(xa)):
        db = math.pow(math.pow(xa[i]-point[i],2),.5) + db

    if db < 1e-6:
        return True 
