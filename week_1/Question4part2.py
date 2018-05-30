#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:15:35 2018

@author: parkershankin-clarke
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def microbes(n,t) :

    x = n[0]
    y = n[1]

    a = 1
    b = -1
    c = -.5
    d =  1
    e = -1

    dxdt = x * a + b * math.pow(x,2) + c * y * x
    dydt = y * d + e * math.pow(y,2) + c * y * x

    return [dxdt,dydt]

n0 = [.9,.1]
t = np.linspace(0,15,100)
n = odeint(microbes,n0,t)

x = n[:,0]
y = n[:,1]

plt.plot(t,n[:,0])
plt.plot(t,n[:,1])
#plt.show()

# save figure as pdf
plt.tight_layout()
plt.savefig('gLV_2dim.pdf')






