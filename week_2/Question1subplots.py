#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 06:47:01 2018

@author: parkershankin-clarke
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def microbes(n,t) :

    x = n[0]
    y = n[1]

    a =  1
    b =  1
    c = .5
    d =  1
    e =  1
    f = .5

    dxdt = x * a + b * math.pow(x,2) + c * y * x
    dydt = y * d + e * math.pow(y,2) + f * y * x

    return [dxdt,dydt]

n0 = [.1,.9]
t = np.linspace(0,.5,1000000)
n = odeint(microbes,n0,t)

x = n[:,0]
y = n[:,1]


def microbes2(n2,t2) :

    x2 = n2[0]
    y2 = n2[1]

    a2 = 1
    b2 = 1
    c2 = .5
    d2 =  1
    e2 =  1
    
    dxdt2 = x2 * a2 + b2 * math.pow(x2,2) + c2 * y2 * x2
    dydt2 = y2 * d2 + e2 * math.pow(y2,2) + c2 * y2 * x2

    return [dxdt2,dydt2]

n02 = [.5,.5]
t2 = np.linspace(0,.5,1000000)
n2 = odeint(microbes2,n02,t2)

x2 = n2[:,0]
y2 = n2[:,1]


def microbes3(n3,t3) :

    x3 = n3[0]
    y3 = n3[1]

    a3 = 1
    b3 = 1
    c3 = 1.5
    d3 =  1
    e3 =  1
    
    dxdt3 = x3 * a3 + b3 * math.pow(x3,2) + c3 * y3 * x3
    dydt3 = y3 * d3 + e3 * math.pow(y3,2) + c3 * y3 * x3

    return [dxdt3,dydt3]

n03 = [.1,.9]
t3 = np.linspace(0,.5,1000000)
n3 = odeint(microbes3,n03,t3)

x3 = n3[:,0]
y3 = n3[:,1]

def microbes2(n4,t4) :

    x4 = n4[0]
    y4 = n4[1]

    a4 = 1
    b4 = 1
    c4 = 1.5
    d4 =  1
    e4 =  1
    
    dxdt4 = x4 * a4 + b4 * math.pow(x4,2) + c4 * y4 * x4
    dydt4 = y4 * d4 + e4 * math.pow(y4,2) + c4 * y4 * x4

    return [dxdt4,dydt4]

n04 = [.5,.5]
t4 = np.linspace(0,.5,1000000)
n4 = odeint(microbes2,n02,t2)

x4 = n4[:,0]
y4 = n4[:,1]



# 
plt.subplot(221)
plt.plot(t,n[:,0])
plt.plot(t,n[:,1])
plt.yscale('linear')
plt.title('Mab= .5 ic= .1,.9')
plt.grid(True)



#  
plt.subplot(222)
plt.plot(t2,n2[:,0])
plt.plot(t2,n2[:,1])
plt.yscale('linear')
plt.title('Mab = .5 ic = .5,.5')
plt.grid(True)
plt.subplots_adjust(hspace=.5)


# 
plt.subplot(223)
plt.plot(t3,n3[:,0])
plt.plot(t3,n3[:,1])
plt.yscale('linear')
plt.title('Mab 1.5 intial conditions .1,.9')
plt.grid(True)

# 
plt.subplot(224)
plt.plot(t4,n4[:,0])
plt.plot(t4,n4[:,1])
plt.yscale('log')
plt.title('Mab and Mba = 1.5 ic = .5,.5')
plt.grid(True)


## https://matplotlib.org/gallery/pyplots/pyplot_scales.html#sphx-glr-gallery-pyplots-pyplot-scales-py


