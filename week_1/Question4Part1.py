#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:17:18 2018

@author: parkershankin-clarke
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt 
import math

#function that returns dx/dt
def model(x,t):
     a = 1
     b = .5
     dxdt = - a * x + b * math.pow(x,2)
     return dxdt

#intial condition
x0 = 1

#time points 
t = np.linspace(0,20,50)
# solve ODE
x = odeint(model,x0,t)

#plot returns 
plt.plot(t,x)
plt.xlabel('time')
plt.ylabel('x(t)')
plt.show()

