#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:39:39 2018

@author: parkershankin-clarke
"""

import matplotlib.animation as animation
from scipy.integrate import odeint
from numpy import arange
from pylab import *

def BoatFishSystem(state, t):
    a = 1
    b = -1
    c = -1 
    d =  1
    e = -1    
    x,y = state
    
    d_x = x * a + b * math.pow(x,2) + c * y * x
    d_y = y * d + e * math.pow(y,2) + c * y * x
    return [d_x, d_y]

t = arange(0, 20, 0.1)
init_state = [1, 1]
state = odeint(BoatFishSystem, init_state, t)

fig = figure()
xlabel('number of microbes x')
ylabel('number of microbes y')
plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)

def animate(i):
    plot(state[0:i, 0], state[0:i, 1], 'b-')

ani = animation.FuncAnimation(fig, animate, interval=1)
show()
