#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from mpl_toolkits import mplot3d

xs = np.linspace(-1, 2, 100)
ys = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(xs, ys)

Z = np.zeros(np.shape(X))

# underlying quadratic landscape
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = 2.2*((x - .1)**2 + (y + .5)**2)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# steady state A at (0, 1)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = -1*np.exp(-(x**2 + (y-1)**2)*50)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# steady state A at (0, 1)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = -2*np.exp(-(x**2 + .01*(y-1)**2)*50)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# steady state B at (sqrt(3), 0)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):

        xp = x - np.sqrt(3)
        yp = y
        translated_vec = np.array([xp, yp])

        A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                     [np.sin(np.pi/6), np.cos(np.pi/6)]])

        rot_vec = np.dot(A, translated_vec)
        u, v = rot_vec

        new_Z[j][i] = -2.5*np.exp(-(.02*u**2 + v**2)*8)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# deepen steady state B at (sqrt(3), 0)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = -2.1*np.exp(-((x-np.sqrt(3))**2 + y**2)*10)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# steady state C at (0, -1)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        xp = x
        yp = y + 1
        translated_vec = np.array([xp, yp])

        A = np.array([[np.cos(-np.pi/6), -np.sin(-np.pi/6)],
                     [np.sin(-np.pi/6), np.cos(-np.pi/6)]])

        rot_vec = np.dot(A, translated_vec)
        u, v = rot_vec

        new_Z[j][i] = -2*np.exp(-(.01*u**2 + v**2)*10)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# deepen steady state C at (0, -1)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = -1*np.exp(-(x**2 + (y+1)**2)*10)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# hump at (0, -.5)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = 3*np.exp(-(.1*(x-.3)**2 + .1*(y+.5)**2)*5)
        Z[j][i] = Z[j][i] + new_Z[j][i]


# hump at (0, 0)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = 1*np.exp(-(.1*(x-.4)**2 + .1*(y-.3)**2)*3)
        Z[j][i] = Z[j][i] + new_Z[j][i]

# hump at (0, -1)
new_Z = np.zeros(np.shape(Z))
for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        new_Z[j][i] = 2*np.exp(-(.1*(x+.3)**2 + .1*(y+.8)**2)*5)
        Z[j][i] = Z[j][i] + new_Z[j][i]

## hump at (0, -1)
#new_Z = np.zeros(np.shape(Z))
#for i,x in enumerate(xs):
#    for j,y in enumerate(ys):
#        new_Z[j][i] = 1*np.exp(-(.05*(x+.3)**2 + .01*(y+.7)**2)*5)
#        Z[j][i] = Z[j][i] + new_Z[j][i]
plt.contourf(X, Y, Z, levels=100, cmap='hsv')
plt.colorbar()

plt.savefig('figs/schematic_v1.pdf')

plt.figure()

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        if Z[j][i] > 6.4:
            Z[j][i] = np.nan
        if y - x/np.sqrt(3) < -1.1:
            Z[j][i] = np.nan
        if x < -.5:
            Z[j][i] = np.nan

ax = plt.axes(projection='3d')
ax.axis('off')
ax.contour3D(X, Y, Z, 40, cmap='hsv', linewidths=1)
ax.view_init(elev=70, azim=260)
plt.savefig('figs/schematic_v0.pdf')

