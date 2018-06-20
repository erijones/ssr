#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:10:43 2018

@author: parkershankin-clarke
"""

import numpy as np
import itertools
import barebones_CDI as bb
import numpy as np
from sympy.abc import x
from sympy.matrices import Matrix, Transpose
from numpy.linalg import inv
from sympy import *


# import data from stein_ic.csv and stein_parameters.csv files
var_data, ic_data = bb.import_data()
# turn "ugly" unmodified data into "nice" parsed and numpy array data
# mu = growth rates; M = interaction values
# eps = antibiotic susceptibility (you can ignore)
labels, mu, M, eps = bb.parse_data(var_data)
# save all of the parameters as a list
param_list = labels, mu, M, eps
# import the fourth initial condition
ic4 = bb.parse_ic((ic_data, 4), param_list)


##M = np.random.rand(2,2)
#M =np.array([[-.004, -.003],[-.003, -.001]])
##mu = np.arange(1, 5)
#mu = np.array([.2,.1])
##x = np.array([1,2,3,4]) 
N = len(mu)

combs = []
listarray = []
lst = list(range(len(M)))

# Credit : #https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
for i in range(1, len(lst)+1):
    els = [list(x) for x in itertools.combinations(lst,i)]    
    combs.extend(els)





fps = []
fixedpointslist = []
for comb in combs:
    temp_M = M[comb, :][:, comb]
    temp_mu = mu[comb]
    temp_fp = np.linalg.solve(temp_M, -temp_mu)
    full_fp = np.zeros(N)
    for i,elem in enumerate(comb):
        full_fp[elem] = temp_fp[i]
        print(full_fp)
        fixedpointslist = [full_fp] + fixedpointslist
        
for elem in fixedpointslist :


    x = elem

    def integrand(x, t, mu, M):
        """ Return N-dimensional gLV equations """
        dxdt = ( np.dot(np.diag(mu), x)
                 + np.dot(np.diag(x), np.dot(M, x)) )
        return dxdt
    
    def jacobian(x, t, mu, M):
        """ Return jacobian of N-dimensional gLV equation at steady state x """
        N = len(x)
        jac = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i is j:
                    val = mu[i] + np.dot(M, x)[i] + M[i,i]*x[i]
                    jac[i, j] = val
                else:
                    val = x[i]*M[i,j]
                    jac[i, j] = val
        return jac
    
    def get_stability(x, mu, M):
        """ Evaluate stability of steady state x """
        jac = jacobian(x, 0, mu, M)
        eig_vals, eig_vecs = np.linalg.eig(jac)
    
        if all(eig_vals < 0):
            print(x)
            print( 'is stable!')
        else :
            print('unstable!')
        
        
        

 

    output = integrand(x, 0, mu, M)
    get_stability(x, mu, M)










