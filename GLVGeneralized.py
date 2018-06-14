#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:13:32 2018

@author: parkershankin-clarke
"""

import barebones_CDI as bb
import numpy as np
from sympy.abc import x
from sympy.matrices import Matrix, Transpose
from numpy.linalg import inv
#
## import data from stein_ic.csv and stein_parameters.csv files
#var_data, ic_data = bb.import_data()
## turn "ugly" unmodified data into "nice" parsed and numpy array data
## mu = growth rates; M = interaction values
## eps = antibiotic susceptibility (you can ignore)
#labels, mu, M, eps = bb.parse_data(var_data)
## save all of the parameters as a list
#param_list = labels, mu, M, eps
## import the fourth initial condition
#ic4 = bb.parse_ic((ic_data, 4), param_list)

############################################################################################################################################################

#  I am making up a test case with a 4x4 M matrix and 4X1 vector to use. 


##print(labels)
##print(M)
##print(ic4)
##print(mu)
#import itertools
#from sympy.matrices import Matrix, Transpose
#import numpy as np
#from numpy import append

#intializations :

#These are test M and mu matrices and vectors respectively :
M = [[3, 4, 7, 8], [4,5,6,9], [3,6,7,10] ,[3,8,8, 10]]
mu = [1,2,3,4]
bbb = [1,2,3,4]

#This is going to generate a list of numbers that will signify the positions of rows in my vector/ matrix:
lst = list(range(1,len(M)+1))

#Intializations of lists that the program uses at somepoints...
combs = []
Matricies = []
perm_set = []
ds_perm_set = []
glst = []
mu1_lst = []
mu2_lst = []
M1_lst = []
M2_lst = []
listarray = []



#Generate combinations of zero positions for rows in matrix/ vector:
# Credit : #https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
for i in range(1, len(lst)+1):
    els = [list(x) for x in itertools.combinations(lst,i)]
    combs.append(els)
    
#This is where the program gets messy...    
#combs is a list of lists (i.e. a superlist.) No matter the size the size combs should ALWAYS have exactly 3 enteries.
#each entry is a different class of solutions.
# The first entry gives zero positions of all zero row combinations where only one row is zero
# The last entry gives zero positions that generate the trivial solution
# The middle entry gives all other zero position combinations 


#generates combinations of matrice/vector for single-zero matricies/vectors :
perm_set_1 = combs[0]
for j in range(len(perm_set_1)):
    M = [[3, 4, 7, 8], [4,5,6,9], [3,6,7, 10] ,[3,8,8, 10]]
    mu = [1,2,3,4]
    perm_set_1_ds = perm_set_1[j]
    perm_set_1_ds_ds = perm_set_1_ds[0]
    M[(perm_set_1_ds_ds)-1] = [0] * len(M)
    mu[(perm_set_1_ds_ds)-1] = 0 * 1
    mu1_lst = mu + mu1_lst
    M1_lst = M + M1_lst 



#generates combinations of matrix/vector for non-single-zero non-trivial matricies/vectors :
perm_set_g1=combs[1:len(combs)-1]
for k in range(len(perm_set_g1)):
    M = [[3, 4, 7, 8], [4,5,6,9], [3,6,7, 10] ,[3,8,8, 10]]
    mu = [1,2,3,4]
    perm_set_g1_ds = perm_set_g1[k]
    for l in range(len(perm_set_g1_ds)):
        perm_set_g1_ds_ds = perm_set_g1_ds[l]
        M = [[3, 4, 7, 8], [4,5,6,9], [3,6,7, 10] ,[3,8,8, 10]]
        mu = [1,2,3,4]
        for m in range (len(perm_set_g1_ds_ds)):
            perm_set_g1_ds_ds_ds = perm_set_g1_ds_ds[m]
            M[(perm_set_g1_ds_ds_ds)-1] = [0] * len(M)
            mu[(perm_set_g1_ds_ds_ds)-1] = 0 * 1
            M2_lst = M + M2_lst   
            mu2_lst = mu + mu2_lst 

#Collects all generated matrices and stores them in a list
le = []
for n in range(0,len(M2_lst),len(M)):
        le = [(M2_lst[n:n+len(M)])] + le
lem = [] 
for n in range(0,len(M1_lst),len(M)):
    lem = [(M1_lst[n:n+len(M)])] + lem

lemo = []
for o in range(0,len(mu1_lst),len(mu)):
    lemo = [(mu1_lst[n:n+len(mu)])] + lemo

lemon = []
for p in range(0,len(mu2_lst),len(mu)):
    lemon = [(mu2_lst[n:n+len(mu)])] + lemon
    
    
# Converts the matricies data types to np.array in order for fixed point analysis.  
# Note that it is important that k and kiwi //  ki and kiw have the same lengths      
k = np.array(le)
ki = np.array(lem)
kiw = np.array(lemo)
kiwi = np.array(lemon)


#Eventually we will plug the above lists in to this part of the code and it will solve for fixed points
# Find fixed points by solving x = A^(-1) b


for q in range(len(k)):
    A = k[q]
    b = kiwi[q]
    IA = inv(A)
#    x = IA.dot(b)


#A = M
#b = mu
#IA = inv(A)
#tb=np.transpose(b)
#x = IA.dot(b)





