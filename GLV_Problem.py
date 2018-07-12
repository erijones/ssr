#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:10:43 2018
@author: parkershankin-clarke
"""

import numpy as np
import itertools
import barebones_CDI as bb
from scipy.integrate import odeint
from itertools import permutations



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

def get_stability(x, mu, M, almost_stable=None, substability=False):
    """ Evaluate stability of steady state x. If stable, returns True; if
    unstable, returns False. If 'substability' is True, we only consider the
    stability of the fixed point in the directions where x is non-zero. If
    'almost_stable' is not None, check if there are almost_stable number of
    positive eigenvalues or fewer."""
    jac = jacobian(x, 0, mu, M)
    # consider reduced jacobian that only considers nonnegative populations
    if substability:
        mask = [i for i in range(len(x)) if abs(x[i]) > 0]
        jac = jac[:, mask][mask, :]
    eig_vals, eig_vecs = np.linalg.eig(jac)

    # check how many directions ss is unstable in
    # if stable, num_unstable_dirs = 0
    num_unstable_dirs = sum(eig_vals > 0)
    if num_unstable_dirs == 0:
        return True
    if almost_stable:
        if num_unstable_dirs <= almost_stable:
            return num_unstable_dirs
    return False


def get_stein_parameters():
    """ Read in parameters M and mu (NxN and Nx1 np.arrays) and initial
    conditions (there are 9) that were generated in Stein et al., 2013"""
    # import data from stein_ic.csv and stein_parameters.csv files
    var_data, ic_data = bb.import_data()
    # turn "ugly" unmodified data into "nice" parsed and numpy array data
    # mu = growth rates; M = interaction values
    # eps = antibiotic susceptibility (you can ignore)
    labels, mu, M, eps = bb.parse_data(var_data)
    # save all of the parameters as a list
    param_list = labels, mu, M, eps
    # import all initial conditions
    # ic4 = bb.parse_ic((ic_data, 4), param_list)
    ics = np.array([bb.parse_ic((ic_data, i), param_list) for i in range(9)])
    return param_list, ics


def get_all_steady_states(mu, M):
    """ Given growth rates mu and interaction values M, calculate all steady
    states of the gLV equations dx_i/dt = x_i(mu_i + \sum_{j=1}^N M_ij x_j)"""
    N = len(mu)
    lst = list(range(N))
    combs = []

    # Credit: #https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
    # calculate the power set of lst
    for i in range(N+1):
        els = [list(x) for x in itertools.combinations(lst,i)]
        combs.extend(els)

    fixedpointslist = []
    for comb in combs:
        # generate subset matrices/vectors of M and mu that correspond to gLV
        # solutions where some populations are 0
        temp_M = M[comb, :][:, comb]
        temp_mu = mu[comb]
        # solve the the fixed points where some populations are 0
        temp_fp = np.linalg.solve(temp_M, -temp_mu)
        full_fp = np.zeros(N)
        for i,elem in enumerate(comb):
            full_fp[elem] = temp_fp[i]
        fixedpointslist.append(full_fp)

    fixedpointslist = np.array(fixedpointslist)
    return fixedpointslist

#def get_stein_steady_states(mu, M):


def get_nonnegative_fixedpoints(fps):
    """ Returns fixed points that are nonnegative """

    fps_positive_list = []
    for fp in fps:
        if all(fp >= -1e-8):
            fps_positive_list.append(fp)
    return np.array(fps_positive_list)

def get_param_line_equation(xa, xb):
    """ Print the equation for the line that goes through xa and xb """
    return print('The parameterization of the line is {}t + {} '.format(xb - xa, xa))

def get_point_on_line(xa, xb, p,mu,M):
    """ Return a point along the line that connects xa and xb, parameterized by
    p, where 0 <= p <= 1. Note p=0 returns xa, while p=1 returns xb. """
    return (1-p)*xa + p*xb

def goes_to_xa(xa, xb, p,mu,M):
    """ This function checks to see if the point parameterized by p converges
    to xa """
    point = get_point_on_line(xa, xb, p,mu,M)
    t = np.linspace(0, 1000)
    sol = odeint(integrand, point, t, args=(mu, M))
    final_sol = sol[-1]
    #print(p)
    #print('final: {}'.format(final_sol))
    #print('ssa  : {}'.format(xa))
    #print('ssb  : {}'.format(xb))
    #print()
    if np.linalg.norm(final_sol - xa) < .001:
        return True
    else:
        return False

def goes_to_xb(xa, xb, p,mu,M):
    """ This function checks to see if the point parameterized by p converges
    to xb """
    point = get_point_on_line(xa, xb, p,mu,M)
    t = np.linspace(0, 1000)
    sol = odeint(integrand, point, t, args=(mu, M))
    final_sol = sol[-1]
    if np.linalg.norm(final_sol - xb) < .001:
        return True
    else:
        return False

def get_separatrix_point(xa, xb, mu, M, num_points=101):
    """ This function find the separatrix between the fixed points xa and xb.
    If their basins of attraction agree, i.e. separatrix_xa and separatrix_xb
    are very close, it returns a tuple of their average. If the basins of
    attraction do not agree, i.e. separatrix_xa and separatrix_xb are
    different, it returns a tuple (separatrix_xa, separatrix_xb) to
    differentiate the basins of attraction. """
    flag_xa = True
    flag_xb = True
    verbose = False
    for p in np.linspace(0, 1, num_points):
        flag_xa = goes_to_xa(xa, xb, p,mu,M)
        if flag_xa is True:
            if verbose:
                print('for p={}:  went to xa is {}'.format(p, flag_xa))
        else:
            separatrix_xa = p
            break
    for p in np.linspace(0, 1, num_points)[::-1]:
        flag_xb = goes_to_xb(xa, xb, p,mu,M)
        if flag_xb is True:
            if verbose:
                print('for p={}:  went to xb is {}'.format(p, flag_xb))
        else:
            separatrix_xb = p
            break

    verbose = False
    if abs(separatrix_xa - separatrix_xb) <= 2/(num_points - 1):
        separatrix = ((separatrix_xa ) + (separatrix_xb)) / 2.0
        if verbose:
            print('separatrix between xa and xb occurs at p={:.5}'.format(separatrix))
        return separatrix, separatrix
    else:
        if verbose:
            print('basin of attraction for xa ends at p={:.5}'.format(separatrix_xa))
            print('basin of attraction for xb ends at p={:.5}'.format(separatrix_xb))
        return separatrix_xa, separatrix_xb
    
    
def SSR(xa,xb,mu,M):  
    """This function performs a steady state reduction by taking in the relevant parameters, then performing the relevant operations,
     and finally returning the steady state reduced forms of the parameters "nu" and "L" """
    
    nu = np.array([np.dot(xa, mu),
                  np.dot(xb, mu)])
#  np.dot(A, np.dot(B, C))
# np.dot(xa.T, np.dot(M,xa))
    L = np.array([[np.dot(xa.T,np.dot(M,xa)), np.dot(xa.T,np.dot(M,xb))],
                 [np.dot(xb.T,np.dot(M,xa)), np.dot(xb.T,np.dot(M,xb))]])
    return nu,L
compare_lists = []

def get_stein_steady_states(stein_values,steady_state_2_list):
   """This function imports  values from the dictionary that contains "stein's steady states" 
   and  also imports steady states with up to 2 unstable directions (almost_stable=2). This 
   code outputs matching steady states between stein's set and the calculated set""" 
   final_list = []
   iterations_list =  list(itertools.product(stein_values,steady_state_2_list))
   for i in range(len(iterations_list )):
       compare_lists = iterations_list[i]
       if np.linalg.norm(compare_lists[0] - compare_lists[1]) < .001:
           final_list = final_list + [compare_lists[0]]
   return final_list



## MAIN FUNCTION

param_list, ics = get_stein_parameters() 
labels, mu, M, eps = param_list
fps = get_all_steady_states(mu, M)
fps = get_nonnegative_fixedpoints(fps)
fp_list = []
num_stable_fps = 0

#for fp in fps:
#    # make sure all fixed points are actually fixed points
#    output = integrand(fp, 0, mu, M)
#    assert(all(abs(output) < 1e-6))
#    
#    is_stable = get_stability(fp, mu, M, almost_stable=0, substability=False)
#    if is_stable:
#        verbose = False
#        if verbose:
#            if is_stable is True:
#                print('hello')
##                print('{} is stable'.format(fp, is_stable))
#            else:
#                print('hello')
##                print('{} is unstable in {} direction'.format(fp, is_stable))
##            print()
#        num_stable_fps += 1
#        fp_list.append(fp)
#fp_list = np.array(fp_list)
##print('there were {} stable fps out of {} total positive cases'.format(num_stable_fps, len(fps)))
#
fp_list2 = []
steady_state_2_list = []
counter = 0
for fp in fps:
    # make sure all fixed points are actually fixed points
    output = integrand(fp, 0, mu, M)
    assert(all(abs(output) < 1e-6))
    stein_stable = get_stability(fp, mu, M, almost_stable=2, substability=False)
    if stein_stable:
        verbose = False
        if stein_stable is True:
            steady_state_2_list = [fp] + steady_state_2_list
            counter += 1
            if verbose:
                print('{} is stable'.format(fp,stein_stable))
                print(counter)
        else:
            steady_state_2_list = [fp] + steady_state_2_list
            counter += 1
            if verbose:
                print('the stein {} is unstable in {} direction'.format(fp, stein_stable))
                print(counter)
        num_stable_fps += 1
        fp_list2.append(fp)
fp_list2 = np.array(fp_list2)
print('there were {} stein stable fps out of {} total cases'.format(num_stable_fps, len(fps)))

test_call = bb.get_all_ss()
stein_stable = get_stability(fp, mu, M, almost_stable=2, substability=False)

stein_values = list(test_call.values())

#xa = fp_list[0]; xb = fp_list[1]  
#sep_xa, sep_xb = get_separatrix_point(xa, xb,M,mu, 101)
#call = SSR(xa,xb,mu,M)
#print(sep_xa, sep_xb)
#This returns Stein's steady states
stein_steady_states = get_stein_steady_states(stein_values, steady_state_2_list)
itertools.permutations(stein_steady_states,2)

#returns all iterations of the possible combinations of Stein's Steady States
combos = list(itertools.combinations(range(5), 2))
for i,j in combos:
    ssa = stein_steady_states[i]
    ssb = stein_steady_states[j]
    temp_separatrix = get_separatrix_point(ssa, ssb,mu,M, num_points=101)
    nu,L = SSR(ssa,ssb,mu,M)
    get = get_separatrix_point(np.array([0,1]), np.array([1,0]), nu, L, num_points=101)
    print(' for the 11-D case the separatrix of ss{} and ss{} occurs at {}'.format(i, j, temp_separatrix))
    print(' for the 2-D case the separatrix of ss{} and ss{} occurs at {}'.format(i, j, get))


#test_call = bb.get_all_ss()
#stein_values = list(test_call.values())
#stein_steady_states = get_stein_steady_states(stein_values, steady_state_2_list)
#
#
#ssa = stein_steady_states[1]
#ssb = stein_steady_states[2]
#temp_separatrix = get_separatrix_point(ssa, ssb,mu,M, num_points=101)
#nu,L = SSR(ssa,ssb,mu,M)
#get = get_separatrix_point(np.array([0,1]), np.array([1,0]), nu, L, num_points=101)
#print('separatrix of ss{} and ss{} occurs at {}'.format(1, 2, temp_separatrix))
    
#mu = np.array([0.36807, 0.31023, 0.3561 , 0.54006, 0.70898, 0.47064, 0.2297 ,
#       0.83005, 0.32367, 0.29075, 0.39181])
#    
#M = np.array([[-0.20516,  0.0984 ,  0.16739, -0.16461, -0.14341,  0.01988,
#        -0.51535, -0.39162, -0.26894,  0.00889,  0.34635],
#       [ 0.06212, -0.10489, -0.04301, -0.15466, -0.1872 ,  0.02703,
#        -0.45919, -0.41388, -0.19657,  0.02208,  0.3013 ],
#       [ 0.14373, -0.19203, -0.10162, -0.13971, -0.16537,  0.01365,
#        -0.50414, -0.7724 , -0.20645, -0.00596,  0.29257],
#       [ 0.22403,  0.13813,  0.00046, -0.83125, -0.2238 ,  0.22027,
#        -0.20529, -1.0097 , -0.40032, -0.03899,  0.66639],
#       [-0.18016, -0.05126, -0.00005, -0.05421, -0.70858,  0.0162 ,
#        -0.50756,  0.55363,  0.10635,  0.22438,  0.15757],
#       [-0.11159, -0.03721, -0.04259,  0.04104,  0.26134, -0.42266,
#        -0.18536, -0.43231, -0.26461, -0.06104,  0.1647 ],
#       [-0.12669, -0.18576, -0.12222,  0.3809 ,  0.4003 , -0.16078,
#        -1.2124 ,  1.3897 , -0.09635,  0.19189, -0.37922],
#       [-0.07126,  0.0006 ,  0.08035, -0.4548 , -0.50349,  0.16899,
#        -0.56222, -4.3508 , -0.2074 , -0.22341,  0.44315],
#       [-0.3742 ,  0.27843,  0.24887, -0.16829,  0.08399,  0.03369,
#        -0.23242, -0.39513, -0.3841 , -0.03876,  0.31454],
#       [-0.04225, -0.01311,  0.02398, -0.11784, -0.32893,  0.02075,
#         0.05477, -2.0963 ,  0.02382, -0.19213,  0.11124],
#       [-0.03754, -0.03333, -0.04991, -0.09042, -0.10211,  0.03229,
#        -0.18179, -0.30301, -0.00767,  0.01436, -0.05577]])
#
#nu,L = SSR(np.array([0.     , 0.     , 0.     , 0.98443, 1.88204, 1.4871 , 0.02138,
#        0.0237 , 2.77094, 0.     , 2.26514]),np.array([0.     , 0.     , 0.     , 0.1667 , 0.88707, 0.98981, 0.36639,
#        0.02345, 0.80461, 0.     , 0.     ]),mu,M)
#get = get_separatrix_point(np.array([0,1]), np.array([1,0]), nu, L, num_points=101)