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

def get_point_on_line(xa, xb, p):
    """ Return a point along the line that connects xa and xb, parameterized by
    p, where 0 <= p <= 1. Note p=0 returns xa, while p=1 returns xb. """
    return (1-p)*xa + p*xb


t = np.linspace(0,100)
p = .3



def goes_to_xa(xa,xb,t, p) :
    ''' This function checks to see if a given point convereges to xa  '''
    point = get_point_on_line(xa, xb, p)
    t = np.linspace(0,100)
    sol = odeint(integrand, point, t, args=(mu, M))
    final_sol =  sol[-1]
    for i in range(len(xa)) :
        if all(np.subtract(xa,final_sol)) > .001:
            return True
        else :
            return False

def goes_to_xb(xa,xb,t, p) :
     ''' This function checks to see if a given point convereges to xb  '''
    point = get_point_on_line(xa, xb, p)
    t = np.linspace(0,100)
    sol = odeint(integrand, point, t, args=(mu, M))
    final_sol =  sol[-1]
    for i in range(len(xa)) :
        if all(np.subtract(xb,final_sol)) > .001:
            return True
        else :
            return False

### MAIN FUNCTION

param_list, ics = get_stein_parameters()
labels, mu, M, eps = param_list
fps = get_all_steady_states(mu, M)
fps = get_nonnegative_fixedpoints(fps)
fp_list = []
num_stable_fps = 0
for fp in fps:
    # make sure all fixed points are actually fixed points
    output = integrand(fp, 0, mu, M)
    assert(all(abs(output) < 1e-6))

    is_stable = get_stability(fp, mu, M, almost_stable=0, substability=False)
    if is_stable:
        verbose = False
        if verbose:
            if is_stable is True:
                print('{} is stable'.format(fp, is_stable))
            else:
                print('{} is unstable in {} direction'.format(fp, is_stable))
            print()
        num_stable_fps += 1
        fp_list.append(fp)

print('there were {} stable fps out of {} total positive cases'.format(num_stable_fps, len(fps)))


xa = fp_list[0]; xb = fp_list[1]
p=.5

result_xa =  goes_to_xa(xa,xb,t,p)
result_xb = goes_to_xa(xa,xb,t,p)


#get_param_line_equation(xa, xb)
intermediate_point = get_point_on_line(xa, xb, p)