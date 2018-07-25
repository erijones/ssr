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
from scipy.integrate import ode
from itertools import permutations

import warnings
warnings.filterwarnings("ignore")


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

def get_point_on_line(xa, xb, p):
    """ Return a point along the line that connects xa and xb, parameterized by
    p, where 0 <= p <= 1. Note p=0 returns xa, while p=1 returns xb. """
    return (1-p)*xa + p*xb

def goes_to_xa(xa, xb, val):
    """ This function checks to see if the point val converges to xa """
    if np.linalg.norm(val - xa) < .001:
        return True
    else:
        return False

def goes_to_xb(xa, xb, val):
    """ This function checks to see if the point val converges to xb """
    if np.linalg.norm(val - xb) < .001:
        return True
    else:
        return False

def get_steady_state(point, mu, M):
    """ This function simulates the gLV equations with parameters mu and M and
    initial condition point until the system reaches a steady state. Initially,
    simulations go until time=1000, but if the system doesn't converge in this
    time additional time is added to the simulation"""
    verbose = False
    t = np.linspace(0, 100000, 100001)
    sol = odeint(integrand, point, t, args=(mu, M))
    while np.linalg.norm(sol[-1] - sol[-100]) > 1e-8:
        error = np.linalg.norm(sol[-1] - sol[-2])
        if verbose:
            print(t[-1], error)
        t = np.linspace(t[-1], t[-1] + 100000, 100001)
        sol = odeint(integrand, sol[-1], t, args=(mu, M), Dfun=jacobian)

    if False:
        t_max = t[-1]
        print('  integrated until t={}'.format(t_max))

    final_sol = sol[-1]
    return final_sol

def get_separatrix_point(xa, xb, mu, M, num_points=101):
    """ This function find the separatrix between the fixed points xa and xb.
    If their basins of attraction agree, i.e. separatrix_xa and separatrix_xb
    are very close, it returns a tuple of their average. If the basins of
    attraction do not agree, i.e. separatrix_xa and separatrix_xb are
    different, it returns a tuple (separatrix_xa, separatrix_xb) to
    differentiate the basins of attraction. """

    ps = np.linspace(0, 1, num_points)
    points = np.array([get_point_on_line(xa, xb, p) for p in ps])
    final_vals = np.array([get_steady_state(point, mu, M) for point in points])

    went_to_xa = [goes_to_xa(xa, xb, val) for val in final_vals]
    went_to_xb = [goes_to_xb(xa, xb, val) for val in final_vals]
#     went_to_neither values are True if the corresponding point p went to neither xa nor xb
    went_to_neither = [0 for i in range(num_points)]
    for i in range(num_points):
        if (not went_to_xa[i]) and (not went_to_xb[i]):
            break
            went_to_neither[i] = True
        else:
            went_to_neither[i] = False

    for p, went_to_xa in zip(ps, went_to_xa):
        if not went_to_xa:
            separatrix_xa = p
            break

    for p, went_to_xb in zip(ps[::-1], went_to_xb[::-1]):
        if not went_to_xb:
            separatrix_xb = p
            break

    verbose = True
    if abs(separatrix_xa - separatrix_xb) <= 2/(num_points - 1):
        separatrix = ((separatrix_xa ) + (separatrix_xb)) / 2.0
        if verbose:
            print('separatrix between xa and xb occurs at p={:.5}'.format(separatrix))
        return separatrix, separatrix
    else:
        if sum(went_to_neither) > 0:
            neither_index = went_to_neither.index(True)
            neither_val = get_steady_state(points[neither_index], mu, M)
            print('    coexistent steady state occurs at {}'.format(neither_val))
        if verbose:
            print('basin of attraction for xa ends at p={:.5}'.format(separatrix_xa))
            print('basin of attraction for xb ends at p={:.5}'.format(separatrix_xb))
        return separatrix_xa, separatrix_xb


def SSR(xa,xb,mu,M):
    """This function performs a steady state reduction by taking in the relevant parameters, then performing the relevant operations,
     and finally returning the steady state reduced forms of the parameters "nu" and "L" """

    nu = np.array([np.dot(xa, mu),
                  np.dot(xb, mu)])

    L = np.array([[np.dot(xa.T,np.dot(M,xa)), np.dot(xa.T,np.dot(M,xb))],
                 [np.dot(xb.T,np.dot(M,xa)), np.dot(xb.T,np.dot(M,xb))]])
    return nu,L

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

def bisection(xa,xb,eps, mu, M):
    """Identify the separatrix (as a proportion p between xa and xb) for the
    system parameterized by mu and M, to eps precision, via the bisection
    method"""
    p1 = 0
    p2 = 1
    while np.linalg.norm(p2 - p1) > eps:
        po = (p2 + p1)/2.0
        point = get_point_on_line(xa, xb, po)
        val = get_steady_state(point,mu,M)
        if goes_to_xa(xa,xb,val) and not goes_to_xb(xa,xb,val):
            p1 = po
        elif not goes_to_xa(xa,xb,val) and goes_to_xb(xa,xb,val):
            p2 = po
        else:
            po = get_separatrix_point(xa,xb,mu,M, 101)
            break
            
    return po

def project_to_2D(traj, ssa, ssb):
    """Projects a high-dimensional trajectory traj into a 2D system, and
    returns a 2-dimensional trajectory"""
    new_traj = []
    for elem in traj:
        uu = np.dot(ssa, ssa); vv = np.dot(ssb, ssb)
        xu = np.dot(elem, ssa); xv = np.dot(elem, ssb)
        uv = np.dot(ssa, ssb)
        new_traj.append([(xu*vv - xv*uv)/(uu*vv - uv**2),
                         (uu*xv - xu*uv)/(uu*vv - uv**2)])
    new_traj = np.array(new_traj)
    return new_traj

def inflate_to_ND(traj, ssa, ssb):
    """Takes a 2D trajectory and "inflates" it to a high-dimensional
    trajectory that lies on a 2D plane (spanned by 0, ssa, and ssb) embedded in
    high-dimensional space."""
    new_traj = []
    for elem in traj:
        new_traj.append(elem[0]*ssa + elem[1]*ssb)
    new_traj = np.array(new_traj)
    return new_traj

def get_relative_deviation(xa,xb,p):  
    """ This function generates a trajectory in 11D with an initial condition.  
    The 11D trajectory is projected onto a 2D plane and subsequently inflated back to 11D. 
    Subsequently the trajectory length is found by summing  infinitesimally small
    arc-lengths of the 11D inflation. The function returns the relative deviation 
    by finding the difference between the original 11D trajectory and inflated 11D 
    trajectory and dividing by the length of the 11D inflated trajectory """
    ic = get_point_on_line(xa, xb, p)
    t = np.linspace(0, 25, 26)
    traj_ND = odeint(integrand, ic, t, args=(mu, M))
    traj_2D = project_to_2D(traj_ND, xa, xb)
    traj_on_plane = inflate_to_ND(traj_2D, xa, xb)
    dxds = [np.linalg.norm(traj_ND[i+1] - traj_ND[i]) for i in range(len(traj_ND)-1)]
    traj_length = sum(dxds)
    deviation_from_plane = np.linalg.norm(traj_ND - traj_on_plane)
    relative_deviation = deviation_from_plane/traj_length
    return relative_deviation



## MAIN FUNCTION

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
                print('hello')
#                print('{} is stable'.format(fp, is_stable))
            else:
                print('hello')
#                print('{} is unstable in {} direction'.format(fp, is_stable))
#            print()
        num_stable_fps += 1
        fp_list.append(fp)
fp_list = np.array(fp_list)
#print('there were {} stable fps out of {} total positive cases'.format(num_stable_fps, len(fps)))

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

xa = fp_list[0]; xb = fp_list[1]
sep_xa, sep_xb = get_separatrix_point(xa, xb,mu,M, 101)
call = SSR(xa,xb,mu,M)

#This returns Stein's steady states
stein_steady_states = get_stein_steady_states(stein_values, steady_state_2_list)

#returns all iterations of the possible combinations of Stein's Steady States
if False:
    combos = list(itertools.combinations(range(5), 2))
    for i,j in combos:
            ssa = stein_steady_states[i]
            ssb = stein_steady_states[j]
            temp_separatrix_11D = get_separatrix_point(ssa, ssb,mu,M, num_points=101)
            nu,L = SSR(ssa,ssb,mu,M)
            temp_separatrix_2D = get_separatrix_point(np.array([1,0]), np.array([0,1]), nu, L, num_points=101)
            print(' for the 11-D case the separatrix of ss{} and ss{} occurs at {}'.format(i, j, temp_separatrix_11D))
            print(' for the 2-D case the separatrix of ss{} and ss{} occurs at {}'.format(i, j, temp_separatrix_2D))
            bisected_separatrix_11D = bisection(ssa, ssb, .0001, mu, M)
            print(' The bisection method for the 11-D case yields the separatrix of ss{} and ss{} occurs at {}'.format(i, j, bisected_separatrix_11D))
  

     

#If this block is designated as True then all of the pairs of Stein's steady states are found. Using the bisection method the separatrices are found
# if the steady states that correspond to the sepatratrices have meaningful trajectories then the are subsequently passed as arguments in the function
# get_relative_deviation.
if True:
    combos = list(itertools.combinations(range(5), 2))
    for i,j in combos:
            ssa = stein_steady_states[i]
            ssb = stein_steady_states[j]
            pstar = bisection(ssa,ssb,.0001,mu,M)
            if isinstance(pstar, float):
                print(pstar)
                p1 = 1.1 * pstar
                p2 = 0.9 * pstar
                relative_dev =  get_relative_deviation(ssa,ssb,p1)
                relative_dev =  get_relative_deviation(ssa,ssb,p2)
                print('The relative devaition is {} for stein_state{} and stein_state{}'.format(relative_dev,i,j))
            else:
               print(pstar)
        
## for some random initial condition
#pstar = bisection(xa,xb,.0001,mu,M)
#p = pstar * 1.1
#example_ic = get_point_on_line(xa, xb, p)
## run a simulation to obtain high-dimensional (in this case 11D) trajectory
#t = np.linspace(0, 25, 26)
#traj_ND = odeint(integrand, example_ic, t, args=(mu, M))
## use 'project_to_2D' to turn it into 2D trajectory
#traj_2D = project_to_2D(traj_ND, xa, xb)
## use 'inflate_to_ND' to turn it into 11D trajectory that is on the embedded 2D
## plane
#traj_on_plane = inflate_to_ND(traj_2D, xa, xb)
## calculate the norm of the derivative of the trajectory at each time--- this
## is effectively computing ds/dt, which I will then integrate (in this case
## sum) to find s, the total arclength (called 'traj_length' here)
#dxds = [np.linalg.norm(traj_ND[i+1] - traj_ND[i]) for i in range(len(traj_ND)-1)]
#traj_length = sum(dxds)
## compute the difference from original 11D trajectory and the trajectory
## confined to the plane
#deviation_from_plane = np.linalg.norm(traj_ND - traj_on_plane)
## return deviation as a proportion of total trajectory length
## if the relative deviation is small, then SSR is working well
#relative_deviation = deviation_from_plane/traj_length
#print('relative deviation is {}'.format(relative_deviation))
#relative_dev =  get_relative_deviation(xa,xb,p)
#






