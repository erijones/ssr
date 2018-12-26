#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 08:42:06 2018
@author: parkershankin-clarke
"""

#import time

#main()

#import packages 
import numpy as np
import barebones_CDI as bb
from itertools import *
import itertools
import pickle
import time
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt
import pickle
import math
import warnings
warnings.filterwarnings("ignore")
import operator
import time
from itertools import islice
import this
#import cProfile
import cProfile, pstats, io
#@profile
#from profilestats import profile


'''
@profile
def my_function(args, etc):
    pass
'''
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
#    print("--- %s seconds for get stein parameters ---" % (time.time() - start_time))
    return param_list, ics

def get_all_steady_states(mu, M):
    """ Given growth rates mu and interaction values M, return all steady
    states of the gLV equations dx_i/dt = x_i(mu_i + \sum_{j=1}^N M_ij x_j)"""
    
    N = len(mu)
    lst = list(range(N))
    combs = []
    # Credit: #https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
    # generate all possible combations of a list of length of M and mu
    for i in range(N+1):
        els = [list(x) for x in itertools.combinations(lst,i)]
        combs.extend(els)
    fixedpointslist = []
    fp2 = []
    for comb in combs:
        # generate all possible combinations of matricies(M)/vectors(mu) 
        temp_M = M[comb, :][:, comb]
        temp_mu = mu[comb]
        # solve the the fixed points where some populations are 0
        temp_fp = np.linalg.solve(temp_M, -temp_mu)
        full_fp = np.zeros(N)
        # ensures when a fixedpoint is not present a zero is assigned in it's place 
        # because an absent fixed point corresponds to a microbial population of zero
        for i,elem in enumerate(comb):
            full_fp[elem] = temp_fp[i]
        fixedpointslist.append(full_fp)
        fp2.append(temp_fp)
    fixedpointslist = np.array(fixedpointslist)
    
    return fixedpointslist

def get_nonnegative_fixedpoints(fps):
    """ Returns fixed points that are nonnegative """
    fps_positive_list = []
    for fp in fps:
        if all(fp >= -1e-8):
            fps_positive_list.append(fp)
    return np.array(fps_positive_list)

def integrand(x, t, mu, M):
    """ Return N-dimensional gLV equations """
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(M, x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
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

def get_stein_steady_states(fps):
    """This function imports  values from the dictionary that contains "stein's steady states" 
    and  also imports steady states with up to 2 unstable directions (almost_stable=2). This 
    code outputs matching steady states between stein's set and the calculated set"""
    # imports stein steady states A through E
    stein_dict = bb.get_all_ss()
    # takes each value from stein_dict (keys : A->E) and creates a list of lists which correspond to each value
    stein_ss = np.array([stein_dict[val] for val in stein_dict])
    # creates a matrix of size stein_ss filled with zeros
    final_list = np.array([np.zeros(len(stein_ss[0])) for i in range(len(stein_ss))])
    # generates permuations of all elements of fps and stein_ss
    iterations_list =  list(itertools.product(stein_ss, fps))
    # take each permutation
    for i in range(len(iterations_list)):
        compare_lists = iterations_list[i]
        # check to see whether the lists that are being combined are equal 
        if np.linalg.norm(compare_lists[0] - compare_lists[1]) < .001:
            for j in range(len(stein_ss)):
                # verify that compare_lists[1] and [2] equal stein_ss
                if np.linalg.norm(compare_lists[1] - stein_ss[j]) < .001:
                    #if they are equal, then add the element compare_lists[1] to final_list
                    final_list[j] = compare_lists[1]
    return final_list

def get_nonnegative_fps(mu, M, almost_stable,substability):
    """This fuction returns a list of nonnegative fixed points with stabilities chosen by the user"""
    # grab all fixed points
    all_fps = get_all_steady_states(mu, M)
    # filter out negative fixed points
    fps = get_nonnegative_fixedpoints(all_fps)
    fp_list = []
    for fp in fps:
        # make sure all fixed points are actually fixed points
        output = integrand(fp, 0, mu, M)
        assert(all(abs(output) < 1e-6))
        # check for stability
        is_stable = get_stability(fp, mu, M, almost_stable = almost_stable, substability=False)
        if is_stable:
            fp_list.append(fp)
    fp_list = np.array(fp_list)
    return fp_list

def SSR(xa,xb,mu,M):
    """This function performs a steady state reduction by taking in the
    relevant parameters, then performing the relevant operations, and finally
    returning the steady state reduced forms of the parameters "nu" and "L" """
    
    
    if True:
        new_mu_a = ((np.dot(xa, mu))/sum(xa))
        new_mu_b = ((np.dot(xb, mu))/sum(xb))
        new_M_aa = np.dot(xa.T,np.dot(M,xa)) / sum(xa)
        new_M_ab = np.dot(xa.T,np.dot(M,xb)) / sum(xa)
        new_M_ba = np.dot(xb.T,np.dot(M,xa)) / sum(xb)
        new_M_bb = np.dot(xb.T,np.dot(M,xb)) / sum(xb)
            
        
        
        
        
        #nu is the steady-state reduced mu
        nu = np.array([new_mu_a,
                      new_mu_b])
        #L is the steady-state reduced M
        L = np.array([[new_M_aa, new_M_ab],
                     [new_M_ba, new_M_bb]])


#    
#    if False :
#        new_mu_a = np.dot(np.dot(np.diag(xa), xa), mu)/(np.linalg.norm(xa)**2)
#        new_mu_b = np.dot(np.dot(np.diag(xb), xb), mu)/(np.linalg.norm(xb)**2)
#    
#        new_M_aa = np.dot(np.dot(np.diag(xa), xa).T, np.dot(M, xa))/(np.linalg.norm(xa)**2)
#        new_M_bb = np.dot(np.dot(np.diag(xb), xb).T, np.dot(M, xb))/(np.linalg.norm(xb)**2)
#        ya = xa/np.linalg.norm(xa)
#        yb = xb/np.linalg.norm(xb)
#        numerator = sum([sum([ M[i][j]*(ya[i]*yb[j] + yb[i]*ya[j]) *sum([ya[i]*yb[k]**2 - yb[i]*ya[k]*yb[k] for k in range(len(xa))]) for j in range(len(xa))]) for i in range(len(xa))])
#        denom = (sum([ya[i]**2 for i in range(len(xa))])
#                * sum([yb[i]**2 for i in range(len(xa))])
#                - sum([ya[i]*yb[i] for i in range(len(xa))])**2)
#        new_M_ab = numerator*np.linalg.norm(xb)/denom
#    
#        ya = xa/np.linalg.norm(xa)
#        yb = xb/np.linalg.norm(xb)
#        numerator = sum([sum([
#                    M[i][j]*(ya[i]*yb[j] + yb[i]*ya[j])
#                    *sum([yb[i]*ya[k]**2 - ya[i]*ya[k]*yb[k] for k in
#                        range(len(xa))])
#            for j in range(len(xa))]) for i in range(len(xa))])
#        denom = (sum([ya[i]**2 for i in range(len(xa))])
#                * sum([yb[i]**2 for i in range(len(xa))])
#                - sum([ya[i]*yb[i] for i in range(len(xa))])**2)
#        new_M_ba = numerator*np.linalg.norm(xa)/denom
#        
#        
#        #nu is the steady-state reduced mu
#        nu = np.array([new_mu_a,
#                      new_mu_b])
#        #L is the steady-state reduced M
#        L = np.array([[new_M_aa, new_M_ab],
#                     [new_M_ba, new_M_bb]])
#        print('--------------------NEW-------------------------')
#        print(nu,L)
    #    import sys
    #    sys.exit()
    
    return nu,L


def get_analytic_separatrix(eps, mu, M):
    """Use the analytically determined separatrix from barebones_CDI.py to
    compute the separatrix point, rather than the computationally expensive
    bisection method"""
    # call function Params from barbones_CDI.py 
    p = bb.Params((M, None, mu))
    # return eigenvalues (where wither xa or xb = 0)
    xa_eigs = np.linalg.eig(p.get_jacobian(p.get_10_ss()))[0]
    xb_eigs = np.linalg.eig(p.get_jacobian(p.get_01_ss()))[0]
    # return eigenvalues  where both eigenvalues are nonzero
    mixed_ss_eigs = np.linalg.eig(p.get_jacobian(p.get_11_ss()))[0]
    zero = 1e-10
    if any(xa_eigs > zero) and all(xb_eigs <= zero):
        # xa is unstable => trajectory will go to xb
        sep_point = 0
    elif all(xa_eigs <= zero) and any(xb_eigs > zero):
        # xb is unstable
        sep_point = 1
    elif all(mixed_ss_eigs < zero):
        # mixed steady state is stable => not a bistable system
        sep_point = (0, 1)
    else:
        # standard bistable system; compute separatrix analytically
        coeffs = p.get_taylor_coeffs(7)
        u, v = p.get_11_ss()
        p1 = 0
        p2 = 1
        while abs(p2 - p1) > eps:
            po = (p1 + p2)/2.0
            sep_val = sum([(coeffs[i]/math.factorial(i))*(po - u)**i for i in range(len(coeffs))])
            val = 1 - po - sep_val
            if val > 0:
                p1 = po
            elif val < 0:
                p2 = po
        sep_point = 1 - po

    print('    2D analytic separatrix at p={}'.format(sep_point))
    return sep_point


def get_stability(x, mu, M, almost_stable=None, substability=False):
    """ Evaluate stability of steady state x. If stable, returns True; if
    unstable, returns False. If 'substability' is True, we only consider the
    stability of the fixed point in the directions where x is non-zero. If
    'almost_stable' is not None, check if there are almost_stable number of
    positive eigenvalues or fewer."""
    # get jacobian
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
    # returns number of stable directions if not fully stable
    if almost_stable:
        if num_unstable_dirs <= almost_stable:
            return num_unstable_dirs
    #if not stable False is returned    
    return False

def get_point_on_line(xa, xb, p):
    """ Return a point along the line that connects xa and xb, parameterized by
    p, where 0 <= p <= 1. Note p=0 returns xa, while p=1 returns xb. """
    return (1-p)*xa + p*xb

def get_steady_state(point, mu, M):
    """ This function simulates the gLV equations with parameters mu and M and
    initial condition point until the system reaches a steady state. Initially,
    simulations go until time=1000, but if the system doesn't converge in this
    time additional time is added to the simulation"""
    verbose = False
    t = np.linspace(0, 10000, 10001)
    sol = odeint(integrand, point, t, args=(mu, M))
    # check to see whether the solution converged to a steady state
    while np.linalg.norm(sol[-1] - sol[-100]) > 1e-8:
        # if the solution does not converge calculate the error
        error = np.linalg.norm(sol[-1] - sol[-2])
        # print error 
        if verbose:
            print(t[-1], error)
        # add more time in order to probe further for convergence     
        t = np.linspace(t[-1], t[-1] + 10000, 10001)
        # re-evaluate in with addition time
        sol = odeint(integrand, sol[-1], t, args=(mu, M))

    if False:
        # print the maximum time that it took to converge 
        t_max = t[-1]
        print('  integrated until t={}'.format(t_max))
    # grab and return final solution
    final_sol = sol[-1]
    return final_sol

    
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
    # compute where points between two steady states go
    went_to_xa = [goes_to_xa(xa, xb, val) for val in final_vals]
    went_to_xb = [goes_to_xb(xa, xb, val) for val in final_vals]
    # went_to_neither values are True if the corresponding point p went to neither xa nor xb
    went_to_neither = [0 for i in range(num_points)]
    # iterate through the locations of where the points ended
    for i in range(num_points):
        if (not went_to_xa[i]) and (not went_to_xb[i]):
            went_to_neither[i] = True
        else:
            went_to_neither[i] = False
    # if points went to xa find the place where it switched and the points started converging to xb. Call that point the separtrix
    for p, went_to_xa in zip(ps, went_to_xa):
        if not went_to_xa:
            separatrix_xa = p
            print(separatrix_xa)
            break

    for p, went_to_xb in zip(ps[::-1], went_to_xb[::-1]):
        if not went_to_xb:
            separatrix_xb = p
            break

    verbose = False
    # check to see whether separatrix for xa and xb are in a resonable range 
    if abs(separatrix_xa - separatrix_xb) <= 2/(num_points - 1):
        # if they are in a reasonable range then average the two values
        separatrix = ((separatrix_xa ) + (separatrix_xb)) / 2.0
        if verbose:
            print('separatrix between xa and xb occurs at p={:.5}'.format(separatrix))
        return separatrix, separatrix
    else:
        # if the seperatrix points are not in a reasonable range display the basin of attraction that the two points encompass. 
        if verbose:
            print('basin of attraction for xa ends at p={:.5}'.format(separatrix_xa))
            print('basin of attraction for xb ends at p={:.5}'.format(separatrix_xb))
            # if point did not converege to xa or xb re-calculate what staedy state it did converege to.
        if sum(went_to_neither) > 0:
            neither_index = went_to_neither.index(True)
            neither_val = get_steady_state(points[neither_index], mu, M)
            if len(mu) == 2:
                print('    instead {}D traj goes to steady state at {}'.format(len(mu), neither_val))

            if len(mu) > 2:
                unstable_2_fps =  get_nonnegative_fps(mu,M,2,False)
                stein_fps = get_stein_steady_states(unstable_2_fps)
                coexist_index = False
                labels = ['A', 'B', 'C', 'D', 'E']
                for i,stein_fp in enumerate(stein_fps):
                    if np.linalg.norm(neither_val - stein_fp) < .0001:
                        coexist_index = labels[i]
                print('    instead {}D traj goes to steady state {}'.format(len(mu), coexist_index))
            return separatrix_xa, separatrix_xb
        else:
            return separatrix_xa, separatrix_xb
    
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
            po = get_separatrix_point(xa, xb, mu, M, 201)
            break
    verbose = True
    if verbose:
        N = len(mu)
        if isinstance(po, float):
            print('    {}D separatrix at p={}'.format(N, po))

    return po


def TimeAndState(labels,stein_steady_states,mu,M,filename):
    sep_list_2D = {}
    sep_list_11D = {}
    a = (0,2)
#     intialize the variables used to calculate time for generating sepatricies through bisection and analyticaly 
    bisect_time_2D = 0
    analytic_time_2D = 0
    bisect_time_11D = 0
    # generate all possible paths for states A-->E
    combos = list(itertools.combinations(range(len(labels)), 2))
    # iterate through possible paths
    for i,j in combos:
        if True:
#        if i == a[0] and j == a[1]: 
            print(i,j)
            # choose a start and end point (11D)
            ssa_11 = stein_steady_states[i]
            ssb_11 = stein_steady_states[j]
            ## calculate 11D separatrices using the bisection method
            # start the clock
            t0 = time.time()
            # calculate the separatrices using bisection
            temp_separatrix_11D = bisection(ssa_11, ssb_11, .0001, mu, M)
            # record time
            bisect_time_11D += time.time() - t0
            ## calculate 11D separatrices analytically
            # reduce 11D vectors to 2D
            nu, L = SSR(ssa_11, ssb_11, mu, M)
            # start the clock
            t0 = time.time()
            # calculate the 2D sepatricries analytically
            temp_separatrix_2D = get_analytic_separatrix(.0001, nu, L)
            # record time
            analytic_time_2D += time.time() - t0
            ## calculate 2D sepatrices using the bisection method
            #steady states in 2D
            #ssa_2 = np.array([1, 0])
            #ssb_2 = np.array([0, 1])
            ##start clock
            #t0 = time.time()
            ## get 2D sepatrices using the bisection method
            #bisection(ssa_2, ssb_2, .0001, nu, L)
            ##record time
            #bisect_time_2D += time.time() - t0
            # record  2D sepatricies using into the array sep_list_2D (via analytic method )
            # record  11D sepatricies using into the array sep_list_11D (via bisection method )
            sep_list_2D[(i, j)] = temp_separatrix_2D
            sep_list_11D[(i, j)] = temp_separatrix_11D
            # checks the datatypes of the sepatricies
            try:
                sep_list_2D[(j, i)] = 1 - temp_separatrix_2D
            except TypeError:
                sep_list_2D[(j, i)] = tuple(1 - val for val in temp_separatrix_2D)
            try:
                sep_list_11D[(j, i)] = 1 - temp_separatrix_11D
            except TypeError:
                sep_list_11D[(j, i)] = tuple(1 - val for val in temp_separatrix_11D)
            #open file for writing in binary mode        
            with open('data/{}'.format(filename), 'wb') as f:
            # write a pickled representation of sep_list_2D and sep_list_11D to the open file sep_lists_analytic.
                pickle.dump((sep_list_2D, sep_list_11D), f)
                
            print('bisection 11D time: {} s'.format(bisect_time_11D))
            print('bisection 2D time: {} s'.format(bisect_time_2D))
            print('analytic 2D time: {} s'.format(analytic_time_2D))


    return sep_list_2D,sep_list_11D
            
    
def FastTimeAndState(labels,stein_steady_states,mu,M,filename):
    #
    sep_list_2D = {}
    sep_list_11D = {}
   #open file for reading in binary mode
    with open('data/{}'.format(filename), 'rb') as f:
        # read a pickled separtrices from the open file sep_lists_analytic return the reconstituted object hierarchy specified therein.
        sep_list_2D, sep_list_11D = pickle.load(f)
        # if verbose equals True the 11D separtrix (analytic) and the 2D separtricies (analytic, bisection) will be printed
#        verbose = True
#        if verbose:
#            # iterates through the combintations of the labels and display each path (i.e. two labels)  
#            for i in range(len(labels)):
#                for j in range(len(labels)):
#                    if i < j:
#                        print(labels[i], labels[j])
#                        for N, sep_list in zip([2, 11], [sep_list_2D, sep_list_11D]):
#                            p = sep_list[(i, j)]
#                            if isinstance(p, float) or isinstance(p, int):
#                                print('    {}D separatrix at p={}'.format(N, p))
#                            else:
#                                print('    {}D separatrices at p={} and p={}'.format(N, p[0], p[1]))
    return sep_list_2D,sep_list_11D

def NormAndSep(sep_list_11D,sep_list_2D,labels,stein_steady_states):
#    a = (0,2)
    a = range(len(labels))
    len(labels)
    #arrays intialized for part1.) (look above for more information)
    sep_matrix_11D = np.array([])
    sep_matrix_2D = np.array([])
   

        
    #arrays intialized for part2.) (look above for more information)
    norm_matrix_2D = np.array([])
    norm_matrix_11D = np.array([])
    
    index_matrix= np.array([])
    
    #11-D 
    if True:
        for i in a:#a:
            for j in a: #a:
                if i != j:
                    for N, sep_list in zip([2, 11], [sep_list_11D]):
#                        print(sep_list)
                        p = sep_list[(i, j)]
                        ssa = stein_steady_states[i]
                        ssb = stein_steady_states[j]
                        if isinstance(p, float):
                            index_matrix = np.append(index_matrix,(i,j))
                            sep_matrix_11D = np.append(sep_matrix_11D, p)
                            norm = p * np.linalg.norm(ssb - ssa)
                            norm_matrix_11D = np.append(norm_matrix_11D,norm)                          
                        else:
                            index_matrix = np.append(index_matrix,(i,j))
                            sep_matrix_11D = np.append(sep_matrix_11D, None)
                            norm_matrix_11D = np.append(norm_matrix_11D,None)
                elif  i == j : 
                     p=0
                     index_matrix = np.append(index_matrix,(i,j))
                     sep_matrix_11D = np.append(sep_matrix_11D,p)
                     norm_matrix_11D = np.append(norm_matrix_11D,p)
       
    #2-D        
    if True:
        for i in a: #a:
            for j in a: #a:
                if i != j : 
                    
                    for N, sep_list in zip([2, 11], [sep_list_2D]):
#                        print('sep list 2d')
#                        print(sep_list)
                        p = sep_list[(i, j)]
                        if isinstance(p, float):
                            sep_matrix_2D = np.append(sep_matrix_2D, p)
                            norm = p * np.linalg.norm(np.array([0,1]) - np.array([1,0]))
                            norm_matrix_2D = np.append(norm_matrix_2D,norm) 
                        elif isinstance(p,int):
#                            print('int p is')
#                            print(p)
                            sep_matrix_2D = np.append(sep_matrix_2D, .0000000001)
                            norm = p * np.linalg.norm(ssb - ssa)
                            norm_matrix_2D = np.append(sep_matrix_2D,norm) 
                        else:
                            sep_matrix_2D = np.append(sep_matrix_2D, None)
                            norm_matrix_2D = np.append(norm_matrix_2D,None)
                elif  i == j : 
                    p=0
                    sep_matrix_2D = np.append(sep_matrix_2D,p)
                    norm_matrix_2D = np.append(norm_matrix_2D,p)
    
    
    ##reformat both arrays into  numss x numss matricies:

    #arrays for part1.) (look above for more information)                  
    sep_matrix_2D = np.resize(sep_matrix_2D,(len(a),len(a)))
    sep_matrix_11D = np.resize(sep_matrix_11D,(len(a),len(a)))
#    index_matrix = np.resize( index_matrix(5,5))


    #arrays for part2.) (look above for more information)  
    norm_matrix_2D = np.resize(norm_matrix_2D,(len(labels),len(labels)))
    norm_matrix_11D = np.resize(norm_matrix_11D,(len(labels),len(labels)))
    return sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D

def make_food_web(sep_list_2D, sep_list_11D,labels):
    """This function simulates a network of steady-states solutions and their sepatricies. """
    fig, ax = plt.subplots(figsize=(6,6))
    wheels = []
#    labels = list(range(0,len(sep_list_2D)))
    for i in range(len(labels)):
        # save the order of colors when plotting
        wheels.append(ax.plot([0], [0], linewidth=0.0))

    import matplotlib.patches as mpatches
    circ_size = 0.15
    # circs are a matplotlib object
    # xx and yy are locations of these circles (to be plotted)
    circs = []; xx = []; yy = [];
    circ_xx = []; circ_yy = []
    #print('colorwheel is', wheels[1][0].get_color())
#    labels = list(range(0,len(sep_list_2D)))
    for i in range(len(labels)):
        xx.append(np.sin(2*np.pi*i/len(labels)))
        yy.append(np.cos(2*np.pi*i/len(labels)))
        circ_xx.append(1.13*np.sin(2*np.pi*i/len(labels)))
        circ_yy.append(1.13*np.cos(2*np.pi*i/len(labels)))
        # here I append a "Circle" object, and say that its color is what I
        # saved earlier (in wheels)
        circs.append(plt.Circle((circ_xx[-1], circ_yy[-1]), circ_size, lw=0,
            color=wheels[i][0].get_color()))
        plt.text(circ_xx[-1], circ_yy[-1], labels[i], weight='bold', ha='center',
                va='center', fontsize=28, color='black')

        # plot all of the circles in ax
    for i in range(len(labels)):
        ax.add_artist(circs[i])
        True

    for dim,sep_list in zip(['2D', '11D'], [sep_list_2D, sep_list_11D]):
        # add lines connecting circles
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    continue
                arrow_color = wheels[i][0].get_color()
                p = sep_list[(i,j)]
                try:
                    point = np.array([xx[i]*(1-p) + xx[j]*p, yy[i]*(1-p) + yy[j]*p])
                except TypeError:
                    #print(i, j, p)
                    if i < j:
                        point = np.array([xx[i]*(1-p[0]) + xx[j]*p[0], yy[i]*(1-p[0]) + yy[j]*p[0]])
                    else:
                        point = np.array([xx[i]*(1-p[1]) + xx[j]*p[1], yy[i]*(1-p[1]) + yy[j]*p[1]])
                #print(point)
                if dim == '11D':
                    plt.plot(point[0], point[1], marker='.', color='black',
                            markersize=20, zorder=5,
                            markerfacecolor='none')
                if dim == '2D':
                    plt.plot(point[0], point[1], marker='s', color='black',
                            markersize=9, zorder=5, markerfacecolor='none')
                curve_type = 'arc3,rad=0'
                if dim == '11D':
                    thickness = 6
                    if i == 0 or i == 2:
                        zorder = 2
                    else:
                        zorder = 1
                    outer_color = arrow_color
                if dim == '2D':
                    thickness = 2
                    if i == 0 or i == 2:
                        zorder = 4
                    else:
                        zorder = 3
                    outer_color = lighten_color(arrow_color, 1.4)
                # make arrows pointing from one circle to another
                ax.plot([xx[i],point[0]], [yy[i], point[1]],
                        color=outer_color, linewidth=thickness, zorder=zorder,
                        solid_capstyle='butt')

    edge = (1+circ_size)*1.2
    ax.set_aspect('equal')
    plt.axis([-edge, edge, -edge, edge])
    plt.axis('off')
    plt.tight_layout()
    filename = 'figs/example_food_web_5.pdf'
    plt.savefig(filename)
    print('saved fig to {}'.format(filename))
    return                
                
# from Ian Hincks, https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

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
    t = np.linspace(0, 25, 151)
    traj_ND = odeint(integrand, ic, t, args=(mu, M))
    traj_2D = project_to_2D(traj_ND, xa, xb)
    traj_on_plane = inflate_to_ND(traj_2D, xa, xb)

    ds = np.array([np.linalg.norm(traj_ND[i+1] - traj_ND[i])
                   for i in range(len(traj_ND) - 1)])
    traj_diff = traj_ND - traj_on_plane
    integrate_diff = [np.linalg.norm(traj_diff[i])*ds[i] for i in range(len(traj_diff)-1)]
    dxds_total = [np.linalg.norm(traj_ND[i+1] - traj_ND[i]) for i in range(len(traj_ND)-1)]
    deviation_from_plane = sum(integrate_diff)
    traj_length = sum(dxds_total)
    relative_deviation = deviation_from_plane/traj_length
    return relative_deviation

def generate_paths(SE_point,steadystate):
    """This function generates all possible paths from a given start point to a given end point """
    selectpathlist = []
    #generate all possible permuations
    for i in range(2,5):
        p = list(itertools.permutations(steadystate, i))
        for elem in p:
            #take only the permuations that lead from the start point to the end point
            if elem[0] == SE_point[0] and elem[len(elem)-1] == SE_point[1]:
                selectpathlist = selectpathlist + [elem]
    return selectpathlist

def conv(allpaths):
    """ this function will take all paths from a given start point to a given endpoint and convert them from 
    letters to numbers"""
    allpathslist = []
    #this dictionary will be used to convert each path in allpaths from letters to their corresponding numbers
    #credit: https://stackoverflow.com/questions/17295776/how-to-replace-elements-in-a-list-using-dictionary-lookup
    values = ('A', 'B', 'C', 'D','E')
    keys = (0,1,2,3,4)
    convdict = dict(zip(keys,values))
    for path in allpaths:
        #replace an element in a list using a dictionary lookup
        rev_subs = { v:k for k,v in convdict.items()}
        p = [rev_subs.get(item,item) for item in path]
        allpathslist = allpathslist + [p]
    return allpathslist

def red_sum(allpaths,separtrix_matrix):
    """ This function reduces all 1 by n paths to 1 by 2 paths and sums their corresponding separatrix values 
    in order to find the total path length"""
    sep = separtrix_matrix
    sepa = 0
    sum_list = []
    for paths in allpaths:
        #This block of code would be utilized if there were less than or equal to two steady states
        if len(paths) == 1:
            print('path==1')
            for path in paths:
                # 1 by 2 paths
                if len(path) == 2:
                    row = path[0]
                    column = path[1]
                    separatrix = sep[row,column]
                    sum_list += [separatrix]
                #1 by n paths    
                if len(path) > 2:
                    path_list = []
                    for i in range(0,len(path)-1):           
                        path_list = [path[i],path[i+1]]  + path_list
                        chunksumm = [path_list[x:x+2] for x in range(0, len(path_list), 2)]
                    single_list = [] 
                    for chunk in chunksumm:
                        row = chunk[0]
                        column = chunk[1]
                        separatrix = sep[row,column]
                        single_list =  single_list + [separatrix]
                        if all(isinstance(n, float) for n in single_list):
                            #credit : https://stackoverflow.com/questions/8964191/test-type-of-elements-python-tuple-list
                            sepa = sum(single_list)
    
                        else:
                            sepa = None
    
              
                    sum_list += [[sepa]]

            
        #This block of code would be utilized if there were greater than or equal to two steady states
        else:

            for path in paths:
                if len(path) == 2:
                    row = path[0]
                    column = path[1]
                    separatrix = sep[row,column]
                    sum_list = sum_list + [separatrix]
                if len(path) > 2:
                    path_list = []
                    for i in range(0,len(path)-1):           
                        path_list = [path[i],path[i+1]]  + path_list
                        #credit: https://stackoverflow.com/questions/4501636/creating-sublists
                        chunksumm = [path_list[x:x+2] for x in range(0, len(path_list), 2)]
                    single_list = [] 
                    for chunk in chunksumm:
                        row = chunk[0]
                        column = chunk[1]
                        separatrix = sep[row,column]
                        single_list =  [separatrix]  + single_list 
                        if all(isinstance(n, float) for n in single_list):
                            sepsum = sum(single_list)
                        else:
                            sepsum = None
                    sum_list =  sum_list + [sepsum]
                    
        cchunksumm = [sum_list[x:x+len(paths)] for x in range(0, len(sum_list), len(paths))]
    return cchunksumm

def navigate_between_fps(sep_matrix, verbose, labels,hrz):
    """ Return all possible ways from one steady state to another. Returns a
    dictionary ordered_paths that takes a pair of steady states as a key (e.g.
    ordered_paths['AC']) and returns a list of the shortest to the longest
    path. Also returns a dictionary path_lengths that takes a path (e.g.
    path_lengths['ADCE']) and returns the length of that path """
    ss_names = labels
    path_lengths = {}
    ordered_paths = {}
    
    

    starts_ends = itertools.permutations(ss_names, 2)

    
    for start_end in starts_ends:
        if True : #start_end[0] == '3' :
           

            # start_end looks like ('A', 'B'); start_end_str looks like 'AB'
            start_end_str = ''.join(start_end)
            ordered_paths[start_end_str] = []

            # get all possible in-between paths that start and end with start_end 
            remainder = [item for item in ss_names if item not in start_end]
       

            
            within_paths = []
            for N in range(hrz):
                within_paths.extend(itertools.combinations(remainder, N))

            # find the distance of all possible paths that start and end with start_end
            for within_path in within_paths:
                within_path = list(within_path)
            
            
                full_path = [start_end[0]] + within_path + [start_end[-1]]
#                print(full_path)
                full_path_str = ''.join(full_path)
                full_path_indices = [ss_names.index(val) for val in full_path]
                
#                print(full_path_indices)
                full_path_distance = 0
                
                for i in range(len(full_path_indices)-1):
                    i1 = full_path_indices[i]
                    i2 = full_path_indices[i+1]
                
                    try:
                        full_path_distance += sep_matrix[i1, i2]
                    except TypeError:
                        full_path_distance += 1000
                if verbose:
                    print(full_path_str, full_path_distance)
                path_lengths[full_path_str] = full_path_distance
    
                # add the distance of possible paths from start to end in order
                # (this is effectively 'insertion sort')
                if len(ordered_paths[start_end_str]) == 0:
                    ordered_paths[start_end_str].append(full_path_str)
                else:
                    insert_flag = False
                    for i,path in enumerate(ordered_paths[start_end_str]):
                        if (full_path_distance < path_lengths[path]) and not insert_flag:
                            ordered_paths[start_end_str].insert(i, full_path_str)
                            insert_flag = True
                    if not insert_flag:
                        ordered_paths[start_end_str].append(full_path_str)
            if verbose:
                print(ordered_paths[start_end_str])
                

    return ordered_paths, path_lengths


def sortedpathdic(sep_matrix_11D,sep_matrix_2D):
    steady_states = 'ABCDE'
    pathlist = []
    convpaths = []
    d11= []
    d2 = []
    # produces a list of tuples each of which represent a start and end point for a path
    enteries = list(itertools.permutations(steady_states,2))
    # generate a all possible paths given a startpoint and endpoint
    for entry in enteries:
        path = generate_paths(entry,steady_states)
        pathlist =   pathlist + [path] 
    # feeds in all paths and collected the converted paths into a list
    for path in pathlist:
        convpath = conv(path)
        convpaths = [convpath] + convpaths
    #reverse convpaths in order to match the labels in pathlist
    revconpaths = list(reversed(convpaths))
    #reduce and sum in order to find path lengths in 11-D
    pathlengths11d = red_sum(revconpaths,sep_matrix_11D)
    #reduce and sum in order to find path lengths in 2-D
    pathlengths2d = red_sum(revconpaths,sep_matrix_2D)
    #put pathlengths in dictionary and sort them 11d
    for i in range(len(pathlist)):
       pathdict11d = dictionary(pathlengths11d[i],pathlist[i])
       sortedict11d = delandsort(dict(pathdict11d))
       d11 = [[(sortedict11d)]] + d11
    #put pathlengths in dictionary and sort them
    for i in range(len(pathlist)): 
       pathdict2d = dictionary(pathlengths2d[i],pathlist[i])
       sortedict2d = delandsort(dict(pathdict2d))
       d2 = [[(sortedict2d)]] + d2
       
       
    return d11,d2
#               

def dictionary(paths,labels):
    """This function returns a dictionary of paths and pathlength """
    # credit: ##https://stackoverflow.com/questions/209840/convert-two-lists-into-a-dictionary-in-python
    keys = labels
    values = paths
    dictionary  = dict(zip(keys,values))
    return dictionary

def delandsort(mydict):
    """This function returns a sorted dictionary with the None data type excluded from the dictionary """
    # credit : https://stackoverflow.com/questions/15158599/removing-entries-from-a-dictionary-based-on-values
    # credit: http://thomas-cokelaer.info/blog/2017/12/how-to-sort-a-dictionary-by-values-in-python/
    mydict = { k:v for k, v in mydict.items() if v }
    sorted_d = dict(sorted(mydict.items(), key=operator.itemgetter(1)))
    return sorted_d

def compare(array1,array2):
    "The purpose of this function is to compare 2-D and N-D optimal paths. It returns path trajectory, and the agreement between the two respective paths"
    counter = 0
    start = 0
    end = 0
    if abs(len(array1) - len(array2)) == 0:
        for j in range(len(array1)):
            elem1 = array1[j]
            elem2 = array2[j]
            start = elem1[0]
            end = elem1[len(elem1)-1]
            if elem1 == elem2 :
                counter = counter + 1
    return counter,start,end

def justkeys(sep11,sep2,norm11,norm2):
    indexs11d= []
    indexs2d= []
    indexn11d= []
    indexn2d= []
    key_listsep11d = []
    key_listsep2d = []  
    key_listnorm2d = []
    key_listnorm11d = []
    for dictt in sep2:
        key_listsep2d  = list(dictt[0].keys()) + key_listsep2d
        indexs2d =  [len(dictt[0].keys())] + indexs2d 
    for dictionary in sep11:
        key_listsep11d  = list(dictionary[0].keys()) + key_listsep11d
        indexs11d = [len(dictionary[0].keys())] + indexs11d 
    for dicttt in sep2:
        key_listnorm2d   = list(dictt[0].keys()) + key_listnorm2d 
        indexn2d =  [len(dictt[0].keys())] + indexn2d 
    for dictttt in sep11:
        key_listnorm11d  = list(dictionary[0].keys()) + key_listnorm11d
        indexn11d = [len(dictionary[0].keys())] + indexn11d 
    
    
    mylist = key_listsep11d
    seclist = indexs11d
    it = iter(mylist) 
    ntuit11sd =[list(islice(it, 0, i)) for i in seclist]

    mylist = key_listsep2d
    seclist = indexs2d
    it = iter(mylist) 
    ntuit2sd =[list(islice(it, 0, i)) for i in seclist]
    
    mylist = key_listnorm11d
    seclist = indexn11d
    it = iter(mylist) 
    ntuit11nd =[list(islice(it, 0, i)) for i in seclist]
    
    mylist = key_listnorm2d
    seclist = indexn2d
    it = iter(mylist) 
    ntuit2nd =[list(islice(it, 0, i)) for i in seclist]
    
    return ntuit11sd,ntuit2sd,ntuit11nd,ntuit2nd

def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def extra():
    """ Extra code from the main function. Won't run as is, unless necessary
    parameters (e.g. stein_steady_states) are generated elsewhere """
    # If this block is designated as True then all of the pairs of Stein's
    # steady states are found. Using the bisection method the separatrices are
    # found if the steady states that correspond to the sepatratrices have
    # meaningful trajectories then the are subsequently passed as arguments in
    # the function get_relative_deviation.
    if True:
        combos = list(itertools.combinations(range(5), 2))
        for i,j in combos:
            ssa = stein_steady_states[i]
            print(ssa)
            ssb = stein_steady_states[j]
            pstar = bisection(ssa,ssb,.0001,mu,M)
            if isinstance(pstar, float):
                print('the pstar is')
                print(pstar)
                p1 = 1.1 * pstar
                p2 = 0.9 * pstar
                relative_dev_1 =  get_relative_deviation(ssa,ssb,p1)
                relative_dev_2 =  get_relative_deviation(ssa,ssb,p2)
                print('The relative deviation is {} and {} for stein_state{} and stein_state{}'.format(relative_dev_1, relative_dev_2,i,j))
            else:
                print(pstar)
    
    if True:
        make_food_web(sep_list_2D, sep_list_11D)

def Generate_And_Save_FixedPoints(read_data,UD,stein,generateFigure) :
    global sep_list_2D
    global sep_list_11D
    global stein_steady_states
    global stable_fps
    # load stein parameters
    param_list, ics = get_stein_parameters()
    # assign individual parameters from param_list
    bacteriaLabels, mu, M, eps = param_list
    #stable nonnegative fixedpoint list
    stable_fps = get_nonnegative_fps(mu, M,UD,False)
    # unstable (2 directions) nonegative fixedpoint list
    unstable_2_fps = get_nonnegative_fps(mu,M,2,False)
    # get mactching steady states from a dictionary of stein's steady states
    stein_steady_states = get_stein_steady_states(unstable_2_fps)
    # track changes in figures
    if not stein:
        labelStates = {0:list(range(0,2)),1:list(range(0,8)),2:list(range(0,30))}
        labels = labelStates.get(UD)
        #file that contains precalculated separatricies(speeds up program time)
        filename = 'sep_lists_analytic{}'.format(UD)
#        match(labels,stable_fps,mu,M)
        if not read_data:
            print('Generating ns  states for {} states'.format(UD))
            # calculates the 2-D and 11-D separatrix for each path (using bisection and analytic methods)
            # record time that it takes to calculate generate separatricies 
#            print('**********************sep_list_2D, sep_list_11D**********************')
#            print(sep_list_11D)
            sep_list_2D, sep_list_11D = TimeAndState(labels,stable_fps,mu,M,filename)
#            make_food_web(sep_list_2D, sep_list_11D,labels)
        else:
           # reads pre-generated analytic sepatrix data from the file 'sep_lists_analytic'
           print('Pulling nss states for {} states from files'.format(UD))
           sep_list_2D, sep_list_11D = FastTimeAndState(labels,stable_fps,mu,M,filename)
           if generateFigure:
               make_food_web(sep_list_2D, sep_list_11D,labels)
        sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D = NormAndSep(sep_list_11D,sep_list_2D,labels,stable_fps)
#        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
#        print(sep_matrix_2D)
#        print(sep_matrix_11D)
#        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        return sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D,labels   
    else:
#        
         labelStates = {0:list(range(0,5))}
         labels = labelStates.get(UD)
         print('Printing stein steady states')
         #file that contains precalculated separatricies(speeds up program time)
         filename = 'sep_lists_analyticS'
#         match(labels,stein_steady_states,mu,M)
         if not read_data:
            print('Generating  s steady states')
            # calculates the 2-D and 11-D separatrix for each path (using bisection and analytic methods)
            # record time that it takes to calculate generate separatricies 
            sep_list_2D, sep_list_11D = TimeAndState(labels,stein_steady_states,mu,M,filename)
            print('sep list 2D in generate stein steady states')
         else:
           print('Pulling sss from files')
           # reads pre-generated analytic sepatrix data from the file 'sep_lists_analytic'
           sep_list_2D, sep_list_11D = FastTimeAndState(labels,stein_steady_states,mu,M,filename)
           if generateFigure :
               make_food_web(sep_list_2D, sep_list_11D,labels)
#           results(generateFoodWeb)
#         print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
##         print(sep_matrix_2D)
##         print(sep_matrix_11D)
#         print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
         sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D = NormAndSep(sep_list_11D,sep_list_2D,labels,stein_steady_states)
         return sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D,labels



def Track(Stein,UD1,UD2):
    # load stein parameters
    param_list, ics = get_stein_parameters()
    # assign individual parameters from param_list
    bacteriaLabels, mu, M, eps = param_list
    #stable nonnegative fixedpoint list
    if Stein:
        state1 = stein_steady_states
    else:
        state1 = get_nonnegative_fps(mu, M,UD1,False)
    state2 = get_nonnegative_fps(mu, M,UD2,False)
    state1 = [l.tolist() for l in state1]
    state2 = [l.tolist() for l in state2]    
    result = []
    for element in state1:
        if element in state2:
            result.append(element)
    #find the indexes of the matching elements
    LargeD = []
    SmallD = []
    for element in state2:
        element = list(element)
    for element in result:
        LargeD =  [state2.index(element)] + LargeD
        SmallD =  [state1.index(element)] + SmallD  
    print(LargeD)
    print(SmallD)

  
def profile(fnc):
    #credit https://osf.io/upav8/
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def hammingLists(ordered_paths_2D,ordered_paths_11D,inpt): 
    ##2 UNSTABLE DIRECTIONS 
    #686: 25 -> 0 
    #25: 0 -> 25
    #43: 6 -> 0
    #6: 0 -> 6
    
    uniStableDict = { 'Dual Unstable1':686,'Dual Unstable2':25,'Unitary Unstable1':43, 'Unitary Unstable2':6,'Stein Unstable1':2,'Stein Unstable2':9 }
    filename = 'OrderedLists{}.txt'.format(inpt) 

    count = 0
    print('NORMEDss ------------------------------------')
    for key in ordered_paths_2D:
        count = count + 1
        if count == uniStableDict.get(inpt):
            orderedDict = {'Key2D':ordered_paths_2D[key],'Key11D':ordered_paths_2D[key]}
            with open(filename, 'wb') as file:
                file.write(pickle.dumps(orderedDict)) 

def OLists(norm_matrix_2D,norm_matrix_11D,sep_matrix_2D,sep_matrix_11D,labellist,hrz):
    ordered_paths_2D, path_lengths_2D = navigate_between_fps(norm_matrix_2D,False,labellist,hrz)
    ordered_paths_11D, path_lengths_11D = navigate_between_fps(norm_matrix_11D, False,labellist,hrz)
    total_hamming_distance = 0
    print('NORMED ------------------------------------')
    for key in ordered_paths_2D:
        print(key)
        print('  ', ordered_paths_2D[key])
        print('  ', ordered_paths_11D[key])
        hd = hamming_distance(ordered_paths_2D[key], ordered_paths_11D[key])
        print('   hamming distance:', hd)
        total_hamming_distance += hd
    print('TOTAL HAMMING DISTANCE: {}'.format(total_hamming_distance))
    print(); print()
    print('SEP ------------------------------------')
    ordered_paths_2D, path_lengths_2D = navigate_between_fps(sep_matrix_2D, False,labellist,hrz)
    ordered_paths_11D, path_lengths_11D = navigate_between_fps(sep_matrix_11D, False,labellist,hrz)
    total_hamming_distance = 0
    for key in ordered_paths_2D:
        print(key)
        print('  ', ordered_paths_2D[key])
        print('  ', ordered_paths_11D[key])
        hd = hamming_distance(ordered_paths_2D[key], ordered_paths_11D[key])
        print('   hamming distance:', hd)
        total_hamming_distance += hd
    print('TOTAL HAMMING DISTANCE: {}'.format(total_hamming_distance))


def CompareUnstableDirections(UD1,UD2):
    
    filenamesDict = {2:['OrderedListsDual Unstable1.txt','OrderedListsDual Unstable2.txt'],1: ['OrderedListsUnitary Unstable1.txt','OrderedListsUnitary Unstable2.txt'],'S' : ['OrderedListsStein Unstable1.txt','OrderedListsStein Unstable2.txt']}
    Flst = [filenamesDict.get(UD1),filenamesDict.get(UD2)]
    flattenFlst = [item for sublist in Flst for item in sublist]
    for i in range(len(flattenFlst)):
        filenames = flattenFlst[i]
        with open(filenames, "rb") as f:
            myobj = pickle.load(f)
            print('**************************************')
            print('**************************************')
            print('**************************************')
            print('**************************************')
            print(myobj)
            print('**************************************')
            print('**************************************')
            print('**************************************')
            print('**************************************')
            
            
def resultGenerator(OrderedPaths):
    return
            
        

#def instateProfile(Profile):
#    if Profile:
#        @profile
        
def main():

    #in S : 0,2;in 1UD :2 --> 0,0 --> 6;in 2UD : 2 --> 0, 0 --> 25
    generateAndRead = True
    if generateAndRead :
        read_data_from_file = None
        unstable_directions = None
        stein_states = None
        generateFigure = False
        options = {'generate stein states':[False,0,True,None], 'pull stein states from file':[True,0,True,generateFigure],'generate UD0 states':[False,0,False,None],'generate UD1 states':[False,1,False,None],'generate UD2 states':[False,2,False,None],'read UD0 states':[True,0,False,generateFigure],'read UD1 states':[True,1,False,generateFigure],'read UD2 states':[True,2,False,generateFigure] }
        optionInput = ['generate stein states','pull stein states from file','generate UD0 states','generate UD1 states','generate UD2 states','read UD0 states','read UD1 states','read UD2 states']
        criteria = [read_data_from_file ,unstable_directions,stein_states,generateAndRead]
        #CONTROL COMMANDS FOR GENERATING DATA
        ## 0 : generate stein states
        ## 1 : save stein states from file
        ## 2 : generate UD0 states
        ## 3 : generate UD1 states
        ## 4 : generate UD2 states
        ## 5 : save UD0 states
        ## 6 : save UD1 states
        ## 7 : save UD2 states
        inpt = 1
        safetyOn = True
        for i in range(len(criteria)):
            criteria[i] = options[optionInput[inpt]][i]
            if safetyOn :
                if inpt in (0,2,3,4) :
                    import sys
                    sys.exit()
           
        sep_matrix_2D,sep_matrix_11D,norm_matrix_2D,norm_matrix_11D,labels = Generate_And_Save_FixedPoints(criteria[0],criteria[1],criteria[2],criteria[3])
    trackstates = True
    if trackstates:
        Stein = None
        UD1 = None
        UD2 = None
        #CONTROL COMMANDS FOR TRACKING CHANGES IN STATES
        ## 0 : track changes from 0 to 1 steady states
        ## 1 : track changes from 0 to 2 steady states
        ## 2 : track changes from 1 to 2 steady states
        ## 3 : track changes from 0 to stein steady states
        ## 4 : track changes from 1 to stein steady states
        ## 5 : track changes from 2 to stein steady states
        options = {'01':[False,0,1],'02':[False,0,2],'12':[False,1,2],'0S':[True,None,0],'1S':[True,None,1],'2S':[True,None,2]}
        optionInput = ['01','02','12','0S','1S','2S']
        criteria = [Stein,UD1,UD2]
        for i in range(len(criteria)):
            criteria[i] = options[optionInput[4]][i]
        Track(criteria[0],criteria[1],criteria[2])
#       labellist = ['0','2']
        labellist = range(len(labels))
       
    orderedPaths = False
    if orderedPaths and generateAndRead:
    
        labellist = []
        labellist = list(map(str, labels))
        
        hrz = 2
        ordered_paths_2D, path_lengths_2D = navigate_between_fps(norm_matrix_2D, False,labellist,hrz)
        ordered_paths_11D, path_lengths_11D = navigate_between_fps(norm_matrix_11D, False,labellist,hrz) 
        # 0(S) = 6(U1) = 25(U2)
        # 2(S) = 0(U1) = 0(U2)
        ##CONTROL COMMANDS 
        ## (*) equivalent path 1
        ## (**) equivalent path 2
        ## 0 :  Save and load 25 - > 0 (*)
        ## 1 :  Save and load 0 - > 25 (**) 
        ## 2 :  Save and load 6 - > 0 (*)
        ## 3 :  Save and load 0 - > 6 (**)
        ## 4 :  Save and load 0 - > 2 (*)
        ## 5 :  Save and load 2 - > 0 (**)
        control = 2
        inptlist = ['Dual Unstable1','Dual Unstable2','Unitary Unstable1','Unitary Unstable2','Stein Unstable1','Stein Unstable2']
        if control in (0,1) and inpt != 7:
            print('Check inpt variable and control variable')
            import sys
            sys.exit()
        if control in (2,3) and inpt != 6:
            print('Check inpt variable and control variable')
            import sys
            sys.exit()
        if control in (4,5) and inpt != 1:
            print('Check inpt variable and control variable')
            import sys
            sys.exit()
        hammingLists(ordered_paths_2D,ordered_paths_11D,inptlist[control])
        CompareUnstableDirections(2,1)
        
        
        
    AllLists = False 
    if AllLists and generateAndRead:
        labellist = []
        labellist = list(map(str, labels))
        hrz =2
        OLists(norm_matrix_2D,norm_matrix_11D,sep_matrix_2D,sep_matrix_11D,labellist,hrz)

       


if __name__ == "__main__":
    main()




