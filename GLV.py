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
import matplotlib.pyplot as plt

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
    verbose = True
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

    verbose = False
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

def example_food_web(separatrices):
    """This function simulates a network of steady-states solutions and their sepatricies. """
    fig, ax = plt.subplots()
    wheels = []
    labels = ['A', 'B', 'C', 'D', 'E']
    for i in range(len(labels)):
        # save the order of colors when plotting
        wheels.append(ax.plot([0], [0], linewidth=0.0))

    import matplotlib.patches as mpatches
    circ_size = 0.15
    # circs are a matplotlib object
    # xx and yy are locations of these circles (to be plotted)
    circs = []; xx = []; yy = [];
    print('colorwheel is', wheels[1][0].get_color())
    for i in range(len(labels)):
        xx.append(np.sin(2*np.pi*i/len(labels)))
        yy.append(np.cos(2*np.pi*i/len(labels)))
        # here I append a "Circle" object, and say that its color is what I
        # saved earlier (in wheels)
        circs.append(plt.Circle((xx[-1], yy[-1]), circ_size, lw=0,
                     color=wheels[i][0].get_color()))

    # plot all of the circles in ax
    for i in range(len(labels)):
        ax.add_artist(circs[i])

    # add lines connecting circles
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i < j:
                arrow_color = wheels[0][0].get_color()
                curve_type = 'arc3,rad=0'
                thickness = 1.6
                # make arrows pointing from one circle to another
                ax.annotate("", xy=(xx[i],yy[i]), xytext=(xx[j],yy[j]),
                        zorder=1,
                        arrowprops = dict(arrowstyle='-', facecolor=arrow_color,
                            edgecolor=arrow_color, alpha=.5*thickness,
                            patchA=mpatches.Circle((xx[i], yy[i]), circ_size), shrinkA=23,
                        patchB=mpatches.Circle((xx[j], yy[j]), circ_size), shrinkB=27,
                        connectionstyle=curve_type, linewidth = thickness))#2*abs(M[i][j])))

    # add example point between two lines
    
    if  weigsep_list[0] < .5:
        cgr = 'g'
    else :
        cgr = 'tab:red'
    
    if  weigsep_list[1] < .5:
        cgb = 'g'
    else :
        cgb = 'tab:blue'   
        
    if  weigsep_list[2] < .5:
        cro = 'tab:red'
    else :
        cro = 'tab:orange'       
    if weigsep_list[3] < .5:
        cpg = 'tab:purple'
    else :
        cpg = 'g'
    if weigsep_list[4] < .5:
        cpo = 'tab:orange'
    else :
        cpo = 'tab:purple'
    if weigsep_list[5] < .5:
        cbr = 'tab:blue'
    else :
        cbr = 'tab:red'
    if weigsep_list[6] < .5:
        cbp = 'tab:blue'
    else :
        cbp = 'tab:purple'
    if weigsep_list[7] < .5:
        cbo = 'tab:orange'
    else :
        cbo = 'tab:blue'
    if weigalt_list[0] < .5:
        cpr = 'tab:purple'
    else :
        cpr = 'tab:red'
    if weigalt_list[2]< .5:
        cog = 'tab:orange'
    else :
        cog = 'tab:green'
    if weigalt_list[3]< .5:
        cogg = 'tab:orange'
    else :
        cogg = 'tab:green'
#    
#    
    
#    1.) blue to red    
    p0 = weigsep_list[0]
    plt.plot( (p0*xx[-2] + (1-p0)*xx[-3]), (p0*yy[-2] + (1-p0)*yy[-3]), marker='.',
            color= cgr, markersize=20, zorder=5)
    
    p1 = weigsep_list[1]
     #2.) blue to green (p0*xx[0] + (1-p0)*xx[2]), (p0*yy[0] + (1-p0)*yy[2])
    plt.plot( (p1*xx[0] + (1-p1)*xx[2]), (p1*yy[0] + (1-p1)*yy[2]), marker='.',
            color= cgb, markersize=20, zorder=5)
   
    #3.) orange to red (p2*xx[1] + (1-p2)*xx[3]), (p2*yy[1] + (1-p2)*yy[3])
    p2 = weigsep_list[2]
    plt.plot((p2*xx[1] + (1-p2)*xx[3]), (p2*yy[1] + (1-p2)*yy[3]), marker='.',
            color=cro, markersize=20, zorder=5)
    
    #4.) purple to green(p3*xx[2] + (1-p3)*xx[4]), (p4*yy[2] + (1-p3)*yy[4])
    p3 = weigsep_list[3]
    plt.plot((p3*xx[2] + (1-p3)*xx[4]), (p3*yy[2] + (1-p3)*yy[4]), marker='.',
            color=cpg , markersize=20, zorder=5)
###Don't delete
#    5.) purple to orange (p4*xx[-1] + (1-p4)*xx[1]), (p4*yy[-1] + (1-p4)*yy[1])
    p4 = weigsep_list[4]
    plt.plot((p4*xx[-1] + (1-p4)*xx[1]), (p4*yy[-1] + (1-p4)*yy[1]), marker='.',
            color=cpo, markersize=20, zorder=5)
    #blue to red(p5*xx[-2] + (1-p5)*xx[0]), (p5*yy[-2] + (1-p5)*yy[0])
    p5 = weigsep_list[5]
    plt.plot((p5*xx[-2] + (1-p5)*xx[0]), (p5*yy[-2] + (1-p5)*yy[0]), marker='.',
            color=cbr, markersize=20, zorder=5)
    #blue to purple (p6*xx[-1] + (1-p6)*xx[0]), (p6*yy[-1] + (1-p6)*yy[0])
    p6 = weigsep_list[6]
    plt.plot((p6*xx[-1] + (1-p6)*xx[0]), (p6*yy[-1] + (1-p6)*yy[0]), marker='.',
            color=cbp, markersize=20, zorder=5)
    # blue to orange (p7*xx[0] + (1-p7)*xx[1]), (p7*yy[0] + (1-p7)*yy[1])
    p7 = weigsep_list[7]
    plt.plot((p7*xx[0] + (1-p7)*xx[1]), (p7*yy[0] + (1-p7)*yy[1]), marker='.',
            color=cbo, markersize=20, zorder=5)
    
    #purple to red (p8*xx[-2] + (1-p8)*xx[4]), (p8*yy[-2] + (1-p8)*yy[4])
    p8 = weigalt_list[0]
    plt.plot((p8*xx[-2] + (1-p8)*xx[4]), (p8*yy[-2] + (1-p8)*yy[4]), marker='.',
            color=cpr, markersize=20, zorder=5)
    p8 = weigalt_list[1]*.9
    plt.plot((p8*xx[-2] + (1-p8)*xx[4]), (p8*yy[-2] + (1-p8)*yy[4]), marker='.',
            color=cpr, markersize=20, zorder=5)

  
#    #orange to green (p9*xx[2] + (1-p9)*xx[-4]), (p9*yy[2] + (1-p9)*yy[-4])
    p9 = weigalt_list[2]
    plt.plot((p9*xx[2] + (1-p9)*xx[-4]), (p9*yy[2] + (1-p9)*yy[-4]), marker='.',
            color=cog, markersize=20, zorder=5)
    p10 = weigalt_list[3]
    plt.plot((p10*xx[2] + (1-p10)*xx[-4]), (p10*yy[2] + (1-p10)*yy[-4]), marker='.',
            color=cogg, markersize=20, zorder=5)
    
    edge = (1+circ_size)*1.02
    plt.axis([-edge, edge, -edge, edge])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
#    filename = 'figs/example_food_web.pdf'
#    plt.savefig(filename)
#    print('saved fig to {}'.format(filename))


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
if False:
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
sepp_list = []
twodsepp_list = []
elevendsepp_list = []
if True:
    combos = list(itertools.combinations(range(5), 2))
    alt_list = []
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
            
            
            twodsepp_list = twodsepp_list + [temp_separatrix_2D]
            elevendsepp_list = elevendsepp_list + [temp_separatrix_11D]
            
            if type(bisected_separatrix_11D) is float :
                print('float is')
                print(bisected_separatrix_11D)
                sepp_list = sepp_list + [bisected_separatrix_11D]
            elif type(bisected_separatrix_11D) is tuple :
                print('typle is')
                print(np.array(bisected_separatrix_11D))
                conv_elem = np.array(bisected_separatrix_11D)
                if conv_elem[0] == conv_elem[1]:
                    print('conv_elem[0] == conv_elem[1]')
                    sepp_list = [conv_elem[0]] + sepp_list
                if conv_elem[0] != conv_elem[1]:
                    ('conv_elem[0] != conv_elem[1]')
                    print('conv element 1 and two is equal to')
                    print(conv_elem[0])
                    print(conv_elem[1])
                    alt_list  = [conv_elem[0]] +  [conv_elem[1]] + alt_list
                    
       
l = np.array([.84,.99])         
sepp_list = np.array(sepp_list)
sepp_list = list(sepp_list)



weigsep_list = []
weigalt_list = []
for elem in sepp_list:
   
    if elem > .9:
        new_elem = .8
        weigsep_list = weigsep_list + [new_elem]
        
   
    elif elem <.1:
        elem = .2
        weigsep_list = weigsep_list + [elem]
   
    else :
        elem = elem
        weigsep_list = weigsep_list + [elem]

for element in alt_list:
   
    if element > .9:
        element = .8
        weigalt_list = weigalt_list + [element]
    elif element <.1:
        element = .2
        weigalt_list = weigalt_list + [element]
   
    else :
        element = element
        weigalt_list = weigalt_list + [element]
        
        print('weigalt_list')


print(weigalt_list)
if True:
    example_food_web(weigsep_list)
    import sys
    sys.exit()

#If this block is designated as True then all of the pairs of Stein's steady states are found. Using the bisection method the separatrices are found
# if the steady states that correspond to the sepatratrices have meaningful trajectories then the are subsequently passed as arguments in the function
# get_relative_deviation.
if False:
    combos = list(itertools.combinations(range(5), 2))
    for i,j in combos:
        ssa = stein_steady_states[i]
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
            print('--------')
        else:
            print(pstar)
            print('--')

