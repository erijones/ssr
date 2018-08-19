#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:33 2018

@author: parkershankin-clarke
"""

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
import pickle

import warnings
warnings.filterwarnings("ignore")

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

def get_nonnegative_stable_fps(mu, M):
    all_fps = get_all_steady_states(mu, M)
    fps = get_nonnegative_fixedpoints(all_fps)

    fp_list = []
    num_stable_fps = 0

    for fp in fps:
        # make sure all fixed points are actually fixed points
        output = integrand(fp, 0, mu, M)
        assert(all(abs(output) < 1e-6))

        is_stable = get_stability(fp, mu, M, almost_stable=0, substability=False)
        if is_stable:
            num_stable_fps += 1
            fp_list.append(fp)
    fp_list = np.array(fp_list)
    return fp_list

def get_nonnegative_unstable_fps(mu, M, num_unstable):
    all_fps = get_all_steady_states(mu, M)
    fps = get_nonnegative_fixedpoints(all_fps)

    fp_list = []
    num_stable_fps = 0
    for fp in fps:
        # make sure all fixed points are actually fixed points
        output = integrand(fp, 0, mu, M)
        assert(all(abs(output) < 1e-6))

        is_stable = get_stability(fp, mu, M, almost_stable=num_unstable, substability=False)
        if is_stable:
            num_stable_fps += 1
            fp_list.append(fp)
    fp_list = np.array(fp_list)
    return fp_list


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
    t = np.linspace(0, 10000, 10001)
    sol = odeint(integrand, point, t, args=(mu, M))
    while np.linalg.norm(sol[-1] - sol[-100]) > 1e-8:
        error = np.linalg.norm(sol[-1] - sol[-2])
        if verbose:
            print(t[-1], error)
        t = np.linspace(t[-1], t[-1] + 10000, 10001)
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
        if verbose:
            print('basin of attraction for xa ends at p={:.5}'.format(separatrix_xa))
            print('basin of attraction for xb ends at p={:.5}'.format(separatrix_xb))
        if sum(went_to_neither) > 0:
            neither_index = went_to_neither.index(True)
            neither_val = get_steady_state(points[neither_index], mu, M)
            if len(mu) == 2:
                print('    instead {}D traj goes to steady state at {}'.format(len(mu), neither_val))

            if len(mu) > 2:
                unstable_2_fps = get_nonnegative_unstable_fps(mu, M, 2)
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


def SSR(xa,xb,mu,M):
    """This function performs a steady state reduction by taking in the
    relevant parameters, then performing the relevant operations, and finally
    returning the steady state reduced forms of the parameters "nu" and "L" """

    nu = np.array([np.dot(xa, mu),
                  np.dot(xb, mu)])

    L = np.array([[np.dot(xa.T,np.dot(M,xa)), np.dot(xa.T,np.dot(M,xb))],
                 [np.dot(xb.T,np.dot(M,xa)), np.dot(xb.T,np.dot(M,xb))]])
    return nu,L

def get_stein_steady_states(fps):
    """This function imports  values from the dictionary that contains "stein's steady states" 
    and  also imports steady states with up to 2 unstable directions (almost_stable=2). This 
    code outputs matching steady states between stein's set and the calculated set"""
    stein_dict = bb.get_all_ss()
    stein_ss = np.array([stein_dict[val] for val in stein_dict])

    final_list = np.array([np.zeros(len(stein_ss[0])) for i in range(5)])
    iterations_list =  list(itertools.product(stein_ss, fps))
    for i in range(len(iterations_list )):
        compare_lists = iterations_list[i]
        if np.linalg.norm(compare_lists[0] - compare_lists[1]) < .001:
            for j in range(5):
                if np.linalg.norm(compare_lists[1] - stein_ss[j]) < .001:
                    final_list[j] = compare_lists[1]
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
            po = get_separatrix_point(xa, xb, mu, M, 201)
            break
    verbose = True
    if verbose:
        N = len(mu)
        if isinstance(po, float):
            print('    {}D separatrix at p={}'.format(N, po))

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

def make_food_web(sep_list_2D, sep_list_11D):
    """This function simulates a network of steady-states solutions and their sepatricies. """
    fig, ax = plt.subplots(figsize=(6,6))
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
    circ_xx = []; circ_yy = []
    #print('colorwheel is', wheels[1][0].get_color())
    labels = ['A', 'B', 'C', 'D', 'E']
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
    filename = 'figs/example_food_web_4.pdf'
    plt.savefig(filename)
    print('saved fig to {}'.format(filename))
    return


###############################################################################


## MAIN FUNCTION

param_list, ics = get_stein_parameters()
labels, mu, M, eps = param_list

stable_fps = get_nonnegative_stable_fps(mu, M)
unstable_2_fps = get_nonnegative_unstable_fps(mu, M, 2)
stein_steady_states = get_stein_steady_states(unstable_2_fps)
# note the order of stein_steady_states correponds to [A B C D E] of stein_dict
stable_indices = []
for stab_fp in stable_fps:
    for i,stein_fp in enumerate(stein_steady_states):
        if np.linalg.norm(stab_fp - stein_fp) < .001:
            stable_indices.append(i)
# the truly stable steady states are indices 0 and 2 (A and C)

sep_list_2D = {}
sep_list_11D = {}

labels = ['A', 'B', 'C', 'D', 'E']
read_data = True 
if not read_data:
    combos = list(itertools.combinations(range(5), 2))
    for i,j in combos:
        print(labels[i], labels[j])
        ssa_11 = stein_steady_states[i]
        ssb_11 = stein_steady_states[j]
        temp_separatrix_11D = bisection(ssa_11, ssb_11, .0001, mu, M)

        nu, L = SSR(ssa_11, ssb_11, mu, M)
        ssa_2 = np.array([1, 0])
        ssb_2 = np.array([0, 1])
        temp_separatrix_2D = bisection(ssa_2, ssb_2, .0001, nu, L)

        sep_list_2D[(i, j)] = temp_separatrix_2D
        sep_list_11D[(i, j)] = temp_separatrix_11D
        try:
            sep_list_2D[(j, i)] = 1 - temp_separatrix_2D
        except TypeError:
            sep_list_2D[(j, i)] = tuple(1 - val for val in temp_separatrix_2D)
        try:
            sep_list_11D[(j, i)] = 1 - temp_separatrix_11D
        except TypeError:
            sep_list_11D[(j, i)] = tuple(1 - val for val in temp_separatrix_11D)
    with open('data/sep_lists', 'wb') as f:
        pickle.dump((sep_list_2D, sep_list_11D), f)
else:
    with open('data/sep_lists', 'rb') as f:
        sep_list_2D, sep_list_11D = pickle.load(f)
        verbose = True
        labels = ['A', 'B', 'C', 'D', 'E']
        if verbose:
            for i in range(5):
                for j in range(5):
                    if i < j:
                        print(labels[i], labels[j])
                        for N, sep_list in zip([2, 11], [sep_list_2D, sep_list_11D]):
                            p = sep_list[(i, j)]
                            if isinstance(p, float):
                                print('    {}D separatrix at p={}'.format(N, p))
                            else:

                                print('    {}D separatrices at p={} and p={}'.format(N, p[0], p[1]))
                                
  
        ##The lines of code in under the following "if True" statement  : 
        #1.) Collect the permuations of the 11D and 2D separatricies and put them into an two seperate arrays that are subsequently be resized as two nxn matricies. 
        # If separatrices disagree then the corresponding element is entered as a none-type.
        #2.)Instead of collecting the separatrix p, the 11D and 2D norms (i.e. p*norm(ssb-ssa)) are collected as two seperate arrays that subsequently resized as two nxn matricies. 
if True: 
    
    #arrays intialized for part1.) (look above for more information)
    sep_matrix_11D = np.array([])
    sep_matrix_2D = np.array([])
    
    #arrays intialized for part2.) (look above for more information)
    norm_matrix_2D = np.array([])
    norm_matrix_11D = np.array([])
    
    #The number of steady states in the system
    numss = 5
    
    #11-D 
    if True:
        for i in range(numss):
            for j in range(numss):
                if i != j:
                    for N, sep_list in zip([2, 11], [sep_list_11D]):
                        p = sep_list[(i, j)]
                        ssa = stein_steady_states[i]
                        ssb = stein_steady_states[j]
                        if isinstance(p, float):
                            sep_matrix_11D = np.append(sep_matrix_11D, p)
                            norm = p * np.linalg.norm(ssb - ssa)
                            norm_matrix_11D = np.append(norm_matrix_11D,norm)
                        else:
                            sep_matrix_11D = np.append(sep_matrix_11D, None)
                            norm_matrix_11D = np.append(sep_matrix_11D,None)
                elif i == j:
                     p=0
                     sep_matrix_11D = np.append(sep_matrix_11D,p)
                     norm_matrix_11D = np.append(norm_matrix_11D,p)
    #2-D        
    if True:
        for i in range(numss):
            for j in range(numss):
                if i != j:
                    for N, sep_list in zip([2, 11], [sep_list_2D]):
                        p = sep_list[(i, j)]
                        if isinstance(p, float):
                            sep_matrix_2D = np.append(sep_matrix_2D, p)
                            norm = p * np.linalg.norm(np.array([0,1]) - np.array([1,0]))
                            norm_matrix_2D = np.append(norm_matrix_2D,norm)   
                        else:
                            sep_matrix_2D = np.append(sep_matrix_2D, None)
                            norm_matrix_2D = np.append(norm_matrix_2D,None)
                elif i == j:
                    p=0
                    sep_matrix_2D = np.append(sep_matrix_2D,p)
                    norm_matrix_2D = np.append(norm_matrix_2D,p)
    
    ##reformat both arrays into  numss x numss matricies:
    
    #arrays for part1.) (look above for more information)                  
    sep_matrix_2D = np.resize(sep_matrix_2D,(5,5))
    sep_matrix_11D = np.resize(sep_matrix_11D,(5,5))
    
    #arrays for part2.) (look above for more information)  
    norm_matrix_2D = np.resize(norm_matrix_2D,(5,5))
    norm_matrix_11D = np.resize(norm_matrix_11D,(5,5))



if True:
    make_food_web(sep_list_2D, sep_list_11D)

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