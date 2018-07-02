#!/bin/python
#
# Version 0.1 
# This file contains the barebones functions required to generate figures that
# do not involve sporulation or mutation for the paper 'In silico analysis of
# C. difficile infection: remediation techniques and biological adaptations' by
# Eric Jones and Jean Carlson. The code is still in a preliminary form. This
# code is covered under GNU GPLv3.
#
# Send questions to Eric Jones at ewj@physics.ucsb.edu
#
###############################################################################

import numpy as np
import scipy.integrate as integrate
np.set_printoptions(suppress=True, precision=5)

# import data from Stein paper for parameters and initial conditions
def import_data():
    # import messy variables
    with open('stein_parameters.csv','r') as f:
        var_data = [line.strip().split(",") for line in f]
    # import messy initial conditions
    with open('stein_ic.csv','r') as f:
        ic_data = [line.strip().split(",") for line in f]
    return var_data[1:], ic_data

# extract messy (i.e. string) parameters, cast as floats, and then numpy arrays
def parse_data(var_data):
    # extract microbe labels, to be placed in legend
    labels = [label.replace("_"," ") for label in var_data[-1] if label.strip()]
    # extract M, mu, and eps from var_data
    str_inter   = [elem[1:(1+len(labels))] for elem in var_data][:-1]
    str_gro     = [elem[len(labels)+1] for elem in var_data][:-1]
    str_sus     = [elem[len(labels)+2] for elem in var_data][:-1]
    float_inter = [[float(value) for value in row] for row in str_inter]
    float_gro   = [float(value) for value in str_gro]
    float_sus   = [float(value) for value in str_sus]
    # convert to numpy arrays
    M   = np.array(float_inter)
    mu  = np.array(float_gro)
    eps = np.array(float_sus)
    # swap C. diff so that it is the last element
    c_diff_index                     = 8
    M[:, [c_diff_index, -1]]         = M[:, [-1, c_diff_index]]
    M[[c_diff_index, -1], :]         = M[[-1, c_diff_index], :]
    mu[c_diff_index], mu[-1]         = mu[-1], mu[c_diff_index]
    eps[c_diff_index], eps[-1]       = eps[-1], eps[c_diff_index]
    labels[c_diff_index], labels[-1] = labels[-1], labels[c_diff_index]
    c_diff_index                     = labels.index("Clostridium difficile")

    return labels, mu, M, eps

# extract messy (i.e. string) ICs, cast as floats, and then numpy arrays
def parse_ic(ic_vars, param_list):
    ic_data, ic_num      = ic_vars
    labels, mu, M, eps   = param_list
    ic_list_str          = [[elem[i] for elem in ic_data][5:-2] for i in \
                             range(1,np.shape(ic_data)[1]) if float(ic_data[3][i])==0]
    ic_list_float        = [[float(value) for value in row] for row in ic_list_str]

    ic                   = np.array(ic_list_float[:][ic_num])
    old_c_diff_index     = 8
    c_diff_index         = labels.index("Clostridium difficile")

    ic[c_diff_index], ic[old_c_diff_index] = ic[old_c_diff_index], ic[c_diff_index]
    return ic

def solve(ic, t_end, param_list, interventions):
    u_params, cd_inoculation, transplant_params = extract_interventions(interventions)

    if (not cd_inoculation) and (not transplant_params):
        t = np.linspace(0, t_end, num=1001)
        y = integrate.odeint(integrand, ic, t, args=(param_list, u_params), atol=1e-12)
        return t,y

    if transplant_params:
        t_type, t_size, t_time = transplant_params
        if t_time == 0: t_time = 1e-6
        t01 = np.linspace(0, t_time, num=101)
        t12 = np.linspace(t_time, t_end, num=101)
        y01 = integrate.odeint(integrand, ic, t01, args=(param_list, u_params), atol=1e-12)
        new_ic = y01[-1] + np.array([t_size*x for x in t_type])
        y12 = integrate.odeint(integrand, new_ic, t12, args=(param_list, u_params), atol=1e-12)


    if cd_inoculation:
        t01 = np.linspace(0, cd_inoculation, num=101)
        t12 = np.linspace(cd_inoculation, t_end, num=101)
        y01 = integrate.odeint(integrand, ic, t01, args=(param_list, u_params), atol=1e-12)
        new_ic = y01[-1] + np.append(np.zeros(len(y01[0]) - 1), 10**-10)
        y12 = integrate.odeint(integrand, new_ic, t12, args=(param_list, u_params), atol=1e-12)

    return np.concatenate((t01,t12)), np.vstack((y01,y12))

def extract_interventions(interventions):
    try: u_params = interventions['u_params']
    except KeyError: u_params = None
    try: cd_inoculation = interventions['CD']
    except KeyError: cd_inoculation = None
    try: transplant_params = interventions['transplant']
    except KeyError: transplant_params = None
    return u_params, cd_inoculation, transplant_params

def integrand(Y, t, param_list, u_params):
    labels, mu, M, eps = param_list
    return (np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) +
               u(t, u_params)*np.dot(np.diag(eps), Y))

def u(t, u_params):
    if not u_params:
        return 0

    concentration, duration = u_params
    if t < duration:
        return concentration
    else:
        return 0

def get_all_ss():
    var_data, ic_data = import_data()
    labels, mu, M, eps = parse_data(var_data)
    param_list = (labels, mu, M, eps)

    # 'SS attained': (IC num, if CD exposure, if RX applied)
    ss_conditions = {'A': (0, True, False), 'B': (0, False, False),
                     'C': (4, False, False), 'D': (4, True, True),
                     'E': (4, False, True)}
    ss_list = {}
    for ss in ss_conditions:
        ic_num, if_CD, if_RX = ss_conditions[ss]
        ic = parse_ic((ic_data, ic_num), param_list)

        if (not if_CD) and (not if_RX): interventions = {}
        if (if_CD) and (not if_RX): interventions = {'CD': 10}
        if (not if_CD) and (if_RX): interventions = {'u_params': (1, 1)}
        if (if_CD) and (if_RX): interventions = {'u_params': (1, 1), 'CD': 5}

        t, y = solve(ic, 5000, param_list, interventions)
        ss_list[ss] = np.array([max(yy, 0) for yy in y[-1]])

    return ss_list

