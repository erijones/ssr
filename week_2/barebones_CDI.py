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
# ic_vars = (ic_data imported with import_data, ic_num you wish to retrieve)
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

