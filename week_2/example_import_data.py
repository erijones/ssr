#!/bin/python

import barebones_CDI as bb

# import data from stein_ic.csv and stein_parameters.csv files
var_data, ic_data = bb.import_data()
# turn "ugly" unmodified data into "nice" parsed and numpy array data
labels, mu, M, eps = bb.parse_data(var_data)
# save all of the parameters as a list
param_list = labels, mu, M, eps
# import the fourth initial condition
ic4 = bb.parse_ic((ic_data, 4), param_list)


print(labels)
print(M)
print(ic4)


