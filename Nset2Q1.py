#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 04:00:40 2018

@author: parkershankin-clarke
"""

from scipy.optimize import fsolve
from sympy import sin, cos, Matrix
from sympy.abc import x,y,z
from sympy.solvers import solve
from sympy import Symbol
import numpy as np
import math

a = 1
b = 2
c = 3
d = 4
e = 5
f = 6

#The initial GLV system of equations
eqn1 = x * a + b * x*x + c * y * x
eqn2 = y * d + e * y*y + f * y * x

#Find the fixed points which is a data type that is a list of a [list, set]
fixed_points = list(solve([eqn1, eqn2], x, y, set=True))

# parse the above data type starting at the set into another list
L_fixed_points = list(fixed_points[1])


#Extract all four fixed points
first_fixedpoint = list(L_fixed_points[0])
second_fixedpoint= list(L_fixed_points[1])
third_fixedpoint= list(L_fixed_points[2])
forth_fixedpoint= list(L_fixed_points[3])


#Extract the first and second enrtry from each fixed point
x1 = first_fixedpoint[0]
y1 = first_fixedpoint[1]

x2 = second_fixedpoint[0]
y2 = second_fixedpoint[1]

x3 = third_fixedpoint[0]  
y3 = third_fixedpoint[1]

x4 = forth_fixedpoint[0]
y4 = forth_fixedpoint[1]

#Calculate the Jacobian
X = Matrix([eqn1,eqn2])
Y = Matrix([x,y])
jac = X.jacobian(Y)

#Convert the sympy.matrices.dense.MutableDenseMatrix data structure to a list so it is callable 
L_jac = list(jac)

#extract each entry from jacobian 
#parse list to string
first_entryjacobian = str(L_jac[0])
second_entryjacobian =str(L_jac[1])
third_entryjacobian = str(L_jac[2])
forth_entryjacobian = str(L_jac[3])



#1.) substitute first set of fixed points into first jacobian entry (0,-d/e)

s_x_third_fixedpoint = str(third_fixedpoint[0])
s_y_third_fixedpoint = str(third_fixedpoint[1])


if 'y' or 'x' in first_entryjacobian :
    one_jmatrix_first_entryjacobian_aftersub = first_entryjacobian.replace('y',s_y_third_fixedpoint)
    one_jmatrix_first_entryjacobian_aftersub = one_jmatrix_first_entryjacobian_aftersub.replace('x',s_x_third_fixedpoint)
# 1.) substitute second set of fixed points into first jacobian entry (0,-d/e)
if 'y' or 'x' in second_entryjacobian :
    one_jmatrix_second_entryjacobian_aftersub  = second_entryjacobian.replace('y',s_y_third_fixedpoint)
    one_jmatrix_second_entryjacobian_aftersub   =  one_jmatrix_second_entryjacobian_aftersub.replace('x',s_x_third_fixedpoint)
#1.)substitute third set of fixed points into first jacobian entry (0,-d/e)
if 'y' or 'x' in third_entryjacobian :
   one_jmatrix_third_entryjacobian_aftersub = third_entryjacobian.replace('y',s_y_third_fixedpoint)
   one_jmatrix_third_entryjacobian_aftersub = one_jmatrix_third_entryjacobian_aftersub.replace('x',s_x_third_fixedpoint)   
#1.)substitute forth set of fixed points into first jacobian entry (0,-d/e)
if 'y'or 'x' in forth_entryjacobian :
     one_jmatrix_forth_entryjacobian_aftersub = forth_entryjacobian.replace('y',s_y_third_fixedpoint)
     one_jmatrix_forth_entryjacobian_aftersub = one_jmatrix_forth_entryjacobian_aftersub.replace('x',s_x_third_fixedpoint)


#2.) substitute first set of fixed points into second jacobian entry (-a/b, 0)
     
s_x_forth_fixedpoint = str(forth_fixedpoint[0])
s_y_forth_fixedpoint = str(forth_fixedpoint[1])

if 'y' or 'x' in first_entryjacobian :
    two_jmatrix_first_entryjacobian_aftersub = first_entryjacobian.replace('x',s_x_forth_fixedpoint)
    two_jmatrix_first_entryjacobian_aftersub = two_jmatrix_first_entryjacobian_aftersub.replace('y',s_y_forth_fixedpoint)     
#2.) substitute second set of fixed points into second jacobian entry (-a/b, 0)
if 'y' or 'x' in second_entryjacobian :
    two_jmatrix_second_entryjacobian_aftersub  = second_entryjacobian.replace('x',s_x_forth_fixedpoint)
    two_jmatrix_second_entryjacobian_aftersub   =  two_jmatrix_second_entryjacobian_aftersub.replace('y',s_y_forth_fixedpoint)
#2.) substitute third set of fixed points into second jacobian entry (-a/b, 0)
if 'y' or 'x' in third_entryjacobian :
  two_jmatrix_third_entryjacobian_aftersub = third_entryjacobian.replace('x',s_x_forth_fixedpoint)
  two_jmatrix_third_entryjacobian_aftersub = two_jmatrix_third_entryjacobian_aftersub.replace('y',s_y_forth_fixedpoint)
#2.) substitute forth set of fixed points into second jacobian entry (-a/b, 0)
if 'y'or 'x' in forth_entryjacobian :
    two_jmatrix_forth_entryjacobian_aftersub = forth_entryjacobian.replace('x',s_x_forth_fixedpoint)
    two_jmatrix_forth_entryjacobian_aftersub = two_jmatrix_forth_entryjacobian_aftersub.replace('y',s_y_forth_fixedpoint)    
    
    
#3.) substitute first set of fixed points into second jacobian entry((-a*e + c*d)/(b*e - c*f), (a*f - b*d)/(b*e - c*f))
    
s_x_first_fixedpoint = str(first_fixedpoint[0])
s_y_first_fixedpoint = str(first_fixedpoint[1])

if 'y' or 'x' in first_entryjacobian :
    three_jmatrix_first_entryjacobian_aftersub = first_entryjacobian.replace('x',s_x_first_fixedpoint)
    three_jmatrix_first_entryjacobian_aftersub = three_jmatrix_first_entryjacobian_aftersub.replace('y',s_y_first_fixedpoint)
#3.) substitute second set of fixed points into second jacobian entry ((-a*e + c*d)/(b*e - c*f), (a*f - b*d)/(b*e - c*f))
if 'y' or 'x' in second_entryjacobian :
    three_jmatrix_second_entryjacobian_aftersub  = second_entryjacobian.replace('x',s_x_first_fixedpoint)
    three_jmatrix_second_entryjacobian_aftersub   =  three_jmatrix_second_entryjacobian_aftersub.replace('y',s_y_first_fixedpoint)
#3.) substitute third set of fixed points into second jacobian entry ((-a*e + c*d)/(b*e - c*f), (a*f - b*d)/(b*e - c*f))
if 'y' or 'x' in third_entryjacobian :
   three_jmatrix_third_entryjacobian_aftersub = third_entryjacobian.replace('x',s_x_first_fixedpoint)
   three_jmatrix_third_entryjacobian_aftersub =three_jmatrix_third_entryjacobian_aftersub.replace('y',s_y_first_fixedpoint)
#3.) substitute forth set of fixed points into second jacobian entry ((-a*e + c*d)/(b*e - c*f), (a*f - b*d)/(b*e - c*f))
if 'y'or 'x' in forth_entryjacobian :
     three_jmatrix_forth_entryjacobian_aftersub = forth_entryjacobian.replace('x',s_x_first_fixedpoint)
     three_jmatrix_forth_entryjacobian_aftersub = three_jmatrix_forth_entryjacobian_aftersub.replace('y',s_y_first_fixedpoint)   
    

#4.) substitute first set of fixed points into second jacobian entry (0, 0)
     
s_x_second_fixedpoint = str(second_fixedpoint[0])
s_y_second_fixedpoint = str(second_fixedpoint[1])

if 'y' or 'x' in first_entryjacobian :
    four_jmatrix_first_entryjacobian_aftersub = first_entryjacobian.replace('y',s_y_second_fixedpoint)
    four_jmatrix_first_entryjacobian_aftersub = four_jmatrix_first_entryjacobian_aftersub.replace('x',s_x_second_fixedpoint)
#4.) substitute second set of fixed points into second jacobian entry (0, 0)
if 'y' or 'x' in second_entryjacobian :
    four_jmatrix_second_entryjacobian_aftersub  = second_entryjacobian.replace('y',s_y_second_fixedpoint)
    four_jmatrix_second_entryjacobian_aftersub   =  four_jmatrix_second_entryjacobian_aftersub.replace('x',s_x_second_fixedpoint)
#4.) substitute third set of fixed points into second jacobian entry (0, 0)
if 'y' or 'x' in third_entryjacobian :
  four_jmatrix_third_entryjacobian_aftersub = third_entryjacobian.replace('y',s_y_second_fixedpoint)
  four_jmatrix_third_entryjacobian_aftersub = four_jmatrix_third_entryjacobian_aftersub.replace('x',s_x_second_fixedpoint)
#4.) substitute forth set of fixed points into second jacobian entry (0, 0)
if 'y'or 'x' in forth_entryjacobian :
    four_jmatrix_forth_entryjacobian_aftersub = forth_entryjacobian.replace('y',s_y_second_fixedpoint)
    four_jmatrix_forth_entryjacobian_aftersub = four_jmatrix_forth_entryjacobian_aftersub.replace('x',s_x_second_fixedpoint)

# Make 1 through 4 into a list

#1.)
list1_1 = [one_jmatrix_first_entryjacobian_aftersub, one_jmatrix_second_entryjacobian_aftersub]
list1_2 = [one_jmatrix_third_entryjacobian_aftersub,one_jmatrix_forth_entryjacobian_aftersub]
#2.)
list2_1 =[two_jmatrix_first_entryjacobian_aftersub,two_jmatrix_second_entryjacobian_aftersub]
list2_2 =[two_jmatrix_third_entryjacobian_aftersub,two_jmatrix_forth_entryjacobian_aftersub]
#3.)
list3_1 =[three_jmatrix_first_entryjacobian_aftersub,three_jmatrix_second_entryjacobian_aftersub]
list3_2 =[three_jmatrix_third_entryjacobian_aftersub,three_jmatrix_forth_entryjacobian_aftersub]
#4.)
list4_1 = [four_jmatrix_first_entryjacobian_aftersub,four_jmatrix_second_entryjacobian_aftersub]
list4_2 = [four_jmatrix_third_entryjacobian_aftersub,four_jmatrix_forth_entryjacobian_aftersub]
# Create 4 stability matrices
 
#1.)
SMatrix1 = Matrix([list1_1,list1_2])
#2.)
SMatrix2  = Matrix([list2_1,list2_2])
#3.)
SMatrix3 = Matrix([list3_1,list3_2])
#4.)  
SMatrix4 = Matrix([list4_1,list4_2])




# Find eigenvalues
# Extract keys
#Parse keys to list
#Extract both eigenvalues
#1.)
eg1 =  SMatrix1.eigenvals()
eg1k = eg1.keys()
l_eg1k = list(eg1.keys())
eg1_1 = l_eg1k[0]
eg1_2 = l_eg1k[1]

#2.)
eg2 = SMatrix2.eigenvals()
eg2k = eg2.keys()
l_eg2k = list(eg2.keys())
eg2_1 = l_eg2k[0]
eg2_2 = l_eg2k[1]

#3.)
eg3 =SMatrix3.eigenvals()
eg3k = eg3.keys()
l_eg3k = list(eg3k)
eg3_1 = l_eg3k[0]
eg3_2 = l_eg3k[1]

#4.) 
eg4 = SMatrix4.eigenvals()
eg4k = eg4.keys()
l_eg4k = list(eg4k)
eg4_1 = l_eg4k[0]
eg4_2 = l_eg4k[1]

#Convert eigenvlaues to strings 
s_eg1_1 =  str(eg1_1) 
s_eg1_2 =  str(eg1_2) 



s_eg2_1 = str(eg2_1) 
s_eg2_2 = str(eg2_2)


s_eg3_1 =  str(eg3_1)
s_eg3_2 =  str(eg3_2) 


s_eg4_1 = str(eg4_1) 
s_eg4_2  = str(eg4_2)

#Check to see whether eigenvalues have imaginary parts

if 'I' in s_eg1_1 or s_eg1_2 or s_eg2_1 or s_eg2_2 or s_eg3_1 or s_eg3_2 or s_eg4_1 or s_eg4_2 :
    c_s_eg1_1 =  s_eg1_1.replace('I','0')
    c_s_eg1_2 =  s_eg1_2.replace('I','0')
    
    c_s_eg2_1 =  s_eg2_1.replace('I','0')
    c_s_eg2_2 =  s_eg2_2.replace('I','0')
    
    c_s_eg3_1 =  s_eg3_1.replace('I','0')
    c_s_eg3_2 =  s_eg3_2.replace('I','0')
    
    c_s_eg4_1 =  s_eg4_1.replace('I','0')
    c_s_eg4_2 =  s_eg4_2.replace('I','0')
#Convert data types from strings back to Matrix
#Extract value from Matrix to convert data type

c_eg1_1 = Matrix([c_s_eg1_1])
c_c_eg1_1 = c_eg1_1[0]
c_eg1_2 = Matrix([c_s_eg1_2 ])
c_c_eg1_2 = c_eg1_2[0]
 
c_eg2_1 = Matrix([c_s_eg2_1])
c_c_eg2_1 = c_eg2_1[0]
c_eg2_2 = Matrix([c_s_eg2_2])
c_c_eg2_2 = c_eg2_2[0]

c_eg3_1 = Matrix([c_s_eg3_1])
c_c_eg3_1 = c_eg3_1[0]
c_eg3_2 = Matrix([c_s_eg3_2])
c_c_eg3_2 = c_eg3_2[0]

c_eg4_1 = Matrix([c_s_eg4_1])
c_c_eg4_1 = c_eg4_1[0]
c_eg4_2 = Matrix([c_s_eg4_2])
c_c_eg4_2 = c_eg4_2[0]

# Analyze stability of eigenvalues
##1.) third_fixedpoint
if c_c_eg1_1 and c_c_eg1_2 < 0 :
    print('fixed point' + str(first_fixedpoint) +'is stable')
if c_c_eg1_1 and c_c_eg1_2 > 0 :
    print('fixed point' + str(first_fixedpoint) +'is unstable')
if  c_c_eg1_1 < 0 and c_c_eg1_2> 0 or c_c_eg1_1 > 0 and c_c_eg1_2 < 0  :
   print('fixed point' + str(first_fixedpoint) +'is saddle')

#2.)forth_fixedpoint
if c_c_eg2_1 and c_c_eg2_2 < 0 :
    print('fixed point' + str(second_fixedpoint) +'is stable')
if c_c_eg2_1 and c_c_eg2_2 > 0 :
    print('fixed point' + str(second_fixedpoint) +'is unstable')
if  c_c_eg2_1 < 0 and c_c_eg2_2> 0 or c_c_eg2_1 > 0 and c_c_eg2_2 < 0  :
    print('fixed point' + str(second_fixedpoint) +'is saddle')

#3.) first_fixedpoint
if c_c_eg3_1 and c_c_eg3_2 < 0 :
    print('fixed point' + str(third_fixedpoint) +'is stable')
if c_c_eg3_1 and c_c_eg3_2 > 0 :
    print('fixed point' + str(third_fixedpoint) +'is unstable')
if  c_c_eg3_1 < 0 and c_c_eg3_2 > 0 or c_c_eg3_1 > 0 and c_c_eg3_2 < 0  :
    print('fixed point' + str(third_fixedpoint) +'is stable')

#4.) forth_fixedpoint 
if c_c_eg4_1 and c_c_eg4_2 < 0 :
    print('fixed point' + str(forth_fixedpoint) +'is stable')
if c_c_eg4_1 and c_c_eg4_2 > 0 :
    print('fixed point' + str(forth_fixedpoint) +'is unstable')
if  c_c_eg4_1 < 0 and c_c_eg4_2 > 0 or c_c_eg4_1 > 0 and c_c_eg4_2 < 0  :
    print('fixed point' + str(forth_fixedpoint) +'is sattle')
