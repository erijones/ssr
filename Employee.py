#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:52:10 2018

@author: parkershankin-clarke
"""

class Employee :
    #pass if you want to leave the class empty
    
    #raise_amount = 1.04
    
    def __init__(self,first,last,pay):
        self.first = first
        self.last = last
        self.pay = pay
        
    def fullname(self):
        print('{} {}'.format(self.first,self.last))

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)
        
    
       


emp_1 = Employee('Parker','Shankin',5000) 
print(emp_1.pay)
print(emp_1.__dict__)
emp_1.raise_amount = 1.05


#print(emp_1.first)
#print('{} {}'.format(emp_1.first,emp_1.last))
#print(emp_1.fullname())

#emp_1.fullname()
#Employee.fullname(emp_1)


#Not the best way to do this

#    emp_1 = Employee()#unique instance of Employee class
#    emp_2 = Employee()
#    print(emp_1)#place that emp_1 occupies in memory
#    
#    emp_1.first = 'Rachel'
#    emp_1.last = 'Kazemi'
#   
#    emp_1.first = 'Parker'
#    emp_1.last = 'Shankin-Clarke'
#    
#    print(emp_1.first)
