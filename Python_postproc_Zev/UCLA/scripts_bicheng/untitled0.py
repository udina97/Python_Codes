#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:41:59 2023

@author: u0851921
"""
def add(x,y):
    tmp = x[:]
    tmp = tmp*2
    return tmp+y
x = [1,2,3]
y = [1,2,3]
z = add(x,y)

print(x)
print(z)
