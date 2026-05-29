#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 02:21:51 2023

@author: u1450851


"""
import math 
import numpy as np

# Paremeters & resolution
zi      = 520;           # [m]
z0     = 0.01;            # [m]
nz     = 260;           
lz      = 1*zi;              # [m]
h_canopy = 39;        # [m]   canopy height 

u_scale = 0.4; # [m/s]
ug_dim  = 5;        # [m/s]  geostrophic velocity at the top of the domain

# non-dim
lz = lz/zi;
dz = lz/nz;
h_canopy_n = math.ceil(h_canopy/(dz*zi));


#  compute  displacement height d
#  following Thom 1971 "Momentum absorption by vegetation"

# load LAD profile (it varies w height)
Cd = 0.4   # drag coeff
# LAD = 0.17 # m-1
LAD = np.array([0.027, 0.258, 0.174, 0.284, 0.306, 0.375, 0.334, 0.306, 0.286, 0.315, 0.235, 0.195, 0.091, 0.085, 0.041, 0.046, 0.012, 0.018, 0.002, 0])

# compute vector of heights up to canopy top
Z = np.linspace(1,h_canopy_n,20)*(dz*zi)
# Z = (1:h_canopy_n).*dz*zi;   Z = Z.T;            #    [m]    

# compute dimensional u (u_scale = u* usend in LES code to no dim variables)
u_dim = np.mean(data.data[:,:,0:h_canopy_n,6],axis=(0,1))*u_scale;     #    [m/s]   prof_u = mean in x and y  of u velocity

# # compute  variable before integration
Y = (u_dim**2)*Cd*LAD;                  #    [m2/s2]

# # get displacement height
d_dim = np.trapz(Y*Z,Z)/np.trapz(Y,Z);       #    [m] 
print('d_dim = ',d_dim);       #    [m]

# # non dim
# d = d_dim/zi;
# d_o_h = d/h_canopy;

