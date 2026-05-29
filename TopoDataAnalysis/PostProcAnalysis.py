# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 01:34:54 2025

@author: udina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:27:40 2025

@author: u1450851
"""

#%%

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf

#%%#%%Paths to data

directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'

case = 'Sinusoidal'
# pathFig = directory + case + '\\Figures\\'

# pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_sin = directory + case + '/'

#%%#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

Nx = 256
Ny = 256
Nz = 384
Lx = 2.880
Ly = 2.880
Lz = 0.960
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz
mpiProc = 32
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz
z_short = z_uvp[:-5]

# Assuming you have a function build_phi to load phi data from a file
phi = build_phi(path_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf, iintf = build_intf(phi, dz)

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

z_profile = np.arange(Nz) * dz * zi
zeds = np.ones((Nx, Ny, 1)) * z_profile
    
dist = copy.deepcopy(zeds)

dist -= intf[:, :, np.newaxis] * zi

#%%#%%Import data

# import pickle

# # # var = ['u','v','w','p','uu','vv','ww','uuu','vvv','www','dudx','dvdy','dwdz','uv','uw','vw','txx','tyy','tzz','txy','txz','tyz',\
# # #        'dudz','dvdz','dudy','dvdx']

# var_TKE = ['u','v','w','p','uu','vv','ww','uv','uw','vw','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
#                        'txx','tyy','tzz','txy','txz','tyz','uuu','vvu','wwu','uuv','vvv','wwv','uuw','vvw','www',\
#                        'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz','vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
#                        'pu','pv','pw','dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']

# with open(path_sin+'data.pkl', 'rb') as f:
# # Load the data from the pickle file
#     data = pickle.load(f)
    
# data_sin = dict()
# for i in range(len(var)):
#     data_sin[var[i]] = np.mean(data[var[i]],axis=3)

# del data

#%%Save data to netcdf like Giulia's data structure
# NumVariables = 64
# dataTKE = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u','v','w','p','uu','vv','ww','uv','uw','vw','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
#                                               'txx','tyy','tzz','txy','txz','tyz','uuu','vvu','wwu','uuv','vvv','wwv','uuw','vvw','www',\
#                                               'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz','vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
#                                               'pu','pv','pw','dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']})
    
# for i in range(0,NumVariables):
#     dataTKE.data[:,:,:,i] = np.mean(data[var_TKE[i]],axis=(3))[:,:,5:]
#     print(f'Done with {var_TKE[i]}')

# del data

# dataTKE.to_netcdf(path_sin+'dataTKE.nc')

#%%Load netCDF data
NumVariables = 64
data = xr.open_dataarray(path_sin+'dataTKE.nc')

#%%Bicheng functions
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%#%%Compute Resolved REynodls stresses and TKE

# terms_ptb = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,8),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u2_t','uv_t',\
#                        'uw_t','v2_t','vw_t','w2_t','tke','tke_SGS']})

# wn_x = 2*np.pi*np.fft.rfftfreq(Nx, dx)
# wn_y = 2*np.pi*np.fft.rfftfreq(Ny, dy)

# # # Interpolate u and v to w node
# u_h = (data.data[:,:,:,0])
# v_h = (data.data[:,:,:,1])

# terms_ptb.data[:,:,:,0] = data.data[:,:,:,4] - data.data[:,:,:,0]**2
# terms_ptb.data[:,:,:,1] = data.data[:,:,:,7] - data.data[:,:,:,0]*data.data[:,:,:,1]
# terms_ptb.data[:,:,:,2] = (data.data[:,:,:,8] - u_h*data.data[:,:,:,2])
# # terms_ptb.data[:,:,:,2] = wnode2uvpnode(terms_ptb.data[:,:,:,2])

# terms_ptb.data[:,:,:,3] = data.data[:,:,:,5] - data.data[:,:,:,1]**2
# terms_ptb.data[:,:,:,4] = (data.data[:,:,:,9] - v_h*data.data[:,:,:,2])
# # terms_ptb.data[:,:,:,4] = wnode2uvpnode(terms_ptb.data[:,:,:,4])

# terms_ptb.data[:,:,:,5] = (data.data[:,:,:,6] - data.data[:,:,:,2]**2)
# # terms_ptb.data[:,:,:,5] = wnode2uvpnode(terms_ptb.data[:,:,:,5])

# terms_ptb.data[:,:,:,6] = (terms_ptb.data[:,:,:,0] + terms_ptb.data[:,:,:,3] + terms_ptb.data[:,:,:,5]) / 2
# terms_ptb.data[:,:,:,7] = (-data.data[:,:,:,19] - data.data[:,:,:,20] - data.data[:,:,:,21]) / 2

# terms_ptb.to_netcdf(path_sin+'terms_ptb.nc')

# # del terms_ptb

terms_ptb = xr.open_dataarray(path_sin+'terms_ptb.nc')

#%%

# terms_bdg = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,15),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['adv_h','adv_v',\
#                        'adv','uturb_h','uturb_v','ttrans','pturb_h','pturb_v','ptrans','dissip','canopy','totdis',\
#                            'prod_h','prod_v','prod']})

# # Calculate the advection term (all on uvp-nodes)
# print('Calculating the advection term')
# # ue = data.data[:,:,:,0]*terms_ptb['tke']
# duedx = get_dphidx(data.data[:,:,:,0]*terms_ptb.data[:,:,:,6], wn_x)
# # ve = data.data[:,:,:,1]*terms_ptb['tke']
# dvedy = get_dphidy(data.data[:,:,:,1]*terms_ptb.data[:,:,:,6], wn_y)
# tkez = (terms_ptb.data[:,:,:,6])
# # we = data.data[:,:,:,2]*tkez
# dwedz = get_dphidz(data.data[:,:,:,2]*tkez, dz)
# dwedz = wnode2uvpnode(dwedz)
# # terms_bdg.data[:,:,:,0] = -duedx - dvedy
# # terms_bdg.data[:,:,:,1] = -dwedz

# #SGS
# # ue_sgs = data.data[:,:,:,0]*terms_ptb.data[:,:,:,7]
# due_sgsdx = get_dphidx(data.data[:,:,:,0]*terms_ptb.data[:,:,:,7], wn_x)
# # ve_sgs = data.data[:,:,:,1]*terms_ptb.data[:,:,:,7]
# dve_sgsdy = get_dphidy(data.data[:,:,:,1]*terms_ptb.data[:,:,:,7], wn_y)
# tke_sgsz = (terms_ptb.data[:,:,:,7])
# # we_sgs = data.data[:,:,:,2]*tke_sgsz
# dwe_sgsdz = get_dphidz(data.data[:,:,:,2]*tke_sgsz, dz)
# dwe_sgsdz = wnode2uvpnode(dwe_sgsdz)

# terms_bdg.data[:,:,:,0] = - duedx - dvedy - due_sgsdx - dve_sgsdy
# terms_bdg.data[:,:,:,1] = - dwedz - dwe_sgsdz

# del duedx,dvedy,tkez,dwedz,due_sgsdx,dve_sgsdy,tke_sgsz,dwe_sgsdz

# terms_bdg.data[:,:,:,2] = terms_bdg.data[:,:,:,0] + terms_bdg.data[:,:,:,1]

# # Calculate the turbulent transport term (horizontal, all on uvp-nodes)
# print('Calculating the horizontal turbulent transport term')
# uw_c = (data.data[:,:,:,8])
# vw_c = (data.data[:,:,:,9])
# w_c = (data.data[:,:,:,2])
# w2_c = (data.data[:,:,:,6])

# u3_t = data.data[:,:,:,25] - 3*data.data[:,:,:,0]*data.data[:,:,:,4] + 2*(data.data[:,:,:,0]**3)
# uv2_t = data.data[:,:,:,26] - 2*data.data[:,:,:,1]*data.data[:,:,:,7]\
#   + 2*data.data[:,:,:,0]*(data.data[:,:,:,1]**2) - data.data[:,:,:,0]*data.data[:,:,:,5]
# uw2_t = (data.data[:,:,:,27]) - 2*w_c*uw_c + 2*data.data[:,:,:,0]*(w_c**2)\
#   - data.data[:,:,:,0]*w2_c
# ue = 0.5*(u3_t+uv2_t+uw2_t)
# del u3_t,uv2_t,uw2_t
# duedx = get_dphidx(ue, wn_x)
# del ue
# utxx_t = (-data.data[:,:,:,34]) - data.data[:,:,:,0]*(-data.data[:,:,:,19])
# vtxy_t = (-data.data[:,:,:,43]) - data.data[:,:,:,1]*(-data.data[:,:,:,22])
# wtxz_t = ((-data.data[:,:,:,44]) - data.data[:,:,:,2]*(-data.data[:,:,:,23]))
# # wtxz_t = wnode2uvpnode(wtxz_t)
# dutaudx = get_dphidx(0.5*(utxx_t+vtxy_t+wtxz_t), wn_x)
# del utxx_t,vtxy_t,wtxz_t
# # terms_bdg['uturb_h'] = -duedx-dutaudx

# u2v_t = data.data[:,:,:,28] - 2*data.data[:,:,:,0]*data.data[:,:,:,7]\
#   + 2*data.data[:,:,:,1]*(data.data[:,:,:,0]**2) - data.data[:,:,:,1]*data.data[:,:,:,4]
# v3_t = data.data[:,:,:,29] - 3*data.data[:,:,:,1]*data.data[:,:,:,5] + 2*(data.data[:,:,:,1]**3)
# vw2_t = (data.data[:,:,:,30]) - 2*w_c*vw_c + 2*data.data[:,:,:,1]*w_c**2\
#   - data.data[:,:,:,1]*w2_c
# ve = 0.5*(u2v_t+v3_t+vw2_t)
# del u2v_t,v3_t,vw2_t
# dvedy = get_dphidy(ve, wn_y)
# del ve
# utxy_t = (-data.data[:,:,:,45]) - data.data[:,:,:,0]*(-data.data[:,:,:,22])
# vtyy_t = (-data.data[:,:,:,38]) - data.data[:,:,:,1]*(-data.data[:,:,:,20])
# wtyz_t = ((-data.data[:,:,:,46]) - data.data[:,:,:,2]*(-data.data[:,:,:,24]))
# # wtyz_t = wnode2uvpnode(wtyz_t)
# dutaudy = get_dphidy(0.5*(utxy_t+vtyy_t+wtyz_t), wn_y)
# del utxy_t,vtyy_t,wtyz_t

# terms_bdg.data[:,:,:,3] = -duedx-dutaudx-dvedy-dutaudy

# del duedx,dutaudx,dvedy,dutaudy,uw_c,vw_c,w_c,w2_c

# # Calculate the turbulent transport term (vertical, all on uvp-nodes)
# print('Calculating the vertical turbulent transport term')
# u2_h = (data.data[:,:,:,4])
# v2_h = (data.data[:,:,:,5])
# wu2_t = data.data[:,:,:,31] - 2*u_h*data.data[:,:,:,8] + 2*data.data[:,:,:,2]*u_h**2\
#   - data.data[:,:,:,2]*u2_h
# wv2_t = data.data[:,:,:,32] - 2*v_h*data.data[:,:,:,9] + 2*data.data[:,:,:,2]*v_h**2\
#   - data.data[:,:,:,2]*v2_h
# w3_t = data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*data.data[:,:,:,2]**3
# dwedz = get_dphidz(0.5*(wu2_t+wv2_t+w3_t),dz)
# del wu2_t,wv2_t,w3_t
# # dwedz = get_dphidz(we, dz)
# dwedz = wnode2uvpnode(dwedz)
# utxz_t = (-data.data[:,:,:,47]) - u_h*(-data.data[:,:,:,23])
# vtyz_t = (-data.data[:,:,:,48]) - v_h*(-data.data[:,:,:,24])
# wtzz_t = (-data.data[:,:,:,42]) - data.data[:,:,:,2]*(-data.data[:,:,:,21])
# dutaudz = get_dphidz(0.5*(utxz_t+vtyz_t+wtzz_t), dz)
# del utxz_t,vtyz_t,wtzz_t
# dutaudz = wnode2uvpnode(dutaudz)
# terms_bdg.data[:,:,:,4] = -dwedz-dutaudz
# del dwedz,dutaudz

# terms_bdg.data[:,:,:,5] = terms_bdg.data[:,:,:,3] + terms_bdg.data[:,:,:,4]

# # Calculate the pressure transport term (all on uvp-nodes)
# print('Calculating the pressure transport term')
# # dpudx = get_dphidx(data_tavg['pu'], wn_x) - data_tavg['u']*get_dphidx(data_tavg['p'], wn_x)
# pu_t = data.data[:,:,:,49] - data.data[:,:,:,3]*(data.data[:,:,:,0])
# # pu_t = (data.data[:,:,:,49]-0.5*(uvpnode2wnode(data.data[:,:,:,25]+data.data[:,:,:,26])+data.data[:,:,:,27]))\
# #     - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*uvpnode2wnode(data.data[:,:,:,0])
# # terms_ptb['pu_t'] = (data_tavg['pu']\
# #     -(1/3)*(data_tavg['utxx']+data_tavg['utyy']+data_tavg['utzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['u']
# dpudx = (get_dphidx(pu_t, wn_x))
# del pu_t

# # dpvdy = get_dphidy(data_tavg['pv'], wn_y) - data_tavg['v']*get_dphidy(data_tavg['p'], wn_y)
# pv_t = data.data[:,:,:,50] - data.data[:,:,:,3]*(data.data[:,:,:,1])
# # pv_t = (data.data[:,:,:,50]-0.5*(uvpnode2wnode(data.data[:,:,:,28]+data.data[:,:,:,29])+data.data[:,:,:,30]))\
# #     - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*uvpnode2wnode(data.data[:,:,:,1])
# # terms_ptb['pv_t'] = (data_tavg['pv']\
# #     -(1/3)*(data_tavg['vtxx']+data_tavg['vtyy']+data_tavg['vtzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['v']
# dpvdy = (get_dphidy(pv_t, wn_y))
# del pv_t

# # p_h = data_tavg['p'] #uvpnode2wnode(data_tavg['p'])
# # dpwdz = get_dphidz(data_tavg['pw'], dz) - uvpnode2wnode(data_tavg['w'])*get_dphidz(data_tavg['p'], dz)
# pw_t = data.data[:,:,:,51] - data.data[:,:,:,3]*data.data[:,:,:,2]
# # pw_t = (data.data[:,:,:,51]-0.5*(data.data[:,:,:,31]+data.data[:,:,:,32]+data.data[:,:,:,33]))\
# #     - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*data.data[:,:,:,2]
# # terms_ptb['pw_t'] = (data_tavg['pw']\
# #     -(1/3)*(data_tavg['wtxx']+data_tavg['wtyy']+data_tavg['wtzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['w']
# dpwdz = get_dphidz(pw_t, dz)
# del pw_t
# dpwdz = wnode2uvpnode(dpwdz)

# terms_bdg.data[:,:,:,6] = -dpudx-dpvdy
# del dpudx,dpvdy
# terms_bdg.data[:,:,:,7] = -dpwdz
# del dpwdz

# terms_bdg.data[:,:,:,8] = terms_bdg.data[:,:,:,6] + terms_bdg.data[:,:,:,7]

# # Calculate the dissipation rate (all on uvp-nodes)
# print('Calculating the dissipation term')
# # terms_drv['S11'] = terms_drv['dudx']
# # terms_drv['S12'] = 0.5*(terms_drv['dudy'] + terms_drv['dvdx'])
# # terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
# # terms_drv['S22'] = terms_drv['dvdy']
# # terms_drv['S23'] = 0.5*(terms_drv['dvdz'] + terms_drv['dwdy'])
# # terms_drv['S33'] = terms_drv['dwdz']

# dissip = -data.data[:,:,:,52] - 2*data.data[:,:,:,55] - 2*(data.data[:,:,:,56]) - data.data[:,:,:,53] -\
#     2*(data.data[:,:,:,57]) - data.data[:,:,:,54]

# terms_bdg.data[:,:,:,9] = dissip\
#   - (-data.data[:,:,:,19])*data.data[:,:,:,10] - (-data.data[:,:,:,20])*data.data[:,:,:,14]\
#   - (-data.data[:,:,:,21])*(data.data[:,:,:,18])\
#   - 2*(-data.data[:,:,:,22])*0.5*(data.data[:,:,:,11]+data.data[:,:,:,13])\
#   - (2*(-data.data[:,:,:,23])*0.5*(data.data[:,:,:,12]+data.data[:,:,:,16]))\
#   - (2*(-data.data[:,:,:,24])*0.5*(data.data[:,:,:,15]+data.data[:,:,:,17]))
  
# del dissip

# terms_bdg.data[:,:,:,10] = (data.data[:,:,:,-1] - data.data[:,:,:,2]*data.data[:,:,:,-4])\
#                             + (data.data[:,:,:,-3]-data.data[:,:,:,0]*data.data[:,:,:,-6])\
#                                + (data.data[:,:,:,-2]-data.data[:,:,:,1]*data.data[:,:,:,-5])
# # terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
# # terms_bdg['canopy'] += ((data_tavg['ufdx']-data_tavg['u']*data_tavg['fdx'])\
# #   +(data_tavg['vfdy']-data_tavg['v']*data_tavg['fdy']))
  
# terms_bdg.data[:,:,:,11] = - terms_bdg.data[:,:,:,9] + terms_bdg.data[:,:,:,10]

# # # Calculate the production (all on uvp-nodes)
# print('Calculating the production term')
# terms_bdg.data[:,:,:,12] =\
#   -(terms_ptb.data[:,:,:,0]*data.data[:,:,:,10] + terms_ptb.data[:,:,:,1]*data.data[:,:,:,11]
#   + terms_ptb.data[:,:,:,1]*data.data[:,:,:,13] + terms_ptb.data[:,:,:,3]*data.data[:,:,:,14]
#   + terms_ptb.data[:,:,:,2]*(data.data[:,:,:,16]) + terms_ptb.data[:,:,:,4]*(data.data[:,:,:,17])
#   + (-data.data[:,:,:,19])*data.data[:,:,:,10] + (-data.data[:,:,:,20])*data.data[:,:,:,14]
#   + 2*(-data.data[:,:,:,22])*0.5*(data.data[:,:,:,11]+data.data[:,:,:,13]) + ((-data.data[:,:,:,23])*data.data[:,:,:,16])
#   + ((-data.data[:,:,:,24])*data.data[:,:,:,17])
#   )
# terms_bdg.data[:,:,:,13] =\
#   -(terms_ptb.data[:,:,:,2]*(data.data[:,:,:,12]) + terms_ptb.data[:,:,:,4]*(data.data[:,:,:,15])
#   + terms_ptb.data[:,:,:,5]*(data.data[:,:,:,18])
#   + ((-data.data[:,:,:,23])*data.data[:,:,:,12] + (-data.data[:,:,:,24])*data.data[:,:,:,15])
#   + (-data.data[:,:,:,21])*(data.data[:,:,:,18])
#   )
# # terms_bdg['prod_dudz'] = -(terms_ptb['uw_t']*terms_drv['dudz'] + data_tavg['txz']*terms_drv['dudz'])
# # terms_bdg['prod_dvdz'] = -(terms_ptb['vw_t']*terms_drv['dvdz'] + data_tavg['tyz']*terms_drv['dvdz'])
# # terms_bdg['prod_dwdz'] = -(terms_ptb['w2_t']*terms_drv['dwdz'] + data_tavg['tzz']*terms_drv['dwdz'])

# terms_bdg.data[:,:,:,14] = terms_bdg.data[:,:,:,12] + terms_bdg.data[:,:,:,13]

# # terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] - terms_bdg['canopy'] - terms_bdg['dissip']\
# #                     + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
# #                     + terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] 

# print('*'*80)

# terms_bdg.to_netcdf(path_sin + '\\TKE_terms.nc') 

# del u_h,v_h

terms_bdg = xr.open_dataarray(path_sin + 'TKE_terms.nc')

#%%Plot TKE

# terms_bdg = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,15),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['adv_h','adv_v',\
#                        'adv','uturb_h','uturb_v','ttrans','pturb_h','pturb_v','ptrans','dissip','canopy','totdis',\
#                            'prod_h','prod_v','prod']})

plt.figure(figsize=(4,8))
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,2]+terms_bdg.data[:,:,:,5]+terms_bdg.data[:,:,:,8]-terms_bdg.data[:,:,:,9]\
#                     -terms_bdg.data[:,:,:,10]+terms_bdg.data[:,:,:,14],axis=(0,1)),z_short,c='pink',marker='o',markevery=8,label='res')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,2],axis=(0,1)),z_short,c='red',marker='s',markevery=8,label='Adv')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,5],axis=(0,1)),z_short,c='purple',marker='^',markevery=8,label='T_trans')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,8],axis=(0,1)),z_short,c='brown',marker='D',markevery=8,label='P_trans')
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,9],axis=(0,1)),z_short,c='green',marker='v',markevery=8,label='Diss')
# plt.plot(np.nanmean(-terms_bdg.data[:,:,:,10],axis=(0,1)),z_short,c='orange',marker='o',markevery=8,label='Can')
plt.plot(np.nanmean(-terms_bdg.data[:,:,:,11],axis=(0,1)),z_short,c='black',marker='o',markevery=8,label='TotDis')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,14],axis=(0,1)),z_short,c='blue',marker='D',markevery=8,label='Prod')
# plt.plot(np.nanmean(terms_bdg['prod']-terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod-Diss')
# plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')

# plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
# plt.hlines(Hcanopy,-80,80,linestyle='--',colors='k')
# plt.hlines(2*Hcanopy,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,5,linestyle='--',colors='grey')
# plt.vlines(1,0,5,linestyle='--',colors='grey')
# plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(z_uvp[0],z_short[-1])
# plt.xlim(-30,0)
# plt.title(r'Real')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/z_i$')
plt.legend(loc='upper right')
plt.show()

#%%Compute the variance

uu = data.data[:,:,:,4] - data.data[:,:,:,0]*data.data[:,:,:,0] - data.data[:,:,:,19]
e = terms_ptb.data[:,:,:,6]

uu_e = uu/e

uu_e_cut = uu_e[:,:,100:]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_short,uu_e[:,100,:].T,cmap='bwr')

plt.show()

#%%Pcolor plots of TKE terms

yslice = 206

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,z_short,(terms_bdg.data[:,yslice,:,-1]-terms_bdg.data[:,yslice,:,10]+terms_bdg[:,yslice,:,9]).T,cmap='bwr',vmin=-30,vmax=30)
# p = axs.pcolormesh(x,z_short,terms_bdg.data[:,yslice,:,10].T,cmap='bwr',vmin=-50,vmax=50)
axs.plot(x,(intf-5*dz)[:,yslice],c='black')
axs.plot(x,(intf-5*dz)[:,yslice]+(39/zi),c='black',ls='--')
# axs.axhline(Hcanopy,ls='--',c='k')
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$z/z_i$',fontsize=15)
axs.set_ylim(z_short[0],z_short[-1])

cbar = plt.colorbar(p)

plt.show()

#%%Anisotropy function

def Anisotropy(Nx, Ny, Nz, R11, R22, R33, R12, R13, R23):
    """
    Compute barycentric invariants and smallest eigenvalue lambda3
    from Reynolds stress tensor components stored as xarray.DataArrays.
    """

    import numpy as np
    import xarray as xr

    # Extract dimensions and coordinates from one of the input arrays
    dims = R11.dims
    coords = R11.coords

    # Convert DataArrays to NumPy arrays
    R11v = R11.values
    R22v = R22.values
    R33v = R33.values
    R12v = R12.values
    R13v = R13.values
    R23v = R23.values

    # Calculate TKE
    e = R11v + R22v + R33v
    Ntot = Nx * Ny * Nz

    # Flatten arrays for batch processing
    e_flat = e.ravel(order='F')
    R11_flat = R11v.ravel(order='F')
    R22_flat = R22v.ravel(order='F')
    R33_flat = R33v.ravel(order='F')
    R12_flat = R12v.ravel(order='F')
    R13_flat = R13v.ravel(order='F')
    R23_flat = R23v.ravel(order='F')

    # Build full Reynolds stress tensor (Ntot, 3, 3)
    R_all = np.zeros((Ntot, 3, 3))
    R_all[:, 0, 0] = R11_flat
    R_all[:, 1, 1] = R22_flat
    R_all[:, 2, 2] = R33_flat
    R_all[:, 0, 1] = R12_flat
    R_all[:, 1, 0] = R12_flat
    R_all[:, 0, 2] = R13_flat
    R_all[:, 2, 0] = R13_flat
    R_all[:, 1, 2] = R23_flat
    R_all[:, 2, 1] = R23_flat

    # Avoid division by zero
    e_safe = np.where(e_flat == 0.0, 1e-12, e_flat)

    # Identity matrix for subtraction
    Id = np.eye(3)

    # Compute anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3_flat = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB_flat = C1c + 0.5 * C3c
    yB_flat = C3c * (np.sqrt(3) / 2)

    # Reshape back to (Nx, Ny, Nz)
    xB = xr.DataArray(xB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    yB = xr.DataArray(yB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    lambda3 = xr.DataArray(lambda3_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)

    return xB, yB, lambda3

#%%Compute anisotropy

# anisotropy = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,3),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['xB','yB','lambda3']})

# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(Nx, Ny, Nz-5, terms_ptb[:,:,:,0] - data[:,:,:,19],\
#                                                                          terms_ptb[:,:,:,3] - data[:,:,:,20],\
#                                                                              terms_ptb[:,:,:,5] - data[:,:,:,21],\
#                                                                                  terms_ptb[:,:,:,1] - data[:,:,:,22],\
#                                                                                      terms_ptb[:,:,:,2] - (data[:,:,:,23]),\
#                                                                                          terms_ptb[:,:,:,4] - (data[:,:,:,24]))

# anisotropy.to_netcdf(path_sin + 'anisotropy.nc')

anisotropy = xr.open_dataarray(path_sin + 'anisotropy.nc')

#%%Plot Anisotropy

os.chdir('C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\functions\\')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()

yslice = 156

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
levels=np.linspace(0,np.sqrt(3)/2,10)
plt1=axs.contourf(x,z_short,np.where(dist[:,yslice,5:]>0,anisotropy.data[:,yslice,:,1],np.nan).T,cmap=cmap,levels=levels,vmin=0,vmax=np.sqrt(3)/2)
axs.contour(x,z_short,np.where(dist[:,yslice,5:]>0,anisotropy.data[:,yslice,:,1],np.nan).T,levels=[0.38],colors='black')
# plt1=axs.contourf(x,z[0:Nz_SLayer],np.transpose(np.mean(yB,axis=1)),cmap=cmap,levels=8)
# axs.pcolormesh(x,z,np.transpose(mask[:,Ny_Slice,:]),cmap='gray')
# axs.plot(x,z_tpg/zi,color='black')
# axs.axhline(1,ls='--',c='k')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# axs.set_title(f'{cases[case]}',fontsize=15)
# plt.savefig(figPath+'yB.png',dpi=300,facecolor='white', edgecolor='white')


plt.show()

#%%Residual vs yB correlation plot

import numpy as np
import matplotlib.pyplot as plt
import copy

# Optional: remove if not needed
# from scipy.stats import gaussian_kde

level1 = 16*dz*zi
level2 = 80*dz*zi

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] - terms_bdg[:, :, :, 11])

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

# Select y_B and TKE slices
x_yB = anisotropy[:, :, :, 1]
y_TKE = tmpNorm
mask_dist = (dist[:,:,5:] > level1) & (dist[:,:,5:] < level2)

# Compute binned statistics
TKE_median = []
TKE_std = []
yB_mean = []

j = 0.1
for i in range(28):
    if i == 0:
        mask = x_yB < j
        yB_mean.append(0.05)
    else:
        mask = (x_yB > j) & (x_yB < j + 0.025)
        yB_mean.append(j + 0.0125)
        
    combined_mask = mask & mask_dist
    TKE_vals = y_TKE[combined_mask]
    TKE_median.append(np.nanmedian(TKE_vals))
    TKE_std.append(np.nanstd(TKE_vals))
    j += 0.025

# Plot results
axs.plot(yB_mean, TKE_median, c='k', marker='o')
axs.fill_between(
    yB_mean,
    np.array(TKE_median) - np.array(TKE_std),
    np.array(TKE_median) + np.array(TKE_std),
    alpha=0.5,
)

axs.axhline(0, color='k', linestyle='-.')
axs.axvline(0.38, color='k', linestyle='-.')
axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
axs.axvline(0.36, color='k', linestyle='-.')
axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)
# axs.set_title(f'{cases[case]}',fontsize=15)

# Correlation text
x_yB_flat = x_yB.values[mask_dist]
y_TKE_flat = y_TKE[mask_dist]

# Compute correlation, avoiding NaNs
valid = ~np.isnan(x_yB_flat) & ~np.isnan(y_TKE_flat)
corr = np.corrcoef(x_yB_flat[valid], y_TKE_flat[valid])[0, 1]
axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

plt.tight_layout()
plt.show()

del tmpDIS,tmpRES,tmpNorm,x_yB,y_TKE

#%%functions 

def compute_d_twr(data, coord, dist, height, dz, zi, u_scale, LAD):

    npoints = coord.shape[0]
    d_dim = np.zeros(npoints)

    for i in range(npoints):
        ix, iy = coord[i]
        z_nan = dist[ix, iy, :]

        # Find first valid vertical index
        valid_start = np.argmax(z_nan > 0) - 5
        
        # Extract u and v from data array
        u = data.data[ix, iy, valid_start:, 0]
        v = data.data[ix, iy, valid_start:, 1]

        U = np.sqrt(u**2 + v**2)
        
        Z = np.linspace(1, height - 5, 16) * dz * zi - 0.5 * dz * zi

        # Make sure we have enough points
        max_len = min(height - 5, len(U), len(LAD))
        VAR = U[:max_len] * u_scale
        Y = (VAR[:max_len]**2) * 0.4 * np.array(LAD[:max_len])
        Z_use = Z[:max_len]

        d_dim[i] = trapezoid(Y * Z_use, Z_use) / trapezoid(Y, Z_use)

    return d_dim

def find_coordinates(elevation_map, N_twrs, loc='max'):
    """
    Finds the coordinates of local peaks or valleys in a topography map.

    Parameters
    ----------
    elevation_map : 2D numpy array
        Topography height map.
    N_twrs : int
        Number of towers to select.
    loc : str, optional
        'max' for peaks, 'min' for valleys. Default is 'max'.

    Returns
    -------
    coords : ndarray of shape (N_twrs, 2)
        Selected (i, j) coordinates.
    """
    import numpy as np

    arr = np.array(elevation_map)
    rows, cols = arr.shape
    
    if loc == 'flat':
        # Uniform random sampling of N_twrs coordinates
        all_indices = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1).reshape(-1, 2)
        selected = all_indices[np.random.choice(all_indices.shape[0], N_twrs, replace=False)]
        return selected

    # Prepare slices for neighbors (4-connectivity)
    up    = arr[:-2, 1:-1]
    down  = arr[2:, 1:-1]
    left  = arr[1:-1, :-2]
    right = arr[1:-1, 2:]
    center = arr[1:-1, 1:-1]

    if loc == 'max':
        # Local peak condition (center > all neighbors)
        peak_mask = (center >= up) & (center >= down) & (center >= left) & (center >= right)
        # Threshold to keep only the top 10% elevation
        threshold = 0.9 * np.max(arr)
        high_mask = center > threshold
        valid_mask = peak_mask & high_mask
    elif loc == 'min':
        # Local valley condition (low elevation below a certain threshold)
        threshold = 2 * np.min(arr)
        valid_mask = center <= threshold        
    else:
        raise ValueError("loc must be 'max' or 'min'")

    # Get indices of valid points and shift to original indexing
    indices = np.argwhere(valid_mask) + 1  # shift due to 1:-1 slicing

    if len(indices) < N_twrs:
        raise ValueError(f"Not enough {'peaks' if loc == 'max' else 'valleys'} found to select {N_twrs} towers.")

    # Randomly sample N_twrs from candidates
    selected = indices[np.random.choice(len(indices), N_twrs, replace=False)]

    return selected

#%%Select coordinates

import random
# os.chdir('C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\Anisotropy\\')
# from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,Twr_TKE_Multi#,find_coordinates

N_twrs = 100

coord_p = find_coordinates(intf,N_twrs,'max')
coord_v = find_coordinates(intf,N_twrs,'min')

LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183] #ATTO nz=384

mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
# Broadcast to shape (Nx, Ny, Nz_SLayer, NumVariables)
mask4D = np.expand_dims(mask, axis=-1)
# Assign NaN in one operation
data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
    
d_dim_p = compute_d_twr(data,coord_p,dist,height,dz,zi,u_scale,LAD)
d_dim_v = compute_d_twr(data,coord_v,dist,height,dz,zi,u_scale,LAD)

#%%Plot topography and towers

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

fig, ax = plt.subplots(1,1,figsize=(5,6))
contour1 = ax.contourf(x * zi, y * zi, (intf*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
x_coords_p = x[coord_p[:, 0]] * zi
y_coords_p = y[coord_p[:, 1]] * zi
ax.plot(x_coords_p, y_coords_p, 'ok')

x_coords_v = x[coord_v[:, 0]] * zi
y_coords_v = y[coord_v[:, 1]] * zi
ax.plot(x_coords_v, y_coords_v, 'vr')

ax.tick_params(axis='x', which='major', labelsize=12)
ax.set_xlabel(f'x [m]', fontsize=15)
ax.set_xlim(0,x[-1]*zi)
ax.set_ylim(0,y[-1]*zi)

ax.tick_params(axis='y', which='both', labelbottom=False, labelleft=False)
    
ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_ylabel(f'y [m]', fontsize=15)

plt.show()

#%%Create a DataArray where I will store all the profiles and then save to netcdf
Nz_SLayer = 200
TwrAvgProf_v = xr.DataArray(np.zeros(shape = (Nz_SLayer,5),order='F'),\
                       dims=('z','variable'), coords = {'variable':['Umag','Shear','phi','skewness','kurtosis']})
    
TwrAvgProf_p = xr.DataArray(np.zeros(shape = (Nz_SLayer,5),order='F'),\
                       dims=('z','variable'), coords = {'variable':['Umag','Shear','phi','skewness','kurtosis']})

#%%Compute the velocity magnitude and plot it

N_twrs = coord_p.shape[0]
Umag_p = np.zeros((N_twrs, Nz_SLayer), dtype='float64', order='F')
Umag_v = np.zeros((N_twrs, Nz_SLayer), dtype='float64', order='F')
ustar = np.zeros(N_twrs)

# Pre-extract needed variable indices
idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

for k, (ix, iy) in enumerate(coord_p):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5

    # Extract Ruw and Rvw at z_start + 16
    z_idx_ustar = z_start + 16
    Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
           data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

    Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
           data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

    ustar[k] = (Ruw**2 + Rvw**2)**0.25

    # Slice velocity components over Nz_SLayer starting from z_start
    z_end = z_start + Nz_SLayer
    u = data.data[ix, iy, z_start:z_end, idx_u]
    v = data.data[ix, iy, z_start:z_end, idx_v]
    w = data.data[ix, iy, z_start:z_end, idx_w]

    # Compute magnitude and normalize
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    Umag_p[k, :] = vel_mag * (u_scale / ustar[k])
    
TwrAvgProf_p.data[:,0] = np.mean(Umag_p,axis=(0))
    
for k, (ix, iy) in enumerate(coord_v):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5

    # Extract Ruw and Rvw at z_start + 16
    z_idx_ustar = z_start + 16
    Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
           data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

    Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
           data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

    ustar[k] = (Ruw**2 + Rvw**2)**0.25

    # Slice velocity components over Nz_SLayer starting from z_start
    z_end = z_start + Nz_SLayer
    u = data.data[ix, iy, z_start:z_end, idx_u]
    v = data.data[ix, iy, z_start:z_end, idx_v]
    w = data.data[ix, iy, z_start:z_end, idx_w]

    # Compute magnitude and normalize
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    Umag_v[k, :] = vel_mag * (u_scale / ustar[k])

TwrAvgProf_v.data[:,0] = np.mean(Umag_v,axis=(0))

#%%Plot velocity magnitude

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_p.data[:,0],z_uvp[:Nz_SLayer]/(39/zi),'-k',label='Peaks')
axs.plot(TwrAvgProf_v.data[:,0],z_uvp[:Nz_SLayer]/(39/zi),'--k',label='Valley')

axs.set_xlabel(r"$\overline{U}/u_*$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend()
axs.set_ylim(0,12)
axs.axhline(1,ls='--',c='k')

plt.show()

#%%Compute the shear stress profiles

N_twrs = coord_p.shape[0]
shear_p = np.zeros((N_twrs, Nz_SLayer), dtype='float64', order='F')
shear_v = np.zeros((N_twrs, Nz_SLayer), dtype='float64', order='F')
ustar = np.zeros(N_twrs)

# Pre-extract needed variable indices
idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

for k, (ix, iy) in enumerate(coord_p):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5

    # Extract Ruw and Rvw at z_start + 16
    z_idx_ustar = z_start + 16
    Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
           data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

    Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
           data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

    ustar[k] = (Ruw**2 + Rvw**2)**0.25

    # Slice velocity components over Nz_SLayer starting from z_start
    z_end = z_start + Nz_SLayer
    uw = data.data[ix, iy, z_start:z_end, idx_uw] - data.data[ix, iy, z_start:z_end, idx_u]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_txz]
    vw = data.data[ix, iy, z_start:z_end, idx_vw] - data.data[ix, iy, z_start:z_end, idx_v]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_tyz]

    # Compute magnitude and normalize
    shear = np.sqrt(uw**2 + vw**2)
    shear_p[k, :] = shear * (u_scale**2 / ustar[k]**2)
    
TwrAvgProf_p[:,1] = np.mean(shear_p,axis=(0))

for k, (ix, iy) in enumerate(coord_v):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5

    # Extract Ruw and Rvw at z_start + 16
    z_idx_ustar = z_start + 16
    Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
           data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

    Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
           data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
           data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

    ustar[k] = (Ruw**2 + Rvw**2)**0.25

    # Slice velocity components over Nz_SLayer starting from z_start
    z_end = z_start + Nz_SLayer
    uw = data.data[ix, iy, z_start:z_end, idx_uw] - data.data[ix, iy, z_start:z_end, idx_u]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_txz]
    vw = data.data[ix, iy, z_start:z_end, idx_vw] - data.data[ix, iy, z_start:z_end, idx_v]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_tyz]

    # Compute magnitude and normalize
    shear = np.sqrt(uw**2 + vw**2)
    shear_v[k, :] = shear * (u_scale**2 / ustar[k]**2)

TwrAvgProf_v[:,1] = np.mean(shear_v,axis=(0))

#%%Plot the shear stress profile for the tower average

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_p.data[:,1],z_uvp[:Nz_SLayer]/(39/zi),'-k',label='Peaks')
axs.plot(TwrAvgProf_v.data[:,1],z_uvp[:Nz_SLayer]/(39/zi),'--k',label='Valley')

axs.set_xlabel(r"$\overline{u'w'}/u_*^2$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend()
axs.set_ylim(0,12)
axs.set_xlim(-0.1,2)
axs.axhline(1,ls='--',c='k')

plt.show()

#%%Functions needed to compute the velocity gradient

kappa = 0.4

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, dist, u, v, twr=False):    
    
    z_start = np.argmax(dist > 0) - 5

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[z_start:z_start+Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [20, 30, 70, 90]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    # Fitted velocity profile across the entire surface layer
    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

def phi_m_loc(Nx, Ny, Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

#%%Compute flux-gradient

phi_m_2D = np.zeros((N_twrs,Nz_SLayer))
z2D = np.zeros((N_twrs,Nz_SLayer))
z0hi = np.zeros((N_twrs),'d',order='F')
ustar = np.zeros((N_twrs),'d',order='F')
z_over_d = np.zeros((N_twrs,Nz_SLayer))

for k in range(N_twrs):
    loc = coord_p[k]
    z_start = np.argmax(dist[loc[0],loc[1],:] > 0) - 5

    z = z_uvp
    z_d = (z - ((d_dim_p[k])/zi))
    z2D[k,:] = z_d[0:Nz_SLayer]
    z_over_d[k,:] = ((z_uvp[0:Nz_SLayer]*zi)-d_dim_p[k])/39 
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)
    
    phi_m_2D[k,:] = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                              data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])
        
    print(f'Done with coord: {k}')

TwrAvgProf_p[:,2] = np.mean(phi_m_2D,axis=(0))

phi_m_2D = np.zeros((N_twrs,Nz_SLayer))
z2D = np.zeros((N_twrs,Nz_SLayer))
z0hi = np.zeros((N_twrs),'d',order='F')
ustar = np.zeros((N_twrs),'d',order='F')
z_over_d = np.zeros((N_twrs,Nz_SLayer))

for k in range(N_twrs):
    loc = coord_v[k]
    z_start = np.argmax(dist[loc[0],loc[1],:] > 0) - 5

    z = z_uvp
    z_d = (z - ((d_dim_v[k])/zi))
    z2D[k,:] = z_d[0:Nz_SLayer]
    z_over_d[k,:] = ((z_uvp[0:Nz_SLayer]*zi)-d_dim_v[k])/39 
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)
    
    phi_m_2D[k,:] = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                              data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])
        
    print(f'Done with coord: {k}')

TwrAvgProf_v[:,2] = np.mean(phi_m_2D,axis=(0))

#%%Plot flux-gradient

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_p.data[:,2],z_uvp[:Nz_SLayer]/(39/zi),'-k',label='Peaks')
axs.plot(TwrAvgProf_v.data[:,2],z_uvp[:Nz_SLayer]/(39/zi),'--k',label='Valley')

axs.set_xlabel(r"$\phi_M$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend()
axs.set_ylim(0,12)
axs.set_xlim(-0.4,2)
axs.axhline(1,ls='--',c='k')

plt.show()

#%%Skewness function

w3_t = data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*data.data[:,:,:,2]**3 + ((-data.data[:,:,:,42]) - data.data[:,:,:,2]*(-data.data[:,:,:,21]))
w2_t = data.data[:,:,:,6] - data.data[:,:,:,2]**2 - data.data[:,:,:,21]
skewness = w3_t/(w2_t)**1.5
skew_p = np.zeros((N_twrs,Nz_SLayer))
skew_v = np.zeros((N_twrs,Nz_SLayer))

for idx, (ix,iy) in enumerate(coord_p):
    z_start = np.argmax(dist[ix,iy,:] > 0) - 5
    skew_p[idx,:] = skewness[ix,iy,z_start:z_start+Nz_SLayer]
    
for idx, (ix,iy) in enumerate(coord_v):
    z_start = np.argmax(dist[ix,iy,:] > 0) - 5
    skew_v[idx,:] = skewness[ix,iy,z_start:z_start+Nz_SLayer]

del w3_t,w2_t,skewness

TwrAvgProf_p[:,3] = np.mean(skew_p,axis=(0))
TwrAvgProf_v[:,3] = np.mean(skew_v,axis=(0))

#%%Plot skewness


fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_p.data[:,3],z_uvp[:Nz_SLayer]/(39/zi),'-k',label='Peaks')
axs.plot(TwrAvgProf_v.data[:,3],z_uvp[:Nz_SLayer]/(39/zi),'--k',label='Valley')

axs.set_xlabel(r"$Sk$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend()
axs.set_ylim(0,12)
axs.set_xlim(-2,2)
axs.axhline(1,ls='--',c='k')

plt.show()

#%%Save profiles to netcdf

# TwrAvgProf_p.to_netcdf(path_sin+'Profile_p.nc')
# TwrAvgProf_v.to_netcdf(path_sin+'Profile_v.nc')









































