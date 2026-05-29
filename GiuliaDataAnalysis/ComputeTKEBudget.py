# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 07:50:16 2025

@author: udina
"""

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

#%%Set case and path to data


# Path to the data files:
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
os.chdir(path)  

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps','Empty_9mps']
case = -1

#%% Defining the main parameters of the simulations

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_w = np.arange(0,Nz)*dz
z_uvp = np.arange(0,Nz)*dz + dz/2

zi = 1000
uscale = 0.313
Hcanopy = 39/zi
kappa = 0.4  # von Karman constant
LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
height = math.ceil(Hcanopy/dz)

Ntwr = 100
Nz_Slayer = 126

# Loading the 3D Momentum, 2D Momentum, and 3D TKE Budget data:

#For the sake of clarity, below I sepcify the variables included in "data" and "dataS":
#--------------------------------------------------------------------------------------------------
#data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                        'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                        'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                        'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                        'avgdudz','avgdvdz','avgNut','avgCs']})



#data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
#                dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})


#data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u','v','w','p',\
#                        'uu','vv','ww','uv','uw','vw',\
#                        'dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
#                        'txx','tyy','tzz','txy','txz','tyz',\
#                        'uuu','uvv','uww','vuu','vvv','vww','wuu','wvv','www',\
#                        'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz',\
#                        'vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
#                        'up','vp','wp',\
#                        'dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']}) 

#%%Load the data
    
data = xr.open_dataarray(cases[case]+'/Data_Momentum_4TKE.nc')

#%% Bicheng Functions:
    
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

#%%Compute Reynolds stresses

# Rxx = data.data[:,:,:,4] - data.data[:,:,:,0]*data.data[:,:,:,0]
# Ryy = data.data[:,:,:,5] - data.data[:,:,:,1]*data.data[:,:,:,1]
# Rzz = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]*data.data[:,:,:,2])
# Rxy = data.data[:,:,:,7] - data.data[:,:,:,0]*data.data[:,:,:,1]
# Rxz = wnode2uvpnode(data.data[:,:,:,8]) - data.data[:,:,:,0]*wnode2uvpnode(data.data[:,:,:,2])
# Ryz = wnode2uvpnode(data.data[:,:,:,9]) - data.data[:,:,:,1]*wnode2uvpnode(data.data[:,:,:,2])

# #%%Reynolds stress profiles

# fig,axs = plt.subplots(1,6,tight_layout=True,figsize=(12,6))

# axs[0].plot(np.mean(Rxx,axis=(0,1)),z_uvp,'k',label='Res')
# axs[1].plot(np.mean(Ryy,axis=(0,1)),z_uvp,'k',label='Res')
# axs[2].plot(np.mean(Rzz,axis=(0,1)),z_uvp,'k',label='Res')
# axs[3].plot(np.mean(Rxy,axis=(0,1)),z_uvp,'k',label='Res')
# axs[4].plot(np.mean(Rxz,axis=(0,1)),z_uvp,'k',label='Res')
# axs[5].plot(np.mean(Ryz,axis=(0,1)),z_uvp,'k',label='Res')

# axs[0].plot(np.mean(-data.data[:,:,:,19],axis=(0,1)),z_uvp,'r',label='SGS')
# axs[1].plot(np.mean(-data.data[:,:,:,20],axis=(0,1)),z_uvp,'r',label='SGS')
# axs[2].plot(np.mean(-data.data[:,:,:,21],axis=(0,1)),z_uvp,'r',label='SGS')
# axs[3].plot(np.mean(-data.data[:,:,:,22],axis=(0,1)),z_uvp,'r',label='SGS')
# axs[4].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,23]),axis=(0,1)),z_uvp,'r',label='SGS')
# axs[5].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,24]),axis=(0,1)),z_uvp,'r',label='SGS')

# for i in range(len(axs)):
#     axs[i].set_ylim(0,0.2)
#     axs[i].axvline(0,c='k',ls='--')

# plt.show()

#%%Compute Resolved REynodls stresses and TKE

# del Rxx,Ryy,Rzz,Rxy,Rxz,Ryz

terms_ptb = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,8),order='F'),\
                       dims=('x','y','z','variable'), coords = {'variable':['u2_t','uv_t',\
                       'uw_t','v2_t','vw_t','w2_t','tke','tke_SGS']})

wn_x = 2*np.pi*np.fft.rfftfreq(Nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(Ny, dy)

# # Interpolate u and v to w node
u_h = uvpnode2wnode(data.data[:,:,:,0])
v_h = uvpnode2wnode(data.data[:,:,:,1])

terms_ptb.data[:,:,:,0] = data.data[:,:,:,4] - data.data[:,:,:,0]**2
terms_ptb.data[:,:,:,1] = data.data[:,:,:,7] - data.data[:,:,:,0]*data.data[:,:,:,1]
terms_ptb.data[:,:,:,2] = wnode2uvpnode(data.data[:,:,:,8] - u_h*data.data[:,:,:,2])
# terms_ptb.data[:,:,:,2] = wnode2uvpnode(terms_ptb.data[:,:,:,2])

terms_ptb.data[:,:,:,3] = data.data[:,:,:,5] - data.data[:,:,:,1]**2
terms_ptb.data[:,:,:,4] = wnode2uvpnode(data.data[:,:,:,9] - v_h*data.data[:,:,:,2])
# terms_ptb.data[:,:,:,4] = wnode2uvpnode(terms_ptb.data[:,:,:,4])

terms_ptb.data[:,:,:,5] = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]**2)
# terms_ptb.data[:,:,:,5] = wnode2uvpnode(terms_ptb.data[:,:,:,5])

terms_ptb.data[:,:,:,6] = (terms_ptb.data[:,:,:,0] + terms_ptb.data[:,:,:,3] + terms_ptb.data[:,:,:,5]) / 2
terms_ptb.data[:,:,:,7] = (-data.data[:,:,:,19] - data.data[:,:,:,20] - data.data[:,:,:,21]) / 2

terms_ptb.to_netcdf(path+cases[case]+'/terms_ptb.nc')

# del terms_ptb

# terms_ptb = xr.open_dataarray(path+cases[case]+'\\terms_ptb.nc')

#%%#%% Calculate the TKE budget

terms_bdg = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,15),order='F'),\
                       dims=('x','y','z','variable'), coords = {'variable':['adv_h','adv_v',\
                       'adv','uturb_h','uturb_v','ttrans','pturb_h','pturb_v','ptrans','dissip','canopy','totdis',\
                           'prod_h','prod_v','prod']})

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')

duedx = get_dphidx(data.data[:,:,:,0]*terms_ptb.data[:,:,:,6], wn_x)
dvedy = get_dphidy(data.data[:,:,:,1]*terms_ptb.data[:,:,:,6], wn_y)
tkez = uvpnode2wnode(terms_ptb.data[:,:,:,6])
dwedz = get_dphidz(data.data[:,:,:,2]*tkez, dz)

#SGS
due_sgsdx = get_dphidx(data.data[:,:,:,0]*terms_ptb.data[:,:,:,7], wn_x)
dve_sgsdy = get_dphidy(data.data[:,:,:,1]*terms_ptb.data[:,:,:,7], wn_y)
tke_sgsz = uvpnode2wnode(terms_ptb.data[:,:,:,7])
dwe_sgsdz = get_dphidz(data.data[:,:,:,2]*tke_sgsz, dz)

terms_bdg.data[:,:,:,0] = - duedx - dvedy - due_sgsdx - dve_sgsdy
terms_bdg.data[:,:,:,1] = - dwedz - dwe_sgsdz

del duedx,dvedy,tkez,dwedz,due_sgsdx,dve_sgsdy,tke_sgsz,dwe_sgsdz

terms_bdg.data[:,:,:,2] = terms_bdg.data[:,:,:,0] + terms_bdg.data[:,:,:,1]

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = wnode2uvpnode(data.data[:,:,:,8])
vw_c = wnode2uvpnode(data.data[:,:,:,9])
w_c = wnode2uvpnode(data.data[:,:,:,2])
w2_c = wnode2uvpnode(data.data[:,:,:,6])

u3_t = data.data[:,:,:,25] - 3*data.data[:,:,:,0]*data.data[:,:,:,4] + 2*(data.data[:,:,:,0]**3)
uv2_t = data.data[:,:,:,26] - 2*data.data[:,:,:,1]*data.data[:,:,:,7]\
  + 2*data.data[:,:,:,0]*(data.data[:,:,:,1]**2) - data.data[:,:,:,0]*data.data[:,:,:,5]
uw2_t = wnode2uvpnode(data.data[:,:,:,27]) - 2*w_c*uw_c + 2*data.data[:,:,:,0]*(w_c**2)\
  - data.data[:,:,:,0]*w2_c
ue = 0.5*(u3_t+uv2_t+uw2_t)
del u3_t,uv2_t,uw2_t
# duedx = get_dphidx(ue, wn_x)
# del ue
utxx_t = (-data.data[:,:,:,34]) - data.data[:,:,:,0]*(-data.data[:,:,:,19])
vtxy_t = (-data.data[:,:,:,43]) - data.data[:,:,:,1]*(-data.data[:,:,:,22])
wtxz_t = wnode2uvpnode((-data.data[:,:,:,44]) - data.data[:,:,:,2]*(-data.data[:,:,:,23]))
ue_sgs = 0.5*(utxx_t+vtxy_t+wtxz_t)
del utxx_t,vtxy_t,wtxz_t
# dutaudx = get_dphidx(ue_sgs, wn_x)

u2v_t = data.data[:,:,:,28] - 2*data.data[:,:,:,0]*data.data[:,:,:,7]\
  + 2*data.data[:,:,:,1]*(data.data[:,:,:,0]**2) - data.data[:,:,:,1]*data.data[:,:,:,4]
v3_t = data.data[:,:,:,29] - 3*data.data[:,:,:,1]*data.data[:,:,:,5] + 2*(data.data[:,:,:,1]**3)
vw2_t = wnode2uvpnode(data.data[:,:,:,30]) - 2*w_c*vw_c + 2*data.data[:,:,:,1]*(w_c**2)\
  - data.data[:,:,:,1]*w2_c
ve = 0.5*(u2v_t+v3_t+vw2_t)
del u2v_t,v3_t,vw2_t
# dvedy = get_dphidy(ve, wn_y)
# del ve
utxy_t = (-data.data[:,:,:,45]) - data.data[:,:,:,0]*(-data.data[:,:,:,22])
vtyy_t = (-data.data[:,:,:,38]) - data.data[:,:,:,1]*(-data.data[:,:,:,20])
wtyz_t = wnode2uvpnode((-data.data[:,:,:,46]) - data.data[:,:,:,2]*(-data.data[:,:,:,24]))
ve_sgs = 0.5*(utxy_t+vtyy_t+wtyz_t)
del utxy_t,vtyy_t,wtyz_t
# dutaudy = get_dphidy(ve_sgs, wn_y)

# terms_bdg.data[:,:,:,3] = -duedx-dutaudx-dvedy-dutaudy
terms_bdg.data[:,:,:,3] = - get_dphidx(ue+ue_sgs, wn_x) - get_dphidy(ve+ve_sgs, wn_y)

del uw_c,vw_c,w_c,w2_c,ue,ve,ue_sgs,ve_sgs

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = uvpnode2wnode(data.data[:,:,:,4])
v2_h = uvpnode2wnode(data.data[:,:,:,5])
wu2_t = data.data[:,:,:,31] - 2*u_h*data.data[:,:,:,8] + 2*data.data[:,:,:,2]*(u_h**2)\
  - data.data[:,:,:,2]*u2_h
wv2_t = data.data[:,:,:,32] - 2*v_h*data.data[:,:,:,9] + 2*data.data[:,:,:,2]*(v_h**2)\
  - data.data[:,:,:,2]*v2_h
w3_t = data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*(data.data[:,:,:,2]**3)
we = 0.5*(wu2_t+wv2_t+w3_t)
# dwedz = get_dphidz(we,dz)
del wu2_t,wv2_t,w3_t
utxz_t = (-data.data[:,:,:,47]) - u_h*(-data.data[:,:,:,23])
vtyz_t = (-data.data[:,:,:,48]) - v_h*(-data.data[:,:,:,24])
wtzz_t = (-data.data[:,:,:,42]) - data.data[:,:,:,2]*uvpnode2wnode(-data.data[:,:,:,21])
we_sgs = 0.5*(utxz_t+vtyz_t+wtzz_t)
# dutaudz = get_dphidz(we_sgs, dz)
del utxz_t,vtyz_t,wtzz_t
# terms_bdg.data[:,:,:,4] = -dwedz-dutaudz
terms_bdg.data[:,:,:,4] = - get_dphidz(we+we_sgs, dz)
del we,we_sgs
# del dwedz,dutaudz

terms_bdg.data[:,:,:,5] = terms_bdg.data[:,:,:,3] + terms_bdg.data[:,:,:,4]

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
# pu_t = data.data[:,:,:,49] - data.data[:,:,:,3]*uvpnode2wnode(data.data[:,:,:,0])
pu_t = (data.data[:,:,:,49]-0.5*(uvpnode2wnode(data.data[:,:,:,25]+data.data[:,:,:,26])+data.data[:,:,:,27]))\
    - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*uvpnode2wnode(data.data[:,:,:,0])

dpudx = wnode2uvpnode(get_dphidx(pu_t, wn_x))
del pu_t

# pv_t = data.data[:,:,:,50] - data.data[:,:,:,3]*uvpnode2wnode(data.data[:,:,:,1])
pv_t = (data.data[:,:,:,50]-0.5*(uvpnode2wnode(data.data[:,:,:,28]+data.data[:,:,:,29])+data.data[:,:,:,30]))\
    - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*uvpnode2wnode(data.data[:,:,:,1])

dpvdy = wnode2uvpnode(get_dphidy(pv_t, wn_y))
del pv_t

# pw_t = data.data[:,:,:,51] - data.data[:,:,:,3]*data.data[:,:,:,2]
pw_t = (data.data[:,:,:,51]-0.5*(data.data[:,:,:,31]+data.data[:,:,:,32]+data.data[:,:,:,33]))\
    - (data.data[:,:,:,3]-0.5*(uvpnode2wnode(data.data[:,:,:,4]+data.data[:,:,:,5])+data.data[:,:,:,6]))*data.data[:,:,:,2]
    
dpwdz = get_dphidz(pw_t, dz)
del pw_t

terms_bdg.data[:,:,:,6] = -dpudx-dpvdy
del dpudx,dpvdy
terms_bdg.data[:,:,:,7] = -dpwdz
del dpwdz

terms_bdg.data[:,:,:,8] = terms_bdg.data[:,:,:,6] + terms_bdg.data[:,:,:,7]

# Calculate the dissipation rate (all on uvp-nodes)
print('Calculating the dissipation term')

dissip = - (data.data[:,:,:,52] + data.data[:,:,:,53] + data.data[:,:,:,54] + data.data[:,:,:,55] +\
    wnode2uvpnode(data.data[:,:,:,56] + data.data[:,:,:,57]))

terms_bdg.data[:,:,:,9] = (dissip\
  - (-data.data[:,:,:,19]*data.data[:,:,:,10] - data.data[:,:,:,20]*data.data[:,:,:,14]\
  - data.data[:,:,:,21]*data.data[:,:,:,18]\
  - data.data[:,:,:,22]*(data.data[:,:,:,11]+data.data[:,:,:,13])\
  - wnode2uvpnode(data.data[:,:,:,23]*(data.data[:,:,:,12]+data.data[:,:,:,16]))\
  - wnode2uvpnode(data.data[:,:,:,24]*(data.data[:,:,:,15]+data.data[:,:,:,17]))))
  
del dissip

terms_bdg.data[:,:,:,10] = - (wnode2uvpnode(data.data[:,:,:,-1] - data.data[:,:,:,2]*(data.data[:,:,:,-4]))\
                            + (data.data[:,:,:,-3]-data.data[:,:,:,0]*(data.data[:,:,:,-6]))\
                               + (data.data[:,:,:,-2]-data.data[:,:,:,1]*(data.data[:,:,:,-5])))
  
terms_bdg.data[:,:,:,11] = terms_bdg.data[:,:,:,9] + terms_bdg.data[:,:,:,10]

# # Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg.data[:,:,:,12] =\
  -(terms_ptb.data[:,:,:,0]*data.data[:,:,:,10] + terms_ptb.data[:,:,:,1]*data.data[:,:,:,11]
  + terms_ptb.data[:,:,:,1]*data.data[:,:,:,13] + terms_ptb.data[:,:,:,3]*data.data[:,:,:,14]
  + terms_ptb.data[:,:,:,2]*wnode2uvpnode(data.data[:,:,:,16]) + terms_ptb.data[:,:,:,4]*wnode2uvpnode(data.data[:,:,:,17])
  + (-data.data[:,:,:,19])*data.data[:,:,:,10] + (-data.data[:,:,:,20])*data.data[:,:,:,14]
  + 2*(-data.data[:,:,:,22])*0.5*(data.data[:,:,:,11]+data.data[:,:,:,13]) + wnode2uvpnode((-data.data[:,:,:,23])*data.data[:,:,:,16])
  + wnode2uvpnode((-data.data[:,:,:,24])*data.data[:,:,:,17])
  )
terms_bdg.data[:,:,:,13] =\
  -(terms_ptb.data[:,:,:,2]*wnode2uvpnode(data.data[:,:,:,12]) + terms_ptb.data[:,:,:,4]*wnode2uvpnode(data.data[:,:,:,15])
  + terms_ptb.data[:,:,:,5]*data.data[:,:,:,18]
  + wnode2uvpnode((-data.data[:,:,:,23])*data.data[:,:,:,12] + (-data.data[:,:,:,24])*data.data[:,:,:,15])
  + (-data.data[:,:,:,21])*data.data[:,:,:,18]
  )

terms_bdg.data[:,:,:,14] = terms_bdg.data[:,:,:,12] + terms_bdg.data[:,:,:,13]

print('*'*80)

terms_bdg.to_netcdf(path+cases[case] + '/TKE_terms.nc') 

del u_h,v_h

# terms_bdg = xr.open_dataarray(path+cases[case] + '\\TKE_terms.nc')


#%%Compute ustar

# T_13 = wnode2uvpnode((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23]))
# T_23 = wnode2uvpnode((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24]))

# # Remove planar mean at each height level
# u_pp = data.data[:,:,:,0] - data.data[:,:,:,0].mean(axis=(0, 1), keepdims=True)
# v_pp = data.data[:,:,:,1] - data.data[:,:,:,1].mean(axis=(0, 1), keepdims=True)
# w_pp = data.data[:,:,:,2] - data.data[:,:,:,2].mean(axis=(0, 1), keepdims=True)

# # Compute dispersive stresses
# D13 = (u_pp) * wnode2uvpnode(w_pp)
# D23 = (v_pp) * wnode2uvpnode(w_pp)

# cov_turb = -np.sqrt((T_13 + D13)**2 + (T_23 + D23)**2)
# ustar_w = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

# del T_13,T_23,u_pp,v_pp,w_pp,D13,D23,cov_turb

# #%%1D functions

# def uvpnode2wnode_1d(phi_c):
#     phi_h = np.zeros_like(phi_c)
#     phi_h[0] = 0  # lower boundary (e.g., ground)
#     phi_h[1:] = 0.5 * (phi_c[:-1] + phi_c[1:])
#     return phi_h

# def wnode2uvpnode_1d(phi_h):
#     phi_c = np.zeros_like(phi_h)
#     phi_c[:-1] = 0.5 * (phi_h[:-1] + phi_h[1:])
#     phi_c[-1] = phi_h[-1]  # extrapolate or apply BC
#     return phi_c

# #%%Plot TKE profiles

# prod = (terms_bdg[:,:,:,14])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# diss = (terms_bdg[:,:,:,9])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# ptrans = (terms_bdg[:,:,:,8])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# ttrans = (terms_bdg[:,:,:,5])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# canopy = (terms_bdg[:,:,:,10])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# totdis = (terms_bdg[:,:,:,11])*Hcanopy/((ustar_w[:,:,np.newaxis])**3)
# # res = np.mean(uvpnode2wnode(terms_bdg[:,:,:,2]+terms_bdg[:,:,:,5]+terms_bdg[:,:,:,8]-terms_bdg[:,:,:,11]+terms_bdg[:,:,:,-1]),\
# #               axis=(0,1))*Hcanopy/(np.mean(ustar_w)**3)

# fig,axs = plt.subplots(1,1,figsize=(6,8),tight_layout=True)

# axs.plot(uvpnode2wnode_1d(np.mean(np.mean(prod,axis=(0)),axis=(0))),z_w/(39/zi),c='b',label='prod')
# # axs.plot(uvpnode2wnode_1d(np.mean(diss,axis=(0,1))),z_w/(39/zi),c='g',label='diss')
# # axs.plot(uvpnode2wnode_1d(np.mean(np.mean(canopy,axis=(0)),axis=(0))),z_w/(39/zi),c='brown',label='canopy')
# axs.plot(uvpnode2wnode_1d(np.mean(ptrans,axis=(0,1))),z_w/(39/zi),c='r',label='Ptrans')
# axs.plot(uvpnode2wnode_1d(np.mean(ttrans,axis=(0,1))),z_w/(39/zi),c='k',label='Ttrans')
# axs.plot(uvpnode2wnode_1d(np.mean(totdis,axis=(0,1))),z_w/(39/zi),c='g',label='TotDis')
# # axs.plot(res,z_w/(39/zi),c='orange')


# axs.set_ylim(0,5)
# axs.set_xlim(-8,8)
# axs.axhline(1,c='k',ls='--')
# axs.grid()
# axs.legend()
# plt.show()
































