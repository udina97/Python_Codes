#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:09:44 2023

@author: u1450851
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import hvplot
plt.rcParams['figure.dpi'] = 300
import os

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/')
from Analysis import Fddx, Fddy, ddz3_uv, ddz3_w
from Stats import ProjectOn_uvp, ProjectOn_w, mean_xy

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute the terms of the 3D TKE budget
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# First upload the files and save the variables 

Nx = 64
Ny = 64
Nz = 64

Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z = np.arange(0,Nz)*dz


path = '/scratch/general/nfs1/u1450851/tke_HomoT300/'
os.chdir(path)

#netcdf files containing the 3D RAV variables
data = xr.open_dataarray('Data_Momentum_60min.nc')
dataS = xr.open_dataarray('Data_Scalar_60min.nc')
dataTKE = xr.open_dataarray('Data_TKEbudget3D_60min.nc')

#%%
# UU = data.data[:,:,:,4]
# U = data.data[:,:,:,0]

# U_xyavg = np.mean(U,axis=(0,1))
# UU_xyavg = np.mean(UU,axis=(0,1))
# # 
# # UU_d = np.zeros((Nx,Ny,Nz), 'd', order='F')

# uu_var = UU - U*U

# uu_var_xyavg = np.mean(uu_var, axis=(0,1))

# UU_d = np.mean(U*U,axis=(0,1)) - (U_xyavg*U_xyavg)  

# # for k in range(0,Nz):
#     # UU_d[:,:,k] = 
    
# figure = plt.figure()
# plt.plot(UU_d,z,color='red',label='UU_d')
# plt.plot(uu_var_xyavg,z, color='green', label='uu_var')
# plt.plot(UU_d+uu_var_xyavg,z,color='black',label='sum')
# plt.legend()
    


# plt.pcolormesh(x,y,np.transpose(U[:,:,10]), shading='gouraud',cmap='coolwarm')

#%%
#3D wind field
U = data.data[:,:,:,0]; V = data.data[:,:,:,1]; W = data.data[:,:,:,2]

#Derivatives of the 3 wind components in 3 directions of space
dUdx = dataTKE.data[:,:,:,0]; dUdy = dataTKE.data[:,:,:,1]; dUdz = dataTKE.data[:,:,:,2]
dVdx = dataTKE.data[:,:,:,3]; dVdy = dataTKE.data[:,:,:,4]; dVdz = dataTKE.data[:,:,:,5]
dWdx = dataTKE.data[:,:,:,6]; dWdy = dataTKE.data[:,:,:,7]; dWdz = dataTKE.data[:,:,:,8]

#dudz and dvdz from Fabien outputRAV
dudz  = data.data[:,:,:,22]; dvdz = data.data[:,:,:,23]

#Components of the SGS stress tensor
txx =  - data.data[:,:,:,16]; tyy = - data.data[:,:,:,17]; tzz = - data.data[:,:,:,18]
txy =  - data.data[:,:,:,19]; txz = - data.data[:,:,:,20]; tyz = - data.data[:,:,:,21]

#Temperature
T = dataS.data[:,:,:,0]

#Coupled wind components products
UU = data.data[:,:,:,4]; VV = data.data[:,:,:,5]; WW = data.data[:,:,:,6]
UV = data.data[:,:,:,13]; UW = data.data[:,:,:,14]; VW = data.data[:,:,:,15]

#Temperature "covariance" resolved and SGS
WT = dataS.data[:,:,:,4]; WTsgs = - dataS.data[:,:,:,7]

#Triple "correlations" for wind speeds
UUU = data.data[:,:,:,7]; VVV = data.data[:,:,:,8]; WWW = data.data[:,:,:,9]
UVV = dataTKE.data[:,:,:,9]; UWW = dataTKE.data[:,:,:,10]; VUU = dataTKE.data[:,:,:,11]
VWW = dataTKE.data[:,:,:,12]; WUU = dataTKE.data[:,:,:,13]; WVV = dataTKE.data[:,:,:,14]

#Pressure terms
PU1 = dataTKE.data[:,:,:,24]; PV1 = dataTKE.data[:,:,:,25]; PW1 = dataTKE.data[:,:,:,26]
PU2 = dataTKE.data[:,:,:,40]; PV2 = dataTKE.data[:,:,:,41]; PW2 = dataTKE.data[:,:,:,42]
PU3 = dataTKE.data[:,:,:,43]; PV3 = dataTKE.data[:,:,:,44]; PW3 = dataTKE.data[:,:,:,45]
P1 = dataTKE.data[:,:,:,27]; P2 = dataTKE.data[:,:,:,46]; P = data.data[:,:,:,3]

#
Utxx = - dataTKE.data[:,:,:,15]; Utyy = - dataTKE.data[:,:,:,16]; Utzz = - dataTKE.data[:,:,:,17]
Vtxx = - dataTKE.data[:,:,:,18]; Vtyy = - dataTKE.data[:,:,:,19]; Vtzz = - dataTKE.data[:,:,:,20]
Wtxx = - dataTKE.data[:,:,:,21]; Wtyy = - dataTKE.data[:,:,:,22]; Wtzz = - dataTKE.data[:,:,:,23]
Utxy = - dataTKE.data[:,:,:,34]; Utxz = - dataTKE.data[:,:,:,35]
Vtxy = - dataTKE.data[:,:,:,36]; Vtyz = - dataTKE.data[:,:,:,37]
Wtxz = - dataTKE.data[:,:,:,38]; Wtyz = - dataTKE.data[:,:,:,39]

#Dissipation terms
dxx = - dataTKE.data[:,:,:,28]; dyy = - dataTKE.data[:,:,:,29]; dzz = - dataTKE.data[:,:,:,30]
dxy = - dataTKE.data[:,:,:,31]; dxz = - dataTKE.data[:,:,:,32]; dyz = - dataTKE.data[:,:,:,33]

#%% Compute the variances and TKE

uu_var = UU - U*U; vv_var = VV - V*V; ww_var = WW - W*W
uv_var = UV - U*V; uw_var = UW - ProjectOn_w(Nx, Ny, Nz, U)*W; vw_var = VW - ProjectOn_w(Nx, Ny, Nz, V)*W
pu1_var = PU1 - P1*U; pv1_var = PV1 - P1*V; pw1_var = PW1 - ProjectOn_w(Nx, Ny, Nz, P1)*W
pu2_var = PU2 - P2*U; pv2_var = PV2 - P2*V; pw2_var = PW2 - ProjectOn_w(Nx, Ny, Nz, P2)*W
pu3_var = PU3 - P*U; pv3_var = PV3 - P*V; pw3_var = PW3 - ProjectOn_w(Nx, Ny, Nz, P)*W

e = 0.5*(ProjectOn_w(Nx, Ny, Nz, uu_var) + ProjectOn_w(Nx, Ny, Nz, vv_var) + ww_var) #tke
e_sgs = ProjectOn_w(Nx, Ny, Nz, 0.5*(txx + tyy + tzz))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Dispersive Terms from Finnigan 2000 eq. 7.1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_xyavg = mean_xy(U); v_xyavg = mean_xy(V); w_xyavg = mean_xy(W)
uu_xyavg = mean_xy(uu_var); vv_xyavg = mean_xy(vv_var); ww_xyavg = mean_xy(ww_var)
uv_xyavg = mean_xy(uv_var); uw_xyavg = mean_xy(uw_var); vw_xyavg = mean_xy(vw_var)
dudx_xyavg = mean_xy(dUdx); dudy_xyavg = mean_xy(dUdy); dudz_xyavg = mean_xy(dUdz)
dvdx_xyavg = mean_xy(dVdx); dvdy_xyavg = mean_xy(dVdy); dvdz_xyavg = mean_xy(dVdz)
dwdx_xyavg = mean_xy(dWdx); dwdy_xyavg = mean_xy(dWdy); dwdz_xyavg = mean_xy(dWdz)
txx_xyavg = mean_xy(txx); tyy_xyavg = mean_xy(tyy); tzz_xyavg = mean_xy(tzz)
txy_xyavg = mean_xy(txx); txz_xyavg = mean_xy(tyy); tyz_xyavg = mean_xy(tzz)

# uutxx_xyavg = mean_xy(uu_var+txx); uvtxy_xyavg = mean_xy(uv_var+txy); uwtxz_xyavg = mean_xy(uw_var+txz)
# vvtyy_xyavg = mean_xy(vv_var+tyy); vwtyz_xyavg = mean_xy(vw_var+tyz); wwtzz_xyavg = mean_xy(ww_var+ProjectOn_w(Nx, Ny, Nz, tzz)) 

# tzz_w = ProjectOn_w(Nx, Ny, Nz, tzz)

ud = np.zeros_like(U); vd = np.zeros_like(V); wd = np.zeros_like(W)
uud = np.zeros_like(UU); vvd = np.zeros_like(VV); wwd = np.zeros_like(WW)
uvd = np.zeros_like(UV); uwd = np.zeros_like(UW); vwd = np.zeros_like(VW)
duddx = np.zeros_like(dUdx); duddy = np.zeros_like(dUdy); duddz = np.zeros_like(dUdz)
dvddx = np.zeros_like(dVdx); dvddy = np.zeros_like(dVdy); dvddz = np.zeros_like(dVdz)
dwddx = np.zeros_like(dWdx); dwddy = np.zeros_like(dWdy); dwddz = np.zeros_like(dWdz)
txxd = np.zeros_like(txx); tyyd = np.zeros_like(tyy); tzzd = np.zeros_like(tzz)
txyd = np.zeros_like(txy); txzd = np.zeros_like(txz); tyzd = np.zeros_like(tyz)

# uutxxd = np.zeros_like(txx); uvtxyd = np.zeros_like(txy); uwtxzd = np.zeros_like(txz)
# vvtyyd = np.zeros_like(tyy); vwtyzd = np.zeros_like(tyz); wwtzzd = np.zeros_like(tzz)

for k in range(0,Nz):
    ud[:,:,k] = U[:,:,k] - u_xyavg[k]
    vd[:,:,k] = V[:,:,k] - v_xyavg[k]
    wd[:,:,k] = W[:,:,k] - w_xyavg[k]
    uud[:,:,k] = uu_var[:,:,k] - uu_xyavg[k]
    vvd[:,:,k] = vv_var[:,:,k] - vv_xyavg[k]
    wwd[:,:,k] = ww_var[:,:,k] - ww_xyavg[k]
    uvd[:,:,k] = uv_var[:,:,k] - uv_xyavg[k]
    uwd[:,:,k] = uw_var[:,:,k] - uw_xyavg[k]
    vwd[:,:,k] = vw_var[:,:,k] - vw_xyavg[k]
    duddx[:,:,k] = dUdx[:,:,k] - dudx_xyavg[k]
    duddy[:,:,k] = dUdy[:,:,k] - dudy_xyavg[k]
    duddz[:,:,k] = dUdz[:,:,k] - dudz_xyavg[k]
    dvddx[:,:,k] = dVdx[:,:,k] - dvdx_xyavg[k]
    dvddy[:,:,k] = dVdy[:,:,k] - dvdy_xyavg[k]
    dvddz[:,:,k] = dVdz[:,:,k] - dvdz_xyavg[k]
    dwddx[:,:,k] = dWdx[:,:,k] - dwdx_xyavg[k]
    dwddy[:,:,k] = dWdy[:,:,k] - dwdy_xyavg[k]
    dwddz[:,:,k] = dWdz[:,:,k] - dwdz_xyavg[k]
    txxd[:,:,k] = txx[:,:,k] - txx_xyavg[k]
    tyyd[:,:,k] = tyy[:,:,k] - tyy_xyavg[k]
    tzzd[:,:,k] = tzz[:,:,k] - tzz_xyavg[k]
    txyd[:,:,k] = txy[:,:,k] - txy_xyavg[k]
    txzd[:,:,k] = txz[:,:,k] - txz_xyavg[k]
    tyzd[:,:,k] = tyz[:,:,k] - tyz_xyavg[k]
    # uutxxd[:,:,k] = (uu_var[:,:,k]+txx[:,:,k]) - uutxx_xyavg[k]
    # uvtxyd[:,:,k] = (uv_var[:,:,k]+txy[:,:,k]) - uvtxy_xyavg[k]
    # uwtxzd[:,:,k] = (uw_var[:,:,k]+txz[:,:,k]) - uwtxz_xyavg[k]
    # vvtyyd[:,:,k] = (vv_var[:,:,k]+tyy[:,:,k]) - vvtyy_xyavg[k]
    # vwtyzd[:,:,k] = (vw_var[:,:,k]+tyz[:,:,k]) - vwtyz_xyavg[k]
    # wwtzzd[:,:,k] = (ww_var[:,:,k]+tzz_w[:,:,k]) - wwtzz_xyavg[k]
    
#%%
def ProjectOn_uvp_1D(Nz,var1D):
    "Projects a 1D vertical profile from w node to uvp"

    import numpy as np

    var1D_uvp = np.zeros((Nz),'d',order='F')

    #Projecting the 3D variable onto the uvp nodes.
    for k in range(0,Nz-1):
        var1D_uvp[k]=0.5*(var1D[k] + var1D[k+1])

    var1D_uvp[-1] = var1D[-1]

    return(var1D_uvp)

def ProjectOn_w_1D(Nz,var1D):
    "Projects a 1D vertical profile from uvp node to w"

    import numpy as np

    var1D_w = np.zeros((Nz),'d',order='F')

    #Projecting the 3D variable onto the w nodes.
    for k in range(1,Nz):
        var1D_w[k]=0.5*(var1D[k] + var1D[k-1])

    return(var1D_w)

def ddz_1D_uvp(Nz,dz,var1D):
    "Computes the vertical derivative of a variable on uvp nodes placing it on w nodes"
    
    ddz_var1D = np.zeros((Nz), 'd', order='F')
    ddz_var1D[1:] = (var1D[1:] - var1D[0:-1])/dz
    ddz_var1D[0] = 0
    # ddz_var1D[0] = var1D[0]/(dz/2)
    
    return(ddz_var1D)
    
def ddz_1D_w(Nz,dz,var1D):
    "Computes the vertical derivative of a variable on w nodes placing it on uvp nodes"
    
    ddz_var1D = np.zeros((Nz), 'd', order='F')
    ddz_var1D[0:-1] = (var1D[1:] - var1D[0:-1])/dz
    ddz_var1D[-1] = var1D[-1]
    
    return(ddz_var1D)

#%%Compute advection terms on the RHS (minus in front)
#Resolved
dedx = Fddx(Nx, Ny, Nz, e, 1, 0, Lx, Nx)
dedy = Fddy(Nx, Ny, Nz, e, 1, 0, Ly, Ny)
dedz = ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, e))

Adv_res = - (ProjectOn_w(Nx, Ny, Nz, U)*dedx + ProjectOn_w(Nx, Ny, Nz, V)*dedy + W*dedz)

#SGS
dedx_sgs = Fddx(Nx, Ny, Nz, e_sgs, 1, 0, Lx, Nx)
dedy_sgs = Fddy(Nx, Ny, Nz, e_sgs, 1, 0, Ly, Ny)
dedz_sgs = ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, e_sgs))

Adv_sgs = -(ProjectOn_w(Nx, Ny, Nz, U)*dedx_sgs + ProjectOn_w(Nx, Ny, Nz, V)*dedy_sgs + W*dedz_sgs)

#tot
Adv_tot = Adv_res + Adv_sgs

#Advection Finnigan
Adv_F = - mean_xy(W)*ProjectOn_w_1D(Nz, (ddz_1D_w(Nz, dz, mean_xy(e))))

#Advection Finnigan SGS
Adv_sgs_F = - mean_xy(W)*ProjectOn_w_1D(Nz, (ddz_1D_w(Nz, dz, mean_xy(e_sgs))))

#Advection Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(Adv_res),z, color='red', ls='--', label=r'$Adv_{res}$')
ax.plot(mean_xy(Adv_sgs),z, color='green', ls='-.', label=r'$Adv_{sgs}$')
# ax.plot(mean_xy(Adv_tot),z, color='black', ls='-', label=r'$Adv_{tot}$')
ax.plot(Adv_F,z, color='grey', ls=':', label=r'$Adv_{res,F}$',marker='.')
ax.plot(Adv_F,z, color='yellow', ls=':', label=r'$Adv_{sgs,F}$',marker='.')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'A', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Advection - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%Buoyancy
g = 9.81 #m/s2
zi = 1000 #m
u_scale = 0.45 #m/s
g_star = g*(zi/(u_scale**2)) #non-dim
# T_scale = 300 #K
T_w = ProjectOn_w(Nx, Ny, Nz, T) #projected the temperature variable from uvp nodes to w nodes
# T0 = 300/T_scale #non-dim
T0 = mean_xy(T_w)  #check the LES code for how fabien computed the Beta parameter, planar average for each level
                         # or used a filter around the grid point

                         
Buoy_res = np.zeros((Nx,Ny,Nz),'d',order='F')
Buoy_sgs = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(1,Nz):
    Buoy_res[:,:,k] = (g_star/T0[k])*(WT[:,:,k] - W[:,:,k]*T_w[:,:,k])
    Buoy_sgs[:,:,k] = (g_star/T0[k])*WTsgs[:,:,k]

#tot
Buoy_tot = Buoy_res + Buoy_sgs

#Buoyancy Finnigan
Buoy_F = np.zeros(Nz)
Buoy_sgs_F = np.zeros(Nz)

for k in range(1,Nz):
    Buoy_F[k] = (g_star/T0[k])*mean_xy(WT-W*T_w)[k]
    Buoy_sgs_F[k] = (g_star/T0[k])*mean_xy(WTsgs)[k]

#Buoyancy Production/Dissipation Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(Buoy_res),z, color='red', ls='--', label=r'$Buoy_{res}$')
ax.plot(mean_xy(Buoy_sgs),z, color='green', ls='-.', label=r'$Buoy_{sgs}$')
ax.plot(mean_xy(Buoy_tot),z, color='black', ls='-', label=r'$Buoy_{tot}$')
ax.plot(Buoy_F,z, color='grey', ls=':', label=r'$Buoy_{res,F}$', marker='.')
ax.plot(Buoy_sgs_F,z, color='yellow', ls=':', label=r'$Buoy_{sgs,F}$', marker='.')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'B', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Buoyancy Production - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%Shear Production

Shear_res = - ProjectOn_w(Nx, Ny, Nz, uu_var*dUdx) - ProjectOn_w(Nx, Ny, Nz, uv_var*dUdy) - uw_var*dUdz \
            - ProjectOn_w(Nx, Ny, Nz, uv_var*dVdx) - ProjectOn_w(Nx, Ny, Nz, vv_var*dVdy) - vw_var*dVdz \
            - uw_var*dWdx - vw_var*dWdy - ww_var*ProjectOn_w(Nx, Ny, Nz, dWdz)
            
Shear_sgs = - ProjectOn_w(Nx, Ny, Nz, txx*dUdx) - ProjectOn_w(Nx, Ny, Nz, tyy*dVdy) - ProjectOn_w(Nx, Ny, Nz, tzz*dWdz) \
            - ProjectOn_w(Nx, Ny, Nz, txy*(dUdy + dVdx)) - txz*(dUdz + dWdx) - tyz*(dVdz + dWdy)
                      
Shear_tot = Shear_res + Shear_sgs

#Shear Prod Finnigan
Sh_F = - mean_xy(uw_var)*ddz_1D_uvp(Nz, dz, mean_xy(U)) \
       - mean_xy(vw_var)*ddz_1D_uvp(Nz, dz, mean_xy(V)) \
       - mean_xy(ww_var)*ProjectOn_w_1D(Nz,ddz_1D_w(Nz, dz, mean_xy(data.data[:,:,:,2])) )

Sh_sgs_F = - mean_xy(txz)*mean_xy(dUdz) \
           - mean_xy(tyz)*mean_xy(dVdz) \
           - ProjectOn_w_1D(Nz, mean_xy(tzz))*ProjectOn_w_1D(Nz,ddz_1D_w(Nz, dz, mean_xy(data.data[:,:,:,2])) )
       
# Sh_sgs_F = - mean_xy(txz)*ddz_1D_uvp(Nz, dz, mean_xy(U)) \
#            - mean_xy(tyz)*ddz_1D_uvp(Nz, dz, mean_xy(V)) \
#            - ProjectOn_w_1D(Nz, mean_xy(tzz))*ProjectOn_w_1D(Nz,ddz_1D_w(Nz, dz, mean_xy(data.data[:,:,:,2])) )
       
# Dispersive Shear Production
Spd = -ProjectOn_w(Nx, Ny, Nz, uud*duddx)-ProjectOn_w(Nx, Ny, Nz, vvd*dvddy)-wwd*ProjectOn_w(Nx, Ny, Nz, dwddz)-uwd*(duddz+dwddx)-vwd*(dvddz+dwddy)-ProjectOn_w(Nx, Ny, Nz,uvd*(duddy+dvddx))
Spd_xyavg = mean_xy(Spd)

# Dispersive Shear Production SGS
Spd_sgs = -ProjectOn_w(Nx, Ny, Nz, txxd*duddx)-ProjectOn_w(Nx, Ny, Nz, tyyd*dvddy)-ProjectOn_w(Nx, Ny, Nz, tzzd*dwddz)-txzd*(duddz+dwddx)-tyzd*(dvddz+dwddy)-ProjectOn_w(Nx, Ny, Nz,txyd*(duddy+dvddx))
Spd_sgs_xyavg = mean_xy(Spd_sgs)

#Shear Production Profiles
figure, ax = plt.subplots(figsize=(10,8))

# ax.plot(mean_xy(Shear_res),z, color='red', ls='--', label=r'$SH_{res}$', marker='.')
# ax.plot(mean_xy(Shear_sgs),z, color='green', ls='-.', label=r'$SH_{sgs}$', marker='.')
# ax.plot(mean_xy(Shear_res_u),z, color='blue', ls='--', label=r'$SH_{res,u}$', marker='.')
# ax.plot(mean_xy(Shear_sgs_u),z, color='black', ls='-.', label=r'$SH_{sgs,u}$', marker='.')
ax.plot(mean_xy(Shear_tot),z, color='black', ls='-', label=r'$SH_{tot}$')
ax.plot(mean_xy(Shear_tot_u),z, color='blue', ls='-', label=r'$SH_{tot,u}$')
# ax.plot(Sh_F,z, color='grey', ls=':', label=r'$SH_F$')
# ax.plot(Sh_sgs_F,z, color='grey', ls=':', label=r'$SH_{sgs,F}$')
# ax.plot(Spd_xyavg, z, color='blue', ls=':', label=r'$SH_D$')
# ax.plot(Spd_sgs_xyavg, z, color='blue', ls=':', label=r'$SH_{sgs,D}$')
# ax.plot(Sh_F+Spd_xyavg,z, color='yellow', label=r'$SH_{res,F+D}$')
# ax.plot(Sh_sgs_F+Spd_sgs_xyavg,z, color='orange', label=r'$SH_{sgs,F+D}$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'SH', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Shear Production - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')


# figure = plt.figure()
# plt.plot(mean_xy(ProjectOn_w(Nx, Ny, Nz, uv_var*dUdy) + ProjectOn_w(Nx, Ny, Nz, uv_var*dVdx)),z, label='res')
# plt.plot(mean_xy(ProjectOn_w(Nx, Ny, Nz, txy*(dUdy + dVdx))),z,label='sgs')
# plt.legend()
#%% Turbulent Transport tt

uuu_var = UUU - 3*UU*U + 2*U*U*U
uvv_var = UVV - 2*UV*V - U*VV + 2*U*V*V
uww_var = ProjectOn_uvp(Nx, Ny, Nz, UWW) - 2*ProjectOn_uvp(Nx, Ny, Nz, UW)*ProjectOn_uvp(Nx, Ny, Nz, W) - U*ProjectOn_uvp(Nx, Ny, Nz, WW) + 2*U*ProjectOn_uvp(Nx, Ny, Nz, W*W)
vuu_var = VUU - 2*UV*U - V*UU + 2*V*U*U
vvv_var = VVV - 3*VV*V + 2*V*V*V
vww_var = ProjectOn_uvp(Nx, Ny, Nz, VWW) - 2*ProjectOn_uvp(Nx, Ny, Nz, VW*W) - V*ProjectOn_uvp(Nx, Ny, Nz, WW) + 2*V*ProjectOn_uvp(Nx, Ny, Nz, W*W)
wuu_var = ProjectOn_uvp(Nx, Ny, Nz, WUU) - 2*ProjectOn_uvp(Nx, Ny, Nz, UW)*U - ProjectOn_uvp(Nx, Ny, Nz, W)*UU + 2*ProjectOn_uvp(Nx, Ny, Nz, W)*U*U
wvv_var = ProjectOn_uvp(Nx, Ny, Nz, WVV) - 2*ProjectOn_uvp(Nx, Ny, Nz, VW)*V - ProjectOn_uvp(Nx, Ny, Nz, W)*VV + 2*ProjectOn_uvp(Nx, Ny, Nz, W)*V*V
www_var = ProjectOn_uvp(Nx, Ny, Nz, WWW) - 3*ProjectOn_uvp(Nx, Ny, Nz, WW*W) + 2*ProjectOn_uvp(Nx, Ny, Nz,W*W*W )

ttx = - Fddx(Nx, Ny, Nz, 0.5*(uuu_var + uvv_var + uww_var), 1, 0, Lx, Nx)
tty = - Fddy(Nx, Ny, Nz, 0.5*(vuu_var + vvv_var + vww_var), 1, 0, Ly, Ny)
ttz = - ProjectOn_uvp(Nx, Ny, Nz, ddz3_uv(Nx, Ny, Nz, dz, 0.5*(wuu_var + wvv_var + www_var)))

tt_tot_res = ttx + tty + ttz

# uuu_var_sgs = Utxx - U*txx
# uvv_var_sgs = Utyy - U*tyy
# uww_var_sgs = Utzz - U*tzz
# vuu_var_sgs = Vtxx - V*txx
# vvv_var_sgs = Vtyy - V*tyy
# vww_var_sgs = Vtzz - V*tzz
# wuu_var_sgs = Wtxx - W*ProjectOn_w(Nx, Ny, Nz, txx)
# wvv_var_sgs = Wtyy - W*ProjectOn_w(Nx, Ny, Nz, tyy)
# www_var_sgs = Wtzz - W*ProjectOn_w(Nx, Ny, Nz, tzz)

# ttx_sgs = - Fddx(Nx, Ny, Nz, 0.5*(uuu_var_sgs + uvv_var_sgs + uww_var_sgs), 1, 0, Lx, Nx)
# tty_sgs = - Fddy(Nx, Ny, Nz, 0.5*(vuu_var_sgs + vvv_var_sgs + vww_var_sgs), 1, 0, Ly, Ny)
# ttz_sgs = - ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, 0.5*(wuu_var_sgs + wvv_var_sgs + www_var_sgs)))

# tt_tot_sgs = ProjectOn_w(Nx, Ny, Nz, ttx_sgs) + ProjectOn_w(Nx, Ny, Nz, tty_sgs) + ttz_sgs

ttx_sgs = - Fddx(Nx, Ny, Nz, 0.5*((Utxx - U*txx) + (Vtxy - V*txy) + ProjectOn_uvp(Nx, Ny, Nz, (Wtxz - W*txz))), 1, 0, Lx, Nx)
tty_sgs = - Fddy(Nx, Ny, Nz, 0.5*((Utxy - U*txy) + (Vtyy - V*tyy) + ProjectOn_uvp(Nx, Ny, Nz, (Wtyz - W*tyz))), 1, 0, Ly, Ny)
ttz_sgs = - ddz3_w(Nx, Ny, Nz, dz, 0.5*((Utxz - ProjectOn_w(Nx, Ny, Nz, U)*txz) + (Vtyz - ProjectOn_w(Nx, Ny, Nz, V)*tyz) + (Wtzz - W*ProjectOn_w(Nx, Ny, Nz, tzz))))

tt_tot_sgs = ProjectOn_w(Nx, Ny, Nz, ttx_sgs + tty_sgs + ttz_sgs)

tt_tot = tt_tot_res + tt_tot_sgs

#Dispersive Turbulent Transport
# dttx = mean_xy(ProjectOn_w(Nx, Ny, Nz, uud*ud) + ProjectOn_w(Nx, Ny, Nz, vvd*ud) + wwd*ProjectOn_w(Nx, Ny, Nz, ud))
# dtty = mean_xy(ProjectOn_w(Nx, Ny, Nz, uud*vd) + ProjectOn_w(Nx, Ny, Nz, vvd*vd) + wwd*ProjectOn_w(Nx, Ny, Nz, vd))
dttz = mean_xy(ProjectOn_w(Nx, Ny, Nz, uud)*wd + ProjectOn_w(Nx, Ny, Nz, vvd)*wd + wwd*wd)

Dtt_F = - 0.5*ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, dttz))

#Dispersive Turbulent transport SGS
dttz_sgs = mean_xy(ProjectOn_w(Nx, Ny, Nz, txxd)*wd + ProjectOn_w(Nx, Ny, Nz, tyyd)*wd + ProjectOn_w(Nx, Ny, Nz, tzzd)*wd)
# dttz_sgs = mean_xy()
Dtt_sgs_F = - 0.5*ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, dttz_sgs))

#Dispersive term computed by summing u_iu_j+t_i,j and then doing the dispersive subtraction
# dttz_test = mean_xy(ProjectOn_w(Nx, Ny, Nz, uutxxd)*wd + ProjectOn_w(Nx, Ny, Nz, vvtyyd)*wd + )

#Turbulent Transport Finnigan
Ttz_F = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy(0.5*(wuu_var + wvv_var + www_var))))
Ttz_sgs_F = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy(0.5*((Utxz - ProjectOn_w(Nx, Ny, Nz, U)*txz) + (Vtyz - ProjectOn_w(Nx, Ny, Nz, V)*tyz) + (Wtzz - W*ProjectOn_w(Nx, Ny, Nz, tzz))))))

#Turbulent Transport Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(tt_tot_res),z, color='red', ls='--', label=r'$TT_{res}$')
ax.plot(mean_xy(tt_tot_sgs),z, color='green', ls='-.', label=r'$TT_{sgs}$')
# ax.plot(mean_xy(tt_tot),z, color='black', ls='-', label=r'$TT_{tot}$')
# ax.plot(Dtt_F,z, color='grey', ls=':', label=r'$TT_{res,D}$')
# ax.plot(Ttz_F,z, color='orange', ls=':', label=r'$TT_{res,F}$', marker='.')
# ax.plot(Ttz_F+Dtt_F,z+Dtt_sgs_F, color='green', ls=':', label=r'$TT_{res,F+D}$', marker='.')
# ax.plot(Ttz_sgs_F,z, color='grey', ls=':', label=r'$TT_{sgs,F}$', marker='.')
# ax.plot(Dtt_sgs_F,z, color='orange', ls=':', label=r'$TT_{sgs,D}$', marker='.')
# ax.plot(Ttz_sgs_F+Dtt_sgs_F,z, color='orange', ls=':', label=r'$TT_{sgs,F+D}$', marker='.')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'TT', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Turbulent Transport - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%Pressure transport
#p = p_les - 0.5*(uu + vv + ww)
dpu1dx = Fddx(Nx, Ny, Nz, pu1_var, 1, 0, Lx, Nx)
dpv1dy = Fddy(Nx, Ny, Nz, pv1_var, 1, 0, Ly, Ny)
dpw1dz = ddz3_w(Nx, Ny, Nz, dz, pw1_var)

# dpu1dx = ProjectOn_w(Nx, Ny, Nz, Fddx(Nx, Ny, Nz, PU1, 1, 0, Lx, Nx) - Fddx(Nx, Ny, Nz, P1, 1, 0, Lx, Nx)*U)
# dpv1dy = ProjectOn_w(Nx, Ny, Nz, Fddy(Nx, Ny, Nz, PV1, 1, 0, Ly, Ny) - Fddy(Nx, Ny, Nz, P1, 1, 0, Ly, Ny)*V)
# dpw1dz = ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, PW1)) - ddz3_uv(Nx, Ny, Nz, dz, P1)*W

#p = p_les - 0.5*(uu + vv + ww) - (1/3)*(txx + tyy + tzz) (turns out to be the same as the previous pressure, trace term small)
dpu2dx = ProjectOn_w(Nx, Ny, Nz, Fddx(Nx, Ny, Nz, pu2_var, 1, 0, Lx, Nx))
dpv2dy = ProjectOn_w(Nx, Ny, Nz, Fddy(Nx, Ny, Nz, pv2_var, 1, 0, Ly, Ny))
dpw2dz = ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, pw2_var))

#p = p_les (probably incorrect, massive value close to the  surface, worse final residual)
dpu3dx = ProjectOn_w(Nx, Ny, Nz, Fddx(Nx, Ny, Nz, pu3_var, 1, 0, Lx, Nx))
dpv3dy = ProjectOn_w(Nx, Ny, Nz, Fddy(Nx, Ny, Nz, pv3_var, 1, 0, Ly, Ny))
dpw3dz = ProjectOn_w(Nx, Ny, Nz, ddz3_w(Nx, Ny, Nz, dz, pw3_var))

pres_trans_1 = - dpu1dx - dpv1dy - dpw1dz
pres_trans_2 = - dpu2dx - dpv2dy - dpw2dz
pres_trans_3 = - dpu3dx - dpv3dy - dpw3dz

#Pressure Transport Finnigan
Pt1_F = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy(pw1_var)))
Pt2_F = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy(pw2_var)))
Pt3_F = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy(pw3_var)))

#Pressure Transport Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(pres_trans_1),z, color='red', ls='--', label=r'$PT1_{res}$',marker='.')
# ax.plot(mean_xy(pres_trans_2),z, color='green', ls='--', label=r'$PT2_{res}$',marker='.')
# ax.plot(mean_xy(pres_trans_3),z, color='black', ls='--', label=r'$PT3_{res}$')
# ax.plot(Pt1_F,z, color='red', ls=':', label=r'$PT1_{F}$')
# ax.plot(Pt2_F,z, color='green', ls=':', label=r'$PT2_{F}$')
# ax.plot(Pt3_F,z, color='black', ls=':', label=r'$PT3_{F}$',marker='.')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'PT', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Pressure Transport - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%Dissipation (SGS because there is no contribution from resolved scales)
dxx_tot = dxx - txx*dUdx #Sxx at u node
dyy_tot = dyy - tyy*dVdy #Syy at u node
dzz_tot = dzz - tzz*dWdz #Szz at u node
dxy_tot = 2*dxy - txy*(dUdy + dVdx) #multiply by 2 because in the LES there is a 1/2, Sxy at u node
dxz_tot = 2*dxz - txz*(dUdz + dWdx) #Sxz at w node
dyz_tot = 2*dyz - tyz*(dVdz + dWdy) #Syz at w node

diss =  (+ ProjectOn_w(Nx, Ny, Nz, dxx_tot) + ProjectOn_w(Nx, Ny, Nz, dyy_tot) + ProjectOn_w(Nx, Ny, Nz, dzz_tot) + ProjectOn_w(Nx, Ny, Nz, dxy_tot) + dxz_tot + dyz_tot) 

diss_u = dxx_tot + dyy_tot + dzz_tot + dxy_tot + ProjectOn_uvp(Nx, Ny, Nz, dxz_tot) + ProjectOn_uvp(Nx, Ny, Nz, dyz_tot)
#Dissipation Finnigan
Diss_F = mean_xy(diss)

#Dissipation Profiles
figure, ax = plt.subplots(figsize=(10,8))

# ax.plot(mean_xy(dxx_tot),z, color='red', ls='--', label=r'$\epsilon_{dxx}$', marker='.')
# ax.plot(mean_xy(dyy_tot),z, color='green', ls='--', label=r'$\epsilon_{dyy}$', marker='.')
# ax.plot(mean_xy(dzz_tot),z, color='blue', ls='--', label=r'$\epsilon_{dzz}$', marker='.')
# ax.plot(mean_xy(dxy_tot),z, color='black', ls='--', label=r'$\epsilon_{dxy}$', marker='.')
# ax.plot(mean_xy(dxy_tot+dxx_tot+dyy_tot+dzz_tot),z, color='black', ls='--', label=r'$\epsilon$', marker='.')
ax.plot(mean_xy(ProjectOn_uvp(Nx, Ny, Nz, tyz*(dVdz + dWdy))),z, color='orange', ls='--', label=r'$\epsilon_{dyz,T2}$', marker='.')
# ax.plot(mean_xy(ProjectOn_uvp(Nx, Ny, Nz, 2*dxz)),z, color='red', ls='--', label=r'$\epsilon_{dxz,T1}$', marker='.')
# ax.plot(mean_xy(ProjectOn_uvp(Nx, Ny, Nz, 2*dyz)),z, color='blue', ls='--', label=r'$\epsilon_{dyz,T1}$', marker='.')
# ax.plot(mean_xy(ProjectOn_uvp(Nx, Ny, Nz, txz*(dUdz + dWdx))),z, color='green', ls='--', label=r'$\epsilon_{dxz,T2}$', marker='.')
# ax.plot(mean_xy(- Shear_tot),z, color='blue', ls='--', label=r'$Prod$', marker='.')
# ax.plot(mean_xy(ProjectOn_uvp(Nx, Ny, Nz, 2*dyz)),z, color='green', ls='--', label=r'$\epsilon_{dyz}$', marker='.')
# ax.plot(Diss_F,z, color='grey', ls=':', label=r'$\epsilon_{F}$', marker='.')
# ax.plot(mean_xy(dxz),z, color='blue', marker='.')
# ax.plot(mean_xy(dyz),z, color='green', marker='.')



ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'$\epsilon$', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Dissipation - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')
ax.axvline(0,color='black',linewidth=0.5)

#%%
#Additional Terms from Giometto 2016
#%%
#Work of the time-averaged velocity spatial fluctuations against the Double-Averaged shear stress, we get 0
Pm_G = - mean_xy(uw_var)*mean_xy(duddz) - mean_xy(vw_var)*mean_xy(dvddz) - mean_xy(ww_var)*ProjectOn_w_1D(Nz, mean_xy(dwddz))

#SGS transport
SGS_t_G = - ProjectOn_w_1D(Nz, ddz_1D_w(Nz, dz, mean_xy((Utxz - ProjectOn_w(Nx, Ny, Nz, U)*txz) + (Vtyz - ProjectOn_w(Nx, Ny, Nz, V)*tyz) + (Wtzz - W*ProjectOn_w(Nx, Ny, Nz, tzz)))))

#Additional Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(Pm_G,z, color='red', ls='--', label=r'$Work$')

ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'$\frac{\partial <e>}{\partial t}$', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Additional Giometto Term - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')
ax.axvline(0,color='black',linewidth=0.5)

#%%

#SOME PLOTS

#%%Residual
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(- (mean_xy(diss)[2:] + mean_xy(tt_tot)[2:]),z[2:], color='red', ls='-', label=r'$\epsilon$')
# ax.plot(mean_xy(Adv_tot)[1:],z[1:], color='black', ls='-', label=r'$Adv_{tot}$')
ax.axvline(0,color='black',linewidth=0.5)
ax.axhline(0,color='black',linewidth=0.5)
# ax.plot(mean_xy(Buoy_tot)[1:],z[1:], color='green', ls=':', label=r'$Buoy_{tot}$')
ax.plot(mean_xy(Shear_tot)[2:],z[2:], color='orange', ls='-.', label=r'$SH_{tot}$')
# ax.plot(mean_xy(tt_tot)[1:],z[1:], color='grey', ls='-', label=r'$TT_{tot}$')
# ax.plot(mean_xy(pres_trans_1)[1:],z[1:], color='blue', ls='--', label=r'$PT_{res}$')
# ax.plot(mean_xy(diss+Shear_tot),z, color ='blue')
# ax.plot(mean_xy(diss+Shear_tot+Buoy_tot+Adv_tot+tt_tot+pres_trans_1),z, color='green', ls='-', label=r'$R1$')
# ax.plot(mean_xy(diss+Shear_tot+Buoy_tot+Adv_tot+tt_tot+pres_trans_2),z, color='orange', ls='-', label=r'$R2$')
# ax.plot(mean_xy(diss+Shear_tot+Buoy_tot+Adv_tot+tt_tot+pres_trans_3),z, color='red', ls='-', label=r'$R3$')
ax.set_xlabel(r'$\frac{\partial <e>}{\partial t}$', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'TKE Budget - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')
ax.axvline(0,color='black',linewidth=0.5)


#%%

#Shear Production Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(Shear_res),z, color='red', ls='--', label=r'$SH_{res}$')
ax.plot(mean_xy(Shear_sgs),z, color='green', ls='-.', label=r'$SH_{sgs}$')
ax.plot(mean_xy(Shear_tot),z, color='black', ls='-', label=r'$SH_{tot}$')
# ax.plot(Sh_F,z, color='grey', ls=':', label=r'$SH_F$')
# ax.plot(Spd_xyavg, z, color='blue', ls=':', label=r'$SH_D$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'SH', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Shear Production - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%
#Buoyancy Production/Dissipation Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(Buoy_res),z, color='red', ls='--', label=r'$Buoy_{res}$')
ax.plot(mean_xy(Buoy_sgs),z, color='green', ls='-.', label=r'$Buoy_{sgs}$')
ax.plot(mean_xy(Buoy_tot),z, color='black', ls='-', label=r'$Buoy_{tot}$')
# ax.plot(Buoy_F,z, color='grey', ls=':', label=r'$Buoy_{F}$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'B', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Buoyancy Production - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%
#Turbulent Transport Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(tt_tot_res),z, color='red', ls='--', label=r'$TT_{res}$')
ax.plot(mean_xy(tt_tot_sgs),z, color='green', ls='-.', label=r'$TT_{sgs}$')
ax.plot(mean_xy(tt_tot),z, color='black', ls='-', label=r'$TT_{tot}$')
# ax.plot(Dtt_F,z, color='blue', ls=':', label=r'$TT_{D}$')
# ax.plot(Ttz_F,z, color='grey', ls=':', label=r'$TT_{F}$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'TT', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Turbulent Transport - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%
#Pressure Transport Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(pres_trans),z, color='red', ls='--', label=r'$PT_{res}$')
# ax.plot(Pt_F,z, color='grey', ls=':', label=r'$PT_{F}$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'PT', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Pressure Transport - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')

#%%
#Dissipation Profiles
figure, ax = plt.subplots(figsize=(10,8))

ax.plot(mean_xy(diss),z, color='red', ls='--', label=r'$\epsilon$')
# ax.plot(Diss_F,z, color='grey', ls=':', label=r'$\epsilon_{F}$')
ax.axvline(0,color='black',linewidth=0.5)

ax.set_xlabel(r'$\epsilon$', fontsize='15')
ax.set_ylabel(r'$z/z_i$', fontsize='15')
ax.set_title(r'Dissipation - Homog Surf T:300K', fontsize='15')

ax.legend(loc='upper right', fontsize='10')
ax.axvline(0,color='black',linewidth=0.5)

