#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy

#%%
#inputs

sim = 'anisotropy_t2'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'

nx = 64
ny = 64
nz = 64
lx = 0.5*np.pi
ly = 0.5*np.pi
lz = 0.5
dx = lx/nx
dy = ly/ny
dz = lz/nz

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 500.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = path
# figPath = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/ZevLES/ATTO1_JETtest/'
# simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/ATTO_sims/';


#%%Import variables


var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgXB','avgYB']
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']
    
# varTKE = ['avgDUDX','avgDUDY','avgDUDZ','avgDVDX','avgDVDY','avgDVDZ','avgDWDX','avgDWDY','avgDWDZ','avgUV2','avgUW2','avgVU2','avgVW2',\
#           'avgWU2','avgWV2','avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ','avgWTXX','avgWTYY','avgWTZZ','avgPU1','avgPV1',\
#           'avgPW1','avgPclean1','avgdxx','avgdyy','avgdzz','avgdxy','avgdxz','avgdyz','avgUTXY','avgUTXZ','avgVTXY','avgVTYZ','avgWTXZ','avgWTYZ',\
#           'avgPU2','avgPV2','avgPW2','avgPU3','avgPV3','avgPW3','avgPclean2']
    
varTKE = ['avgU','avgV','avgW','avgP','avgUU','avgVV','avgWW','avgUV','avgUW','avgVW','avgDUDX','avgDUDY','avgDUDZ','avgDVDX','avgDVDY','avgDVDZ',\
          'avgDWDX','avgDWDY','avgDWDZ','avgTXX','avgTYY','avgTZZ','avgTXY','avgTXZ','avgTYZ','avgUUU','avgUVV','avgUWW',\
          'avgVUU','avgVVV','avgVWW','avgWUU','avgWVV','avgWWW','avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ',\
          'avgWTXX','avgWTYY','avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ','avgUTXZ','avgVTYZ','avgUP','avgVP','avgWP','avgDXX','avgDYY','avgDZZ',\
          'avgDXY','avgDXZ','avgDYZ','avgFDX','avgFDY','avgFDZ','avgUFDX','avgVFDY','avgWFDZ']
    
varSGS = ['avgTXX','avgTYY','avgTZZ','avgTXY','avgTXZ','avgTYZ','avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ','avgWTXX','avgTYY',\
          'avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ','avgUTXZ','avgVTYZ','avgDXX','avgDYY','avgDZZ','avgDXY','avgDXZ','avgDYZ']
    
# data_tkeM1 = xr.open_data_tkearray(path+'data_tke_Momentum_1hr.nc')
# data_tkeM2 = xr.open_data_tkearray(path+'data_tke_Momentum_2hr.nc')
dataM = xr.open_dataarray(path+'Data_Momentum.nc')
# dataS = xr.open_dataarray(path+'Data_Scalar.nc')
# data_tkeS = xr.open_data_tkearray(path+'data_tke_Scalar_3hr.nc')
# dataTKE = xr.open_dataarray(path+'Data_GigiTKE_5hr.nc')

data_mom = dict()
data_sc = dict()
# data_tke2 = dict()
# data_tke = dict()

for i in range(len(var)):
    # data_tke1[var[i]] = data_tkeM1.data_tke[:,:,:,i]
    # data_tke2[var[i]] = data_tkeM2.data_tke[:,:,:,i]
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
# for i in range(len(varS)):
#     data_sc[varS[i]] = dataS.data[:,:,:,i]
    
# for i in range(len(varTKE)):
#     data_tke[varTKE[i]] = dataTKE.data[:,:,:,i]
    
# for i in range(len(varSGS)):
#     data_tke[varSGS[i]] = -data_tke[varSGS[i]]

#%%

# from matplotlib import ticker as mticker

# dz = lz/nz
# z = np.arange(0,nz)*dz + dz/2 #U is defined at the u,v,P nodes!!

# txz = data_tke3['avgtxz']
# txz_sfc = np.mean(txz[:,:,0],axis=(0,1))
# u = data_tke3['avgU']*uscale/np.sqrt(txz_sfc)

# u_xyavg = np.mean(u,axis=(0,1))

# # ax = plt.subplots()
# plt.plot(u_xyavg,z)
# plt.yscale('log')
# # ax.axhline(0,c='black')
# # ax.axvline(0,c='black')
# plt.title(r'Logarithmic profile of $<u>_{xy}$')
# plt.xlabel(r'U/u* [-]')
# plt.ylabel(r'z/zi [-]')
# plt.grid()
# plt.xaxis.set_minor_formatter(mticker.ScalarFormatter())

#%% pcolor test plots

var = 'avgYB'
# var2 = 'avgW'
# tmp = copy.deepcopy(data_tke['p']-(data_tke['uu']+data_tke['vv']+data_tke['ww']))
tmp = copy.deepcopy(data_mom[var])
# tmp2 = copy.deepcopy(data_tke[var2])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.streamplot(X,Y,tmp[:,yslice,:].T,tmp2[:,yslice,:].T)
plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')
# axs.plot((x_ax),(np.mean(topodata_tke['intf'],axis=1) + h_canopy )/dm.zi,'--k')
# axs.set_ylim(z_ax[0],z_ax[-1])

# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var + f' - {yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

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
  phi_c[:, :, -1] = phi_c[:, :, -2]
  return phi_c

#%% Calculate the Reynolds Stresses:
    
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')

# Initialization
terms_ptb = dict()
terms_drv = dict()
terms_bdg = dict.fromkeys(items_bdg, None)

wn_x = 2*np.pi*np.fft.rfftfreq(nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(ny, dy)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data_mom['avgU'])
v_h = uvpnode2wnode(data_mom['avgV'])


terms_ptb['u2_t'] = data_mom['avgU2'] - data_mom['avgU']**2
terms_ptb['uv_t'] = data_mom['avgUV'] - data_mom['avgU']*data_mom['avgV']
terms_ptb['uw_t'] = data_mom['avgUW'] - u_h*data_mom['avgW']
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data_mom['avgV2'] - data_mom['avgV']**2
terms_ptb['vw_t'] = data_mom['avgVW'] - v_h*data_mom['avgW']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data_mom['avgW2'] -data_mom['avgW']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data_mom['avgtxx']+data_mom['avgtyy']+data_mom['avgtzz']) / 2

#%%Plot Reynolds Stresses

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

# plt1 = axs.pcolormesh(x_ax,z_ax,terms_ptb['uv_t'][:,yslice,:].T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,terms_ptb['u2_t'][:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%Compute Anisotropy using the Reynolds stresses

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,terms_ptb['u2_t'],terms_ptb['v2_t'],terms_ptb['w2_t'],terms_ptb['uv_t'],terms_ptb['uw_t'],terms_ptb['vw_t'])

#%% Pcolor of Anisotropy

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

# plt1 = axs.pcolormesh(x_ax,z_ax,yB[:,yslice,:].T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,yB[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()
#%% Flow Overview plots:

#Mean Velocity profiles
fig, axs=plt.subplots(1,3, constrained_layout=True)

tmp_u = copy.deepcopy(data_tke['avgU'])
tmp_v = copy.deepcopy(data_tke['avgV'])
tmp_w = copy.deepcopy(data_tke['avgW'])

axs[0].plot(np.nanmean(tmp_u,axis=(0,1)),z_ax,label='u')
axs[1].plot(np.nanmean(tmp_v,axis=(0,1)),z_ax,label='v')
axs[2].plot(np.nanmean(tmp_w,axis=(0,1)),z_ax,label='w')
axs[0].set_ylabel('$z/z_i$');axs[1].set_ylabel('$z/z_i$');axs[2].set_ylabel('$z/z_i$');
axs[0].set_ylim(z_ax[0],z_ax[-1]);axs[1].set_ylim(z_ax[0],z_ax[-1]);axs[2].set_ylim(z_ax[0],z_ax[-1])
axs[0].legend(); axs[1].legend(); axs[2].legend()

# plt.savefig(figPath+'mean_vel_prof.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

var_Rij = ['Rxx', 'Ryy', 'Rzz', 'Rxy', 'Rxz', 'Ryz']

Rij = dict()
Rij['Rxx'] = data_tke['avgUU'] - data_tke['avgU']**2
Rij['Ryy'] = data_tke['avgVV'] - data_tke['avgV']**2
Rij['Rzz'] = data_tke['avgWW'] - data_tke['avgW']**2
Rij['Rzz'] = wnode2uvpnode(Rij['Rzz'])
Rij['Rxy'] = data_tke['avgUV'] - data_tke['avgU']*data_tke['avgV']
Rij['Rxz'] = data_tke['avgUW'] - u_h*data_tke['avgW']
Rij['Rxz'] = wnode2uvpnode(Rij['Rxz'])
Rij['Ryz'] = data_tke['avgVW'] - v_h*data_tke['avgW']
Rij['Ryz'] = wnode2uvpnode(Rij['Ryz'])

#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.nanmean(Rij['Rxx'],axis=(0,1)),z_ax)
axs[0,1].plot(np.nanmean(Rij['Ryy'],axis=(0,1)),z_ax)
axs[0,2].plot(np.nanmean(Rij['Rzz'],axis=(0,1)),z_ax)
axs[1,0].plot(np.nanmean(Rij['Rxy'],axis=(0,1)),z_ax)
axs[1,1].plot(np.nanmean(Rij['Rxz'],axis=(0,1)),z_ax)
axs[1,2].plot(np.nanmean(Rij['Ryz'],axis=(0,1)),z_ax)

axs[0,0].set_ylim(z_ax[0],z_ax[-1]);axs[0,1].set_ylim(z_ax[0],z_ax[-1]);axs[0,2].set_ylim(z_ax[0],z_ax[-1])
axs[1,0].set_ylim(z_ax[0],z_ax[-1]);axs[1,1].set_ylim(z_ax[0],z_ax[-1]);axs[1,2].set_ylim(z_ax[0],z_ax[-1])

axs[0,0].set_xlabel('Rxx');axs[0,1].set_xlabel('Ryy');axs[0,2].set_xlabel('Rzz')
axs[1,0].set_xlabel('Rxy');axs[1,1].set_xlabel('Rxz');axs[1,2].set_xlabel('Ryz')

axs[0,0].set_ylabel('$z/z_i$');axs[1,0].set_ylabel('$z/z_i$');

# plt.savefig(figPath+'Rij.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()


#%% Graphical Representation of U,W,TKE:

tmp_tke = copy.deepcopy(terms_ptb['tke'])

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
# plt1 = axs[0].pcolormesh(x_ax,z_ax,np.nanmean(tmp_u,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt2 = axs[1].pcolormesh(x_ax,z_ax,np.nanmean(tmp_w,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt3 = axs[2].pcolormesh(x_ax,z_ax,np.nanmean(tmp_tke,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
plt1 = axs[0].pcolormesh(x_ax,z_ax,tmp_u[:,yslice,:].T,cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,tmp_w[:,yslice,:].T,cmap='YlGnBu',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,tmp_tke[:,yslice,:].T,cmap='YlGnBu',shading='gouraud')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])


# axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
# axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])
axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\overline{u}(x,y_{nz/2},z)/u_*$')


axs[1].set_ylabel(r'$z/z_i$');#axs[1].set_xlabel(r'$x/z_i$') 
axs[1].set_title(r'$\overline{w}(x,y_{nz/2},z)/u_*$')


axs[2].set_xlabel(r'$x/z_i$'); axs[2].set_ylabel(r'$z/z_i$')
axs[2].set_title(r'$\overline{e}(x,y_{nz/2},z)/u_*^2$')


plt.show()

# plt.savefig(figPath+'u_w_tke_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')


#%% Calculate the TKE budget

# Calculate the derivatives (all on uvp-nodes)
terms_drv['dudx'] = data_tke['avgDUDX'] #get_dphidx(data_tke['u'], wn_x)
terms_drv['dudy'] = data_tke['avgDUDY'] #get_dphidy(data_tke['u'], wn_y)
terms_drv['dudz'] = data_tke['avgDUDZ'] #get_dphidz(data_tke['u'], dz)
terms_drv['dudz'] = wnode2uvpnode(terms_drv['dudz'])

terms_drv['dvdx'] = data_tke['avgDVDX'] #get_dphidx(data_tke['v'], wn_x)
terms_drv['dvdy'] = data_tke['avgDVDY'] #get_dphidy(data_tke['v'], wn_y)
terms_drv['dvdz'] = data_tke['avgDVDZ'] #get_dphidz(data_tke['v'], dz)
terms_drv['dvdz'] = wnode2uvpnode(terms_drv['dvdz'])

terms_drv['dwdx'] = data_tke['avgDWDX'] #get_dphidx(data_tke['w'], wn_x)
terms_drv['dwdx'] = wnode2uvpnode(terms_drv['dwdx'])
terms_drv['dwdy'] = data_tke['avgDWDY'] #get_dphidy(data_tke['w'], wn_y)
terms_drv['dwdy'] = wnode2uvpnode(terms_drv['dwdy'])
terms_drv['dwdz'] = data_tke['avgDWDZ'] #get_dphidz(data_tke['w'], dz)

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')
ue = data_tke['avgU']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
ve = data_tke['avgV']*terms_ptb['tke']
dvedy = get_dphidy(ve, wn_y)
tkez = uvpnode2wnode(terms_ptb['tke'])
we = data_tke['avgW']*tkez
dwedz = get_dphidz(we, dz)
terms_bdg['adv_h'] = -duedx - dvedy
terms_bdg['adv_v'] = -dwedz

#SGS
ue_sgs = data_tke['avgU']*terms_ptb['tke_SGS']
due_sgsdx = get_dphidx(ue_sgs, wn_x)
ve_sgs = data_tke['avgV']*terms_ptb['tke_SGS']
dve_sgsdy = get_dphidy(ve_sgs, wn_y)
tke_sgsz = uvpnode2wnode(terms_ptb['tke_SGS'])
we_sgs = data_tke['avgW']*tke_sgsz
dwe_sgsdz = get_dphidz(we_sgs, dz)
terms_bdg['advSGS_h'] = -due_sgsdx - dve_sgsdy
terms_bdg['advSGS_v'] = -dwe_sgsdz

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['advSGS_h'] + terms_bdg['advSGS_v']

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = wnode2uvpnode(data_tke['avgUW'])
vw_c = wnode2uvpnode(data_tke['avgVW'])
w_c = wnode2uvpnode(data_tke['avgW'])
w2_c = wnode2uvpnode(data_tke['avgWW'])

terms_ptb['u3_t'] = data_tke['avgUUU'] - 3*data_tke['avgU']*data_tke['avgUU'] + 2*(data_tke['avgU']**3)
terms_ptb['uv2_t'] = data_tke['avgUVV'] - 2*data_tke['avgV']*data_tke['avgUV']\
  + 2*data_tke['avgU']*(data_tke['avgV']**2) - data_tke['avgU']*data_tke['avgVV']
terms_ptb['uw2_t'] = data_tke['avgUWW'] - 2*data_tke['avgW']*data_tke['avgUW'] + 2*u_h*(data_tke['avgW']**2)\
  - u_h*data_tke['avgWW']
terms_ptb['uw2_t'] = wnode2uvpnode(terms_ptb['uw2_t'])
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data_tke['avgUTXX'] - data_tke['avgU']*data_tke['avgTXX']
terms_ptb['vtxy_t'] = data_tke['avgVTXY'] - data_tke['avgV']*data_tke['avgTXY']
terms_ptb['wtxz_t'] = data_tke['avgWTXZ'] - data_tke['avgW']*data_tke['avgTXZ']
terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(0.5*(terms_ptb['utxx_t']+terms_ptb['vtxy_t']
  +terms_ptb['wtxz_t']), wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

terms_ptb['u2v_t'] = data_tke['avgVUU'] - 2*data_tke['avgU']*data_tke['avgUV']\
  + 2*data_tke['avgV']*(data_tke['avgU']**2) - data_tke['avgV']*data_tke['avgUU']
terms_ptb['v3_t'] = data_tke['avgVVV'] - 3*data_tke['avgV']*data_tke['avgVV'] + 2*(data_tke['avgV']**3)
terms_ptb['vw2_t'] = data_tke['avgVWW'] - 2*data_tke['avgW']*data_tke['avgVW'] + 2*data_tke['avgV']*data_tke['avgW']**2\
  - data_tke['avgV']*data_tke['avgWW']
terms_ptb['vw2_t'] = wnode2uvpnode(terms_ptb['vw2_t'])
ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
dvedy = get_dphidy(ve, wn_y)
terms_ptb['utxy_t'] = data_tke['avgUTXY'] - data_tke['avgU']*data_tke['avgTXY']
terms_ptb['vtyy_t'] = data_tke['avgVTYY'] - data_tke['avgV']*data_tke['avgTYY']
terms_ptb['wtyz_t'] = data_tke['avgWTYZ'] - data_tke['avgW']*data_tke['avgTYZ']
terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
dutaudy = get_dphidy(0.5*(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
  +terms_ptb['wtyz_t']), wn_y)
terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = uvpnode2wnode(data_tke['avgUU'])
v2_h = uvpnode2wnode(data_tke['avgVV'])
terms_ptb['wu2_t'] = data_tke['avgWUU'] - 2*u_h*data_tke['avgUW'] + 2*data_tke['avgW']*u_h**2\
  - data_tke['avgW']*u2_h
terms_ptb['wv2_t'] = data_tke['avgWVV'] - 2*v_h*data_tke['avgVW'] + 2*data_tke['avgW']*v_h**2\
  - data_tke['avgW']*v2_h
terms_ptb['w3_t'] = data_tke['avgWWW'] - 3*data_tke['avgW']*data_tke['avgWW'] + 2*data_tke['avgW']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dz)
# dwedz = wnode2uvpnode(dwedz)
terms_ptb['utxz_t'] = data_tke['avgUTXZ'] - u_h*data_tke['avgTXZ']
terms_ptb['vtyz_t'] = data_tke['avgVTYZ'] - v_h*data_tke['avgTYZ']
terms_ptb['wtzz_t'] = data_tke['avgWTZZ'] - data_tke['avgW']*uvpnode2wnode(data_tke['avgTZZ'])
dutaudz = get_dphidz(0.5*(terms_ptb['utxz_t']+terms_ptb['vtyz_t']
  +terms_ptb['wtzz_t']), dz)
# dutaudz = wnode2uvpnode(dutaudz)
terms_bdg['uturb_v'] = -dwedz-dutaudz

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
# dpudx = get_dphidx(data_tke['pu'], wn_x) - data_tke['u']*get_dphidx(data_tke['p'], wn_x)
# terms_ptb['pu_t'] = data_tke['avgUP'] - data_tke['avgP']*uvpnode2wnode(data_tke['avgU'])
# terms_ptb['pu_t'] = (data_tke['pu']-(data_tke['uuu']+data_tke['vvu']+data_tke['wwu']))\
#     - (data_tke['p']-(data_tke['uu']+data_tke['vv']+data_tke['ww']))*data_tke['u']
terms_ptb['pu_t'] = (wnode2uvpnode(data_tke['avgUP'])-0.5*(data_tke['avgUUU']+data_tke['avgUVV']+wnode2uvpnode(data_tke['avgUWW']))\
    -(1/3)*(data_tke['avgUTXX']+data_tke['avgUTYY']+data_tke['avgUTZZ']))\
    - (wnode2uvpnode(data_tke['avgP'])-0.5*(data_tke['avgUU']+data_tke['avgVV']+wnode2uvpnode(data_tke['avgWW']))\
    -(1/3)*(data_tke['avgTXX']+data_tke['avgTYY']+data_tke['avgTZZ']))*data_tke['avgU']
# terms_ptb['pu_t'] = wnode2uvpnode(terms_ptb['pu_t'])
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)
# dpudx = wnode2uvpnode(dpudx)

# dpvdy = get_dphidy(data_tke['pv'], wn_y) - data_tke['v']*get_dphidy(data_tke['p'], wn_y)
# terms_ptb['pv_t'] = data_tke['avgVP'] - data_tke['avgP']*uvpnode2wnode(data_tke['avgV'])
# terms_ptb['pv_t'] = (data_tke['pv']-(data_tke['uuv']+data_tke['vvv']+data_tke['wwv']))\
#     - (data_tke['p']-(data_tke['uu']+data_tke['vv']+data_tke['ww']))*data_tke['v']
terms_ptb['pv_t'] = (wnode2uvpnode(data_tke['avgVP'])-0.5*(data_tke['avgVUU']+data_tke['avgVVV']+wnode2uvpnode(data_tke['avgVWW']))\
    -(1/3)*(data_tke['avgVTXX']+data_tke['avgVTYY']+data_tke['avgVTZZ']))\
    - (wnode2uvpnode(data_tke['avgP'])-0.5*(data_tke['avgUU']+data_tke['avgVV']+wnode2uvpnode(data_tke['avgWW']))\
    -(1/3)*(data_tke['avgTXX']+data_tke['avgTYY']+data_tke['avgTZZ']))*data_tke['avgV']
# terms_ptb['pv_t'] = wnode2uvpnode(terms_ptb['pv_t'])
dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)
# dpvdy = wnode2uvpnode(dpvdy)

# p_h = data_tke['p'] #uvpnode2wnode(data_tke['p'])
# dpwdz = get_dphidz(data_tke['pw'], dz) - uvpnode2wnode(data_tke['w'])*get_dphidz(data_tke['p'], dz)
# terms_ptb['pw_t'] = data_tke['avgWP'] - data_tke['avgP']*data_tke['avgW']
# terms_ptb['pw_t'] = (data_tke['avgWP']-(data_tke['avgWUU']+data_tke['avgWVV']+data_tke['avgWWW']))\
#     - (data_tke['avgP']-(uvpnode2wnode(data_tke['avgUU']+data_tke['avgVV'])+data_tke['avgWW']))*data_tke['avgW']
terms_ptb['pw_t'] = (data_tke['avgWP']-0.5*(data_tke['avgWUU']+data_tke['avgWVV']+data_tke['avgWWW'])\
    -(1/3)*(data_tke['avgWTXX']+data_tke['avgWTYY']+data_tke['avgWTZZ']))\
    - (data_tke['avgP']-0.5*(uvpnode2wnode(data_tke['avgUU'])+uvpnode2wnode(data_tke['avgVV'])+data_tke['avgWW'])\
    -(1/3)*(uvpnode2wnode(data_tke['avgTXX']+data_tke['avgTYY']+data_tke['avgTZZ'])))*data_tke['avgW']
dpwdz = get_dphidz(terms_ptb['pw_t'], dz)
# dpwdz = wnode2uvpnode(dpwdz)

terms_bdg['pturb_h'] = -dpudx-dpvdy
terms_bdg['pturb_v'] = -dpwdz

terms_bdg['ptrans'] = terms_bdg['pturb_h'] + terms_bdg['pturb_v']

# Calculate the dissipation rate (all on uvp-nodes)
print('Calculating the dissipation term')
terms_drv['S11'] = terms_drv['dudx']
terms_drv['S12'] = 0.5*(terms_drv['dudy'] + terms_drv['dvdx'])
terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
terms_drv['S22'] = terms_drv['dvdy']
terms_drv['S23'] = 0.5*(terms_drv['dvdz'] + terms_drv['dwdy'])
terms_drv['S33'] = terms_drv['dwdz']

data_tke['dissip'] = data_tke['avgDXX'] + 2*data_tke['avgDXY'] + 2*wnode2uvpnode(data_tke['avgDXZ']) + data_tke['avgDYY'] + 2*wnode2uvpnode(data_tke['avgDYZ']) + data_tke['avgDZZ']

terms_bdg['dissip'] = data_tke['dissip']\
  - data_tke['avgTXX']*terms_drv['S11'] - data_tke['avgTYY']*terms_drv['S22']\
  - data_tke['avgTZZ']*terms_drv['S33']\
  - 2*data_tke['avgTXY']*terms_drv['S12'] - 2*wnode2uvpnode(data_tke['avgTXZ'])*terms_drv['S13']\
  - 2*wnode2uvpnode(data_tke['avgTYZ'])*terms_drv['S23']

terms_bdg['canopy'] = data_tke['avgWFDZ'] - data_tke['avgW']*data_tke['avgFDZ']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += ((data_tke['avgUFDX']-data_tke['avgU']*data_tke['avgFDX'])\
  +(data_tke['avgVFDY']-data_tke['avgV']*data_tke['avgFDY']))
  
terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] + terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] + terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] + terms_ptb['vw_t']*terms_drv['dwdy']
  + data_tke['avgTXX']*terms_drv['S11'] + data_tke['avgTYY']*terms_drv['S22']
  + 2*data_tke['avgTXY']*terms_drv['S12'] + wnode2uvpnode(data_tke['avgTXZ'])*terms_drv['dwdx']
  + wnode2uvpnode(data_tke['avgTYZ'])*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  + wnode2uvpnode(data_tke['avgTXZ'])*terms_drv['dudz'] + wnode2uvpnode(data_tke['avgTYZ'])*terms_drv['dvdz']
  + data_tke['avgTZZ']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v']

terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] + terms_bdg['canopy'] + terms_bdg['dissip']\
                    + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
                    + terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] 

print('*'*80)

# np.save(path + 'TKE_terms.npy', terms_bdg) 

var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
            'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']

# for i in range(len(var_bdg)):
#     terms_bdg[var_bdg[i]][(dist < 0)] = float("nan")



#%% Profiles of TKE Budget

z_ax = np.arange(0,nz)*dz + dz/2

plt.figure(figsize=(4,8))
plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=8,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1)),z_ax,c='red',marker='s',markevery=8,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=8,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=8,label='P_trans')
plt.plot(np.nanmean(terms_bdg['dissip'],axis=(0,1)),z_ax,c='green',marker='v',markevery=8,label='Diss')
plt.plot(np.nanmean(terms_bdg['canopy'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='Can')
# plt.plot(np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='black',marker='o',markevery=8,label='TotDis')
plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod')
# plt.plot(np.nanmean(terms_bdg['prod']-terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod-Diss')
# plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')

# plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
# plt.hlines(canopyH,-80,80,linestyle='--',colors='k')
# plt.hlines(2*canopyH,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,5,linestyle='--',colors='grey')
# plt.vlines(1,0,5,linestyle='--',colors='grey')
# plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(z_ax[0],z_ax[-1])
# plt.xlim(-50,50)
# plt.title(r'Real')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/z_i$')
plt.legend(loc='upper right')
plt.show()

# plt.savefig(path_fig+'TKE_Budget_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')


#%% pcolor test plots

# tmp = copy.deepcopy(data_tke['p']-(data_tke['uu']+data_tke['vv']+data_tke['ww']))
# tmp = copy.deepcopy(terms_ptb['pu_t'])
# tmp = copy.deepcopy((terms_bdg['prod']-terms_bdg['totdis'])/abs(terms_bdg['totdis']))
# tmp[dist<0] = float('nan')
tmp = copy.deepcopy(terms_bdg['adv']/abs(terms_bdg['totdis']))

x_ax = np.arange(0,nx)*dx
z_ax = np.arange(0,nz)*dz

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

#y-vorticity vertical slice
# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,200,:].T,cmap= 'bwr',vmin=-3,vmax=3)
# axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data_tke['u'],dist,dm).T, DispFluct_withTopo(data_tke['w'],dist,dm).T,
#                density = 1,color=[0.7,0.7,0.7])
# axs.plot((x_ax),(np.mean(topodata_tke['intf'],axis=1) + h_canopy )/dm.zi,'--k')
axs.set_ylim(z_ax[0],z_ax[-1])
# axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
axs.set_ylim([0,0.5])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'pu_avg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()