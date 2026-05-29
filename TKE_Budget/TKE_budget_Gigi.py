#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:09:44 2023

Compute the TKE budget on UVP nodes

@author: u1450851
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# First upload the files and save the variables 

#Simulation geometry
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
z = np.arange(0,Nz)*dz + dz/2

#Upload NetCDF files containing time averaged variables
path = '/scratch/general/nfs1/u1450851/'
os.chdir(path)

#netcdf files containing the 3D RAV variables
data = xr.open_dataarray('Data_Gigi_TKE.nc')

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

wn_x = 2*np.pi*np.fft.rfftfreq(Nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(Ny, dy)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data[:,:,:,0])
v_h = uvpnode2wnode(data[:,:,:,1])


terms_ptb['u2_t'] = data[:,:,:,4] - data[:,:,:,0]**2
terms_ptb['uv_t'] = data[:,:,:,7] - data[:,:,:,0]*data[:,:,:,1]
terms_ptb['uw_t'] = data[:,:,:,8] - u_h*data[:,:,:,2]
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data[:,:,:,5] - data[:,:,:,1]**2
terms_ptb['vw_t'] = data[:,:,:,9] - v_h*data[:,:,:,2]
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data[:,:,:,6] -data[:,:,:,2]**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data[:,:,:,19]+data[:,:,:,20]+data[:,:,:,21]) / 2

#%%%%

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')
ue = data['u']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
# ve = data_tavg['v']*terms_ptb['tke']
# dvedy = get_dphidy(ve, wn_y)
tkez = uvpnode2wnode(terms_ptb['tke'])
we = data_tavg['w']*tkez
dwedz = get_dphidz(we, dm)
terms_bdg['adv_h'] = -duedx #- dvedy
terms_bdg['adv_v'] = -dwedz

#SGS
ue_sgs = data_tavg['u']*terms_ptb['tke_SGS']
due_sgsdx = get_dphidx(ue_sgs, wn_x)
tke_sgsz = uvpnode2wnode(terms_ptb['tke_SGS'])
we_sgs = data_tavg['w']*tke_sgsz
dwe_sgsdz = get_dphidz(we_sgs, dm)
terms_bdg['advSGS_h'] = -due_sgsdx #- dvedy
terms_bdg['advSGS_v'] = -dwe_sgsdz

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['advSGS_h'] + terms_bdg['advSGS_v']

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = wnode2uvpnode(data_tavg['uw'])
vw_c = wnode2uvpnode(data_tavg['vw'])
w_c = wnode2uvpnode(data_tavg['w'])
w2_c = wnode2uvpnode(data_tavg['w2'])

terms_ptb['u3_t'] = data_tavg['u3'] - 3*data_tavg['u']*data_tavg['u2'] + 2*data_tavg['u']**3
terms_ptb['uv2_t'] = data_tavg['uv2'] - 2*data_tavg['v']*data_tavg['uv']\
  + 2*data_tavg['u']*data_tavg['v']**2 - data_tavg['u']*data_tavg['v2']
terms_ptb['uw2_t'] = data_tavg['uw2'] - 2*w_c*uw_c + 2*data_tavg['u']*w_c**2\
  - data_tavg['u']*w2_c
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data_tavg['utxx'] - data_tavg['u']*data_tavg['txx']
terms_ptb['vtxy_t'] = data_tavg['vtxy'] - data_tavg['v']*data_tavg['txy']
terms_ptb['wtxz_t'] = data_tavg['wtxz'] - data_tavg['w']*data_tavg['txz']
terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(0.5*(terms_ptb['utxx_t']+terms_ptb['vtxy_t']               #there was no 0.5 here, why?
  +terms_ptb['wtxz_t']), wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

# terms_ptb['u2v_t'] = data['u2v'] - 2*data['u']*data['uv']\
#   + 2*data['v']*data['u']**2 - data['v']*data['u2']
# terms_ptb['v3_t'] = data['v3'] - 3*data['v']*data['v2'] + 2*data['v']**3
# terms_ptb['vw2_t'] = data['vw2'] - 2*w_c*vw_c + 2*data['v']*w_c**2\
#   - data['v']*w2_c
# ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
# dvedy = get_dphidy(ve, wn_y)
# terms_ptb['utxy_t'] = data['utxy'] - data['u']*data['txy']
# terms_ptb['vtyy_t'] = data['vtyy'] - data['v']*data['tyy']
# terms_ptb['wtyz_t'] = data['wtyz'] - data['w']*data['tyz']
# terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
# dutaudy = get_dphidy(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
#   +terms_ptb['wtyz_t'], wn_y)
# terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = uvpnode2wnode(data_tavg['u2'])
v2_h = uvpnode2wnode(data_tavg['v2'])
terms_ptb['wu2_t'] = data_tavg['u2w'] - 2*u_h*data_tavg['uw'] + 2*data_tavg['w']*u_h**2\
  - data_tavg['w']*u2_h
terms_ptb['wv2_t'] = data_tavg['v2w'] - 2*v_h*data_tavg['vw'] + 2*data_tavg['w']*v_h**2\
  - data_tavg['w']*v2_h
terms_ptb['w3_t'] = data_tavg['w3'] - 3*data_tavg['w']*data_tavg['w2'] + 2*data_tavg['w']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dm)
terms_ptb['utxz_t'] = data_tavg['utxz'] - u_h*data_tavg['txz']
terms_ptb['vtyz_t'] = data_tavg['vtyz'] - v_h*data_tavg['tyz']
terms_ptb['wtzz_t'] = data_tavg['wtzz'] - data_tavg['w']*data_tavg['tzz']
dutaudz = get_dphidz(0.5*(terms_ptb['utxz_t']+terms_ptb['vtyz_t']               #there was no 0.5 here, why?
  +terms_ptb['wtzz_t']), dm)
terms_bdg['uturb_v'] = -dwedz-dutaudz

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
terms_ptb['pu_t'] = data_tavg['pu'] - data_tavg['p']*data_tavg['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

terms_ptb['pv_t'] = data_tavg['pv'] - data_tavg['p']*data_tavg['v']
# dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)

p_h = uvpnode2wnode(data_tavg['p'])
terms_ptb['pw_t'] = data_tavg['pw'] - p_h*data_tavg['w']
dpwdz = get_dphidz(terms_ptb['pw_t'], dm)

terms_bdg['pturb_h'] = -dpudx  #-dpvdy
terms_bdg['pturb_v'] = -dpwdz

terms_bdg['ptrans'] = terms_bdg['pturb_h'] + terms_bdg['pturb_v']

# Calculate the dissipation rate (all on uvp-nodes)
print('Calculating the dissipation term')
terms_drv['S11'] = terms_drv['dudx']
terms_drv['S12'] = 0.5*(terms_drv['dvdx'])  #terms_drv['dudy'] + 
terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
# terms_drv['S22'] = terms_drv['dvdy']
terms_drv['S23'] = 0.5*(terms_drv['dvdz'])  #+ terms_drv['dwdy']
terms_drv['S33'] = terms_drv['dwdz']

#- data_tavg['tyy']*terms_drv['S22']\
    
terms_bdg['dissip'] = data_tavg['dissip']\
  - data_tavg['txx']*terms_drv['S11']\
  - data_tavg['tzz']*terms_drv['S33']\
  - 2*data_tavg['txy']*terms_drv['S12'] - 2*data_tavg['txz']*terms_drv['S13']\
  - 2*data_tavg['tyz']*terms_drv['S23']

terms_bdg['canopy'] = data_tavg['wFcz'] - data_tavg['w']*data_tavg['Fcz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data_tavg['uFcx']-data_tavg['u']*data_tavg['Fcx'])\
  +(data_tavg['vFcy']-data_tavg['v']*data_tavg['Fcy'])
  
terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] #+ terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] #+ terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] #+ terms_ptb['vw_t']*terms_drv['dwdy']
  + data_tavg['txx']*terms_drv['S11']
  + 2*data_tavg['txy']*terms_drv['S12'] + data_tavg['txz']*terms_drv['dwdx']
  #+ data_tavg['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  + data_tavg['txz']*terms_drv['dudz'] + data_tavg['tyz']*terms_drv['dvdz']
  + data_tavg['tzz']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v']

terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] + terms_bdg['canopy'] + terms_bdg['dissip']\
                    + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
                    + terms_bdg['adv_h'] + terms_bdg['adv_v']
                    

print('*'*80)

#%% Remove topography to create terrain following averages

var_topo = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 'prod_h', 'prod_v']
var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 'prod_h', 'prod_v']

terms_topo = dict()

for i in range(len(var_topo)):
    terms_topo[var_topo[i]] = terrainfollowing_2D(terms_bdg[var_bdg[i]])


#%% Profiles of TKE Budget

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/UCLA_tke/flat/'
# path_fig2 = '/uufs/chpc.utah.edu/common/home/calaf-group2/'

z_ax = np.arange(0,Nz)*dm.dz/h_canopy#(dm.dz/dm.zi)

plt.figure(figsize=(4,8))
# plt.plot(np.mean(terms_bdg['res'],axis=(0)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.mean(terms_bdg['adv'],axis=(0)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.mean(terms_bdg['ttrans'],axis=(0)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.mean(terms_bdg['ptrans'],axis=(0)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.mean(terms_bdg['dissip'],axis=(0)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.mean(terms_bdg['canopy'],axis=(0)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot(np.mean(terms_bdg['prod'],axis=(0)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
plt.plot(np.nanmean(terms_bdg['res'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.nanmean(terms_bdg['dissip'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.nanmean(terms_bdg['canopy'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)),z_ax,c='orange',marker='o',markevery=5,label='Can')
plt.plot(np.nanmean(terms_bdg['prod'],axis=(0))/np.mean(-terms_bdg['totdis'],axis=(0)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.vlines(1,0,5,linestyle='--',colors='grey')
plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(0,5)
plt.xlim(-2,2.5)
plt.title(r'Flat')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_t \right \rangle$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper left')
plt.show()

# plt.savefig(path_fig+'TKE_Budget_Flat_res_norm.png',dpi=300,facecolor='white', edgecolor='white')

