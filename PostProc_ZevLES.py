#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:51:18 2023

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import os
import sys
import copy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

#%%
#inputs
# sim = 'flat_256x256x384_out5hr'
sim = 'flat_256x256x384_out5hr_v2'
# sim = 'flat_256x256x384_out30min'
# sim = 'flat_nocanopy_256x256x384_out5hr'
# sim = 'flat_256x256x384_001_1hr'
# sim = 'flat_256x256x384_005_1hr'
# sim = 'flat_256x256x384_001_2min'
# sim = 'ATTO1_256x256x384_v2_out5hr'
# sim = 'ATTO2_256x256x384_out5hr'
# sim = 'ATTO12_256x256x384_out5hr'
# sim = 'ATTO_256x256x384_full_out5hr'
# sim = 'ATTO_256x256x384_full_out30min'
# sim = 'bicheng_hill_256x256x384_out5hr'
# sim = 'bicheng_hill_256x256x384_out30min'
# sim = 'ATTO_256x256x384_ts_120min'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/sim'+sim+'/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

# pcnt3 = 36000 #18000; %Important! used in get_var 
# avg_time = pcnt3*10 #total timesteps averaged  Important! used in get_var
# startavg = 1
# endavg = 1 # max is avg_time/pcnt3
# incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
# slice_profiles = False
# shifting_z = False
# tke_budget_flag = True
# derivatives_flag = True
# PCON_flag = False
# PCON_TEMP_flag = False
# v_fracflag = False
# SAVEAS_flag = False
# clear_var_flag = False

# T_STC = 305 #320; %298.15; %[K], temperature scale
# dt = 0.01 #05; %0.000005;
zi = 1000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

# simPath = '/scratch/general/nfs1/u1450851/LES_Sims/'
simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'
# figPath = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/ZevLES/ATTO1_JETtest/'
# simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/ATTO_sims/';

# if SAVEAS_flag:
#     output_directory = os.path.join(simPath, f'sim{sim}', f'output/sim{sim}DOEplots')
#     os.makedirs(output_directory, exist_ok=True)
#     plotsPath = output_directory

# Change the current working directory
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/MATLAB_Codes/MATLAB_postproc_Zev')

# Append a directory to the Python path
sys.path.append('./functions/')

#%% Plot ke.txt for convergence checking

# parentpath = simPath + f'sim{sim}/'
# file_name = 'ke.txt'

# # Load and import data
# ke = np.genfromtxt(parentpath + file_name)
# ts = np.linspace(0, len(ke) * wbase, len(ke))

# # Create the plot
# plt.figure()
# plt.plot(ts, ke, 'k-', linewidth=2.5)
# plt.xlabel('timesteps')
# plt.ylabel('MKE')
# plt.gca().set_facecolor('white')  # Set background color to white

# # plt.savefig(figPath+'ke.png',dpi=300,facecolor='white', edgecolor='white')
# # Show the plot
# plt.show()

#%% Import simulation parameters and build topography

# Input and output paths
ipath = simPath + f'sim{sim}/output/ta1_field/'
opath = simPath + f'sim{sim}/output/'

# Read parameter file for ta1_field
with open(ipath + 'parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

nx = int(param[0])
ny = int(param[1])
nz = int(param[2])
lx = param[3]
ly = param[4]
lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
nt = int(param[9])
actLaunch = int(param[10])
mpiProc = int(param[11])
Re = param[12]
Ri = param[13]
Pr = param[14]
alpha = param[15]
SGS = param[16]
S_FLAG = param[17]
ibm = param[18]
rot = param[19]
nzTot = nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

# if shifting_z:
#     # Modify axes to move 0 to IBM surface
#     z_uvp = z_uvp[5:]
#     z_uvp = z_uvp - z_uvp[0]
#     z_w = z_w[5:]
#     z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))
    # nt = len(ke) // pcnt3
elif ibm == 1:
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
    intf, iintf = build_intf(phi, dz)
else:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))

if ibm:
    phi_uv = np.zeros_like(phi)
    for k in range(nz - 1):
        phi_uv[:, :, k] = (phi[:, :, k] + phi[:, :, k + 1]) / 2.0

# if PCON_flag and v_fracflag:
#     v_frac = get_var(opath + 'cut_cell/', 'v_frac', nx, ny, nzTot, 1, iintf, mpiProc)
#     a_cut = get_var(opath + 'cut_cell/', 'a_cut', nx, ny, nzTot, 1, iintf, mpiProc)

#%% plot the topography or IBM

# def find_coordinates(array):
#     coordinates = []
#     for i in range(len(array)):
#         for j in range(len(array[i])):
#             if array[i][j] > 0.101:
#                 coordinates.append((i, j))
#     return coordinates

# coord = find_coordinates(intf)

# x_coords = [40,129,218,40,218,40,129,218,129]
# y_coords = [50,128,200,200,50,128,50,128,200]

# x_coords = [84,84,84,173,173,173] #bicheng valley coords
# y_coords = [50,128,200,200,50,128]

# zi = 1000
# fig, ax = plt.subplots()
# contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
# for i in range(len(x_coords)):
#     ax.scatter(x_coords[i]*dx*zi,y_coords[i]*dy*zi,c='r')
# # for i in range(len(coord)):
#     # ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
# # contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
# plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
# plt.xlabel('$x$', usetex=True)
# plt.ylabel('$y$', usetex=True)
# plt.colorbar(contour)
# ax.set_aspect('auto')
# plt.gca().set_facecolor('white')

# # plt.savefig(pathOUT+'Figures/ts_towrs.png',dpi=300,facecolor='white', edgecolor='white')

# plt.show()

 #%%Compute the slopes
# slope = np.zeros((nx))
# for i in range(0,nx-1):
#     slope[i] = math.atan(abs((intf[i+1,1]-intf[i,1])/dx))
    
#%% Loading the topography data:

zeds = np.zeros((nx,ny,nz))

for i in range(0,nx):
    for j in range(0,ny):
        zeds[i,j,:] = np.arange(0,nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            
#%%
# index = []
# for i in range(len(x_coords)):
#     tmp = dist[x_coords[i],y_coords[i]]
#     for j in range(len(tmp) - 1):
#         if tmp[j] < 0 and tmp[j + 1] >= 0:
#             index.append(j)
#%%

# nt_tot = nt * actLaunch - incskip
# print('nt_tot is', nt_tot)

# avgT = int(avg_time / pcnt3)
# print('avgT is', avgT)

# print('startavg is', startavg)
# print('endavg is', endavg)

# timescale = zi / ustar
# timesteps = pcnt3 * (list(range(nt_tot - nt + 1, nt_tot + 1)))
# time = [t * timescale * dt / 60 for t in timesteps]

#%%Import variables

import pickle

save_data = False
load_data = True

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
       'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
       'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
       'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']
    
if load_data:
    with open(pathOUT+'data.pkl', 'rb') as f:
    # Load the data from the pickle file
        data = pickle.load(f)
        
    data_tavg = dict()
    for i in range(len(var)):
        data_tavg[var[i]] = np.mean(data[var[i]],axis=3)
else:
    data = dict()
    data_tavg = dict()

    for i in range(len(var)):
        # data[var[i]] = get_var(ipath,var[i],nx,ny,nzTot,nt_tot,iintf,mpiProc,avgT)
        data_tavg[var[i]] = np.mean(data[var[i]],axis=3)

if save_data:
    with open(pathOUT+'data.pkl', 'wb') as f:
        pickle.dump(data, f)
    


# if ibm == 1:
#     if shifting_z:
#         for i in range(len(var)):
#             data[var[i]][:,:,0:nz-4,:] = data[var[i]][:,:,4:,:]
#             data[var[i]][:,:,nz-4:nz,:] = float('nan')
#             data_tavg[var[i]][:,:,0:nz-4] = data_tavg[var[i]][:,:,4:]
#             data_tavg[var[i]][:,:,nz-4:nz] = float('nan')
#         intf = intf - z_shift
#         iintf = iintf - z_shift
#         phi_reshape = phi[:, :, 4:]  # Remove the first 4 z-levels

#%%Plot Reynolds stress, SGS stress and dispersive stress

Rxz = data_tavg['uw']-data_tavg['u']*data_tavg['w']
TxzTxz = Rxz - data_tavg['txz']
U_d = np.zeros_like(data_tavg['u'])
W_d = np.zeros_like(data_tavg['w'])

for i in range(0,nzTot):
    U_d[:,:,i] = data_tavg['u'][:,:,i]-np.mean(data_tavg['u'][:,:,i],axis=(0,1))
    W_d[:,:,i] = data_tavg['w'][:,:,i]-np.mean(data_tavg['w'][:,:,i],axis=(0,1))
    
UW_D = U_d*W_d

fig,axs = plt.subplots(1,1)
# axs.plot(np.mean(Rxz,axis=(0,1)),np.arange(0,nzTot)*dz,c='k')
# axs.plot(np.mean(-data_tavg['txz'],axis=(0,1)),np.arange(0,nzTot)*dz,c='r')
axs.plot(np.mean(UW_D,axis=(0,1))[5:],np.arange(0,nzTot-5)*dz/(39/zi) + ((dz)/2)/(39/zi) ,c='g')
axs.axvline(0,c='k',ls='--')
axs.axhline(1,c='k',ls='--')
axs.set_ylim(0,20)
plt.show()
            
#%% pcolor test plots

var = 'w'
var2 = 'w'
# tmp = copy.deepcopy(data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))
tmp = copy.deepcopy(data_tavg[var])
tmp2 = copy.deepcopy(data_tavg[var2])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
xslice = 150
yslice = 63 #int(nz/2)
zslice = 50

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,tmp[xslice,:,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')
# axs.set_ylim(z_ax[0],z_ax[-1])
axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]),ls='-',c='k')
axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]+(39/zi)),ls='--',c='k')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
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
u_h = data_tavg['u'] #uvpnode2wnode(data_tavg['u'])
v_h = data_tavg['v'] #uvpnode2wnode(data_tavg['v'])


terms_ptb['u2_t'] = data_tavg['uu'] - data_tavg['u']**2
terms_ptb['uv_t'] = data_tavg['uv'] - data_tavg['u']*data_tavg['v']
terms_ptb['uw_t'] = data_tavg['uw'] - data_tavg['u']*data_tavg['w']
# terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data_tavg['vv'] - data_tavg['v']**2
terms_ptb['vw_t'] = data_tavg['vw'] - data_tavg['v']*data_tavg['w']
# terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data_tavg['ww'] -data_tavg['w']**2
# terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']) / 2

#%% Flow Overview plots:

#Mean Velocity profiles
fig, axs=plt.subplots(1,3, constrained_layout=True)

tmp_u = copy.deepcopy(data_tavg['u'])
tmp_v = copy.deepcopy(data_tavg['v'])
tmp_w = copy.deepcopy(data_tavg['w'])

# tmp_u[dist<0] = float("nan")
# tmp_v[dist<0] = float("nan")
# tmp_w[dist<0] = float("nan")

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
Rij['Rxx'] = data_tavg['uu'] - data_tavg['u']**2
Rij['Ryy'] = data_tavg['vv'] - data_tavg['v']**2
Rij['Rzz'] = data_tavg['ww'] - data_tavg['w']**2
Rij['Rxy'] = data_tavg['uv'] - data_tavg['u']*data_tavg['v']
Rij['Rxz'] = data_tavg['uw'] - data_tavg['u']*data_tavg['w']
Rij['Ryz'] = data_tavg['vw'] - data_tavg['v']*data_tavg['w']

# for i in range(len(var_Rij)):
#     Rij[var_Rij[i]][(dist < 0)] = float("nan")

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
# tmp_tke[dist<0] = float('nan')

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

axs[0].plot(x_ax,canopyH-z_shift+intf[:,yslice],color='k')
# axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
# axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])
axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\overline{u}(x,y_{nz/2},z)/u_*$')

axs[1].plot(x_ax,canopyH-z_shift+intf[:,yslice],color='k')
axs[1].set_ylabel(r'$z/z_i$');#axs[1].set_xlabel(r'$x/z_i$') 
axs[1].set_title(r'$\overline{w}(x,y_{nz/2},z)/u_*$')

axs[2].plot(x_ax,canopyH-z_shift+intf[:,yslice],color='k')
axs[2].set_xlabel(r'$x/z_i$'); axs[2].set_ylabel(r'$z/z_i$')
axs[2].set_title(r'$\overline{e}(x,y_{nz/2},z)/u_*^2$')


plt.show()

# plt.savefig(figPath+'u_w_tke_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')

#%%Plot tau wall

fig,axs = plt.subplots(1,2,figsize=(10,6))
axs[0].plot(np.mean(Rij['Rxz'],axis=(0,1)),np.arange(0,nzTot)*dz + dz/2,c='k')
axs[0].plot(np.mean(data_tavg['txz'],axis=(0,1)),np.arange(0,nzTot)*dz + dz/2,c='r')
axs[0].set_ylabel('z')
axs[0].set_xlabel('Stress')
axs[1].plot(np.mean(Rij['Ryz'],axis=(0,1)),np.arange(0,nzTot)*dz + dz/2,c='k')
axs[1].plot(np.mean(data_tavg['tyz'],axis=(0,1)),np.arange(0,nzTot)*dz + dz/2,c='r')
axs[1].set_ylabel('z')
axs[1].set_xlabel('Stress')
plt.show()

#%% Calculate the TKE budget

tij_sign = True

if tij_sign:
    data_tavg['txx'] = -data_tavg['txx']
    data_tavg['txy'] = -data_tavg['txy']
    data_tavg['txz'] = -data_tavg['txz']
    data_tavg['tyy'] = -data_tavg['tyy']
    data_tavg['tyz'] = -data_tavg['tyz']
    data_tavg['tzz'] = -data_tavg['tzz']
    data_tavg['utxx'] = -data_tavg['utxx']
    data_tavg['vtxy'] = -data_tavg['vtxy']
    data_tavg['wtxz'] = -data_tavg['wtxz']
    data_tavg['utxy'] = -data_tavg['utxy']
    data_tavg['vtyy'] = -data_tavg['vtyy']
    data_tavg['wtyz'] = -data_tavg['wtyz']
    data_tavg['utxz'] = -data_tavg['utxz']
    data_tavg['vtyz'] = -data_tavg['vtyz']
    data_tavg['wtzz'] = -data_tavg['wtzz']

# Calculate the derivatives (all on uvp-nodes)
terms_drv['dudx'] = data_tavg['dudx'] #get_dphidx(data_tavg['u'], wn_x)
terms_drv['dudy'] = data_tavg['dudy'] #get_dphidy(data_tavg['u'], wn_y)
terms_drv['dudz'] = data_tavg['dudz'] #get_dphidz(data_tavg['u'], dz)
# terms_drv['dudz'] = wnode2uvpnode(terms_drv['dudz'])

terms_drv['dvdx'] = data_tavg['dvdx'] #get_dphidx(data_tavg['v'], wn_x)
terms_drv['dvdy'] = data_tavg['dvdy'] #get_dphidy(data_tavg['v'], wn_y)
terms_drv['dvdz'] = data_tavg['dvdz'] #get_dphidz(data_tavg['v'], dz)
# terms_drv['dvdz'] = wnode2uvpnode(terms_drv['dvdz'])
terms_drv['dwdx'] = data_tavg['dwdx'] #get_dphidx(data_tavg['w'], wn_x)
# terms_drv['dwdx'] = wnode2uvpnode(terms_drv['dwdx'])
terms_drv['dwdy'] = data_tavg['dwdy'] #get_dphidy(data_tavg['w'], wn_y)
# terms_drv['dwdy'] = wnode2uvpnode(terms_drv['dwdy'])
terms_drv['dwdz'] = data_tavg['dwdz'] #get_dphidz(data_tavg['w'], dz)

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')
ue = data_tavg['u']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
ve = data_tavg['v']*terms_ptb['tke']
dvedy = get_dphidy(ve, wn_y)
# tkez = uvpnode2wnode(terms_ptb['tke'])
we = data_tavg['w']*terms_ptb['tke'] #tkez
dwedz = get_dphidz(we, dz)
dwedz = wnode2uvpnode(dwedz)
terms_bdg['adv_h'] = -duedx - dvedy
terms_bdg['adv_v'] = -dwedz

#SGS
ue_sgs = data_tavg['u']*terms_ptb['tke_SGS']
due_sgsdx = get_dphidx(ue_sgs, wn_x)
ve_sgs = data_tavg['v']*terms_ptb['tke_SGS']
dve_sgsdy = get_dphidy(ve_sgs, wn_y)
# tke_sgsz = uvpnode2wnode(terms_ptb['tke_SGS'])
we_sgs = data_tavg['w']*terms_ptb['tke_SGS'] #tke_sgsz
dwe_sgsdz = get_dphidz(we_sgs, dz)
dwe_sgsdz = wnode2uvpnode(dwe_sgsdz)
terms_bdg['advSGS_h'] = -due_sgsdx - dve_sgsdy
terms_bdg['advSGS_v'] = -dwe_sgsdz

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['advSGS_h'] + terms_bdg['advSGS_v']

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = data_tavg['uw'] #wnode2uvpnode(data_tavg['uw'])
vw_c = data_tavg['vw'] #wnode2uvpnode(data_tavg['vw'])
w_c = data_tavg['w'] #wnode2uvpnode(data_tavg['w'])
w2_c = data_tavg['ww'] #wnode2uvpnode(data_tavg['ww'])

terms_ptb['u3_t'] = data_tavg['uuu'] - 3*data_tavg['u']*data_tavg['uu'] + 2*(data_tavg['u']**3)
terms_ptb['uv2_t'] = data_tavg['vvu'] - 2*data_tavg['v']*data_tavg['uv']\
  + 2*data_tavg['u']*(data_tavg['v']**2) - data_tavg['u']*data_tavg['vv']
terms_ptb['uw2_t'] = data_tavg['wwu'] - 2*w_c*uw_c + 2*data_tavg['u']*(w_c**2)\
  - data_tavg['u']*w2_c
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data_tavg['utxx'] - data_tavg['u']*(data_tavg['txx'])
terms_ptb['vtxy_t'] = data_tavg['vtxy'] - data_tavg['v']*(data_tavg['txy'])
terms_ptb['wtxz_t'] = data_tavg['wtxz'] - data_tavg['w']*(data_tavg['txz'])
# terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(0.5*(terms_ptb['utxx_t']+terms_ptb['vtxy_t']
  +terms_ptb['wtxz_t']), wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

terms_ptb['u2v_t'] = data_tavg['uuv'] - 2*data_tavg['u']*data_tavg['uv']\
  + 2*data_tavg['v']*(data_tavg['u']**2) - data_tavg['v']*data_tavg['uu']
terms_ptb['v3_t'] = data_tavg['vvv'] - 3*data_tavg['v']*data_tavg['vv'] + 2*(data_tavg['v']**3)
terms_ptb['vw2_t'] = data_tavg['wwv'] - 2*w_c*vw_c + 2*data_tavg['v']*w_c**2\
  - data_tavg['v']*w2_c
ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
dvedy = get_dphidy(ve, wn_y)
terms_ptb['utxy_t'] = data_tavg['utxy'] - data_tavg['u']*(data_tavg['txy'])
terms_ptb['vtyy_t'] = data_tavg['vtyy'] - data_tavg['v']*(data_tavg['tyy'])
terms_ptb['wtyz_t'] = data_tavg['wtyz'] - data_tavg['w']*(data_tavg['tyz'])
# terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
dutaudy = get_dphidy(0.5*(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
  +terms_ptb['wtyz_t']), wn_y)
terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = data_tavg['uu'] #uvpnode2wnode(data_tavg['uu'])
v2_h = data_tavg['vv'] #uvpnode2wnode(data_tavg['vv'])
terms_ptb['wu2_t'] = data_tavg['uuw'] - 2*u_h*data_tavg['uw'] + 2*data_tavg['w']*u_h**2\
  - data_tavg['w']*u2_h
terms_ptb['wv2_t'] = data_tavg['vvw'] - 2*v_h*data_tavg['vw'] + 2*data_tavg['w']*v_h**2\
  - data_tavg['w']*v2_h
terms_ptb['w3_t'] = data_tavg['www'] - 3*data_tavg['w']*data_tavg['ww'] + 2*data_tavg['w']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dz)
dwedz = wnode2uvpnode(dwedz)
terms_ptb['utxz_t'] = data_tavg['utxz'] - u_h*(data_tavg['txz'])
terms_ptb['vtyz_t'] = data_tavg['vtyz'] - v_h*(data_tavg['tyz'])
terms_ptb['wtzz_t'] = data_tavg['wtzz'] - data_tavg['w']*(data_tavg['tzz'])
dutaudz = get_dphidz(0.5*(terms_ptb['utxz_t']+terms_ptb['vtyz_t']
  +terms_ptb['wtzz_t']), dz)
dutaudz = wnode2uvpnode(dutaudz)
terms_bdg['uturb_v'] = -dwedz-dutaudz

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
# dpudx = get_dphidx(data_tavg['pu'], wn_x) - data_tavg['u']*get_dphidx(data_tavg['p'], wn_x)
terms_ptb['pu_t'] = data_tavg['pu'] - data_tavg['p']*data_tavg['u']
# terms_ptb['pu_t'] = (data_tavg['pu']-(data_tavg['uuu']+data_tavg['vvu']+data_tavg['wwu']))\
#     - (data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))*data_tavg['u']
# terms_ptb['pu_t'] = (data_tavg['pu']\
#     -(1/3)*(data_tavg['utxx']+data_tavg['utyy']+data_tavg['utzz']))\
#     - (data_tavg['p']\
#     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

# dpvdy = get_dphidy(data_tavg['pv'], wn_y) - data_tavg['v']*get_dphidy(data_tavg['p'], wn_y)
terms_ptb['pv_t'] = data_tavg['pv'] - data_tavg['p']*data_tavg['v']
# terms_ptb['pv_t'] = (data_tavg['pv']-(data_tavg['uuv']+data_tavg['vvv']+data_tavg['wwv']))\
#     - (data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))*data_tavg['v']
# terms_ptb['pv_t'] = (data_tavg['pv']\
#     -(1/3)*(data_tavg['vtxx']+data_tavg['vtyy']+data_tavg['vtzz']))\
#     - (data_tavg['p']\
#     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['v']
dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)

p_h = data_tavg['p'] #uvpnode2wnode(data_tavg['p'])
# dpwdz = get_dphidz(data_tavg['pw'], dz) - uvpnode2wnode(data_tavg['w'])*get_dphidz(data_tavg['p'], dz)
terms_ptb['pw_t'] = data_tavg['pw'] - data_tavg['p']*data_tavg['w']
# terms_ptb['pw_t'] = (data_tavg['pw']-(data_tavg['uuw']+data_tavg['vvw']+data_tavg['www']))\
#     - (data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))*data_tavg['w']
# terms_ptb['pw_t'] = (data_tavg['pw']\
#     -(1/3)*(data_tavg['wtxx']+data_tavg['wtyy']+data_tavg['wtzz']))\
#     - (data_tavg['p']\
#     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['w']
dpwdz = get_dphidz(terms_ptb['pw_t'], dz)
dpwdz = wnode2uvpnode(dpwdz)

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

data_tavg['dissip'] = data_tavg['dxx'] + 2*data_tavg['dxy'] + 2*data_tavg['dxz'] + data_tavg['dyy'] + 2*data_tavg['dyz'] + data_tavg['dzz']

terms_bdg['dissip'] = data_tavg['dissip']\
  - data_tavg['txx']*terms_drv['S11'] - data_tavg['tyy']*terms_drv['S22']\
  - data_tavg['tzz']*terms_drv['S33']\
  - 2*data_tavg['txy']*terms_drv['S12'] - 2*data_tavg['txz']*terms_drv['S13']\
  - 2*data_tavg['tyz']*terms_drv['S23']

terms_bdg['canopy'] = data_tavg['wfdz'] - data_tavg['w']*data_tavg['fdz']
# terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += ((data_tavg['ufdx']-data_tavg['u']*data_tavg['fdx'])\
  +(data_tavg['vfdy']-data_tavg['v']*data_tavg['fdy']))
  
terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] + terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] + terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] + terms_ptb['vw_t']*terms_drv['dwdy']
  + data_tavg['txx']*terms_drv['S11'] + data_tavg['tyy']*terms_drv['S22']
  + 2*data_tavg['txy']*terms_drv['S12'] + data_tavg['txz']*terms_drv['dwdx']
  + data_tavg['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  + data_tavg['txz']*terms_drv['dudz'] + data_tavg['tyz']*terms_drv['dvdz']
  + data_tavg['tzz']*terms_drv['S33']
  )
# terms_bdg['prod_dudz'] = -(terms_ptb['uw_t']*terms_drv['dudz'] + data_tavg['txz']*terms_drv['dudz'])
# terms_bdg['prod_dvdz'] = -(terms_ptb['vw_t']*terms_drv['dvdz'] + data_tavg['tyz']*terms_drv['dvdz'])
# terms_bdg['prod_dwdz'] = -(terms_ptb['w2_t']*terms_drv['dwdz'] + data_tavg['tzz']*terms_drv['dwdz'])

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v']

terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] - terms_bdg['canopy'] - terms_bdg['dissip']\
                    + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
                    + terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] 

print('*'*80)

# np.save(pathOUT + 'TKE_terms_v2.npy', terms_bdg) 

# var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
#             'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']#,'prod_dudz','prod_dvdz','prod_dwdz']

# for i in range(len(var_bdg)):
#     terms_bdg[var_bdg[i]][(dist < 0)] = float("nan")

#%%

fig,axs = plt.subplots(1,1)
# axs.plot(np.mean(terms_ptb['uw2_t'],axis=(0,1)),np.arange(0,nzTot)*dz+dz/2)
axs.plot(np.mean(terms_ptb['wtxz_t'],axis=(0,1)),np.arange(0,nzTot)*dz+dz/2)
plt.show()

#%% Profiles of TKE Budget

z_ax = np.arange(0,nz)*dz + dz/2

plt.figure(figsize=(4,8))
plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=8,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1)),z_ax,c='red',marker='s',markevery=8,label='Adv')
plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=8,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=8,label='P_trans')
plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1)),z_ax,c='green',marker='v',markevery=8,label='Diss')
# plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='Can')
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
plt.hlines(canopyH,-80,80,linestyle='--',colors='k')
plt.hlines(2*canopyH,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,5,linestyle='--',colors='grey')
# plt.vlines(1,0,5,linestyle='--',colors='grey')
# plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(z_ax[0],z_ax[-1])
# plt.xlim(-30,0)
# plt.title(r'Real')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/z_i$')
plt.legend(loc='upper right')
plt.show()

# plt.savefig(path_fig+'TKE_Budget_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')


#%% pcolor test plots

# tmp = copy.deepcopy(data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))
tmp = copy.deepcopy(terms_bdg['dissip'])
# tmp = copy.deepcopy((terms_bdg['prod']-terms_bdg['totdis'])/abs(terms_bdg['totdis']))
# tmp = copy.deepcopy(terms_bdg['adv']/abs(terms_bdg['totdis']))
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,nx)*dx
z_ax = np.arange(0,nz)*dz

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

#y-vorticity vertical slice
# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,150,:].T,cmap= 'bwr',vmin=-30,vmax=30)
# axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data_tavg['u'],dist,dm).T, DispFluct_withTopo(data_tavg['w'],dist,dm).T,
#                density = 1,color=[0.7,0.7,0.7])
# axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
axs.set_ylim(z_ax[0],z_ax[-1])
# axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.5])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'pu_avg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/')

from Stats import ReynoldsStress
from Anisotropy_Functions_BEN import Anisotropy, Anisotropy_Clustering, ColorAnisotropy, phi_m, phi_m_real
from UCLA_npy_to_netCDF import UCLA_npy_to_netCDF
import xarray as xr

#%% Domain Parameters

Nz_SLayer = int(nz)
Ny_Slice = int(ny/2)

height = 20 # Canopy height in grid points.
kappa = 0.4 #vonKarman

x = (np.arange(0,nx)*dx)
x_m = (np.arange(0,nx)*dx)*zi #in meters
y = (np.arange(0,ny)*dy)
z = (np.arange(0,nz)*dz)
z_m = (np.arange(0,nz)*dz)*zi #in meters
z_slayer = (np.arange(0,Nz_SLayer)*dz)
# z = (np.arange(0,Nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_d = (z - 30.2/zi)
z_on_h = z/canopyH

#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor:

Rstress = xr.DataArray(np.ones(shape = (nx,ny,nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(nx,ny,nz,data_tavg['u'],data_tavg['v'],data_tavg['w'],data_tavg['uu'],\
                         data_tavg['vv'],data_tavg['ww'],data_tavg['uv'],data_tavg['uw'],data_tavg['vw'])


# Vertical profile of the vertical shear stress together:
    
# Dxz = np.mean((data.data[:,:,:,6]*data.data[:,:,:,8]),axis=(0,1)) - (np.mean(data.data[:,:,:,6],axis=(0,1))*np.mean(data.data[:,:,:,8],axis=(0,1)))
# Dyz = np.mean((data.data[:,:,:,7]*data.data[:,:,:,8]),axis=(0,1)) - (np.mean(data.data[:,:,:,7],axis=(0,1))*np.mean(data.data[:,:,:,8],axis=(0,1)))

# Dw =  np.sqrt(Dxz**2 + Dyz**2)
# Rw = np.mean((np.sqrt(Rstress.data[:,:,:,4]**2 + Rstress.data[:,:,:,5]**2)),axis=(0,1))
# SGSw = np.mean((np.sqrt(data.data[:,:,:,4]**2 + data.data[:,:,:,5]**2)),axis=(0,1))
    
# tau_wall1D = Rw + SGSw + Dw


#%% Computing turbulence anisotropy and clustering:

anisotropy_compute = 'false'    
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/ATTO1/'
# /uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/ATTO1
# Decide whether to compute turbulence anisotropy and clustering from scratch or read it from file.

if (anisotropy_compute == 'true'):
    
    [xB,yB,AnisType_1D] = Anisotropy_Clustering(nx,ny,Nz_SLayer,Rstress)

    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)


    Anisotropy_clustering = xr.DataArray(np.zeros(shape = (nx*ny*Nz_SLayer,3),order='F'),\
                        dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
        
    Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

    os.chdir(path)
    Anisotropy_clustering.to_netcdf('Anisotropy_clustering.nc')
    
else:
        
    Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering.nc')

    xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
    yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
    AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
    
    xB = np.reshape(xB_1D,(nx,ny,Nz_SLayer))
    yB = np.reshape(yB_1D,(nx,ny,Nz_SLayer))

#------------ End of the IF statement.

cmap = ColorAnisotropy()    


#%% Plot vertical and horizontal slices of xB and yB

# yB[yB<0] = float('nan')

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
levels=np.linspace(0,1,11)
plt1=axs.contourf(x,z[0:Nz_SLayer],np.transpose(np.mean(yB,axis=1)),cmap=cmap,levels=levels,vmin=0,vmax=1)
# plt1=axs.contourf(x,z[0:Nz_SLayer],np.transpose(np.mean(yB,axis=1)),cmap=cmap,levels=8)
# axs.pcolormesh(x,z,np.transpose(mask[:,Ny_Slice,:]),cmap='gray')
# axs.plot(x,z_tpg/zi,color='black')
axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
axs.set_title(r'yB y averaged')
# plt.savefig(figPath+'yB.png',dpi=300,facecolor='white', edgecolor='white')


plt.show()










