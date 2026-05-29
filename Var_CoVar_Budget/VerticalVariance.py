#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:37:17 2024

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
# sim = 'flat_256x256x384_out5hr_v2'
sim = 'flat_256x256x384_001_1hr'
# sim = 'flat_256x256x384_005_1hr'
# sim = 'flat_256x256x384_001_2min'
# sim = 'ATTO1_256x256x384_v2_out5hr'
# sim = 'ATTO2_256x256x384_out5hr'
# sim = 'ATTO12_256x256x384_out5hr'
# sim = 'ATTO_256x256x384_full_out5hr'
# sim = 'bicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/sim'+sim+'/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

pcnt3 = 36000 #18000; %Important! used in get_var 
avg_time = pcnt3*10 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
slice_profiles = False
shifting_z = False
tke_budget_flag = True
derivatives_flag = True
PCON_flag = False
PCON_TEMP_flag = False
v_fracflag = False
SAVEAS_flag = False
clear_var_flag = False

T_STC = 305 #320; %298.15; %[K], temperature scale
dt = 0.01 #05; %0.000005;
zi = 1000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = '/scratch/general/nfs1/u1450851/LES_Sims/'
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

parentpath = simPath + f'sim{sim}/'
file_name = 'ke.txt'

# Load and import data
ke = np.genfromtxt(parentpath + file_name)
ts = np.linspace(0, len(ke) * wbase, len(ke))

# Create the plot
plt.figure()
plt.plot(ts, ke, 'k-', linewidth=2.5)
plt.xlabel('timesteps')
plt.ylabel('MKE')
plt.gca().set_facecolor('white')  # Set background color to white

# plt.savefig(figPath+'ke.png',dpi=300,facecolor='white', edgecolor='white')
# Show the plot
plt.show()

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

if shifting_z:
    # Modify axes to move 0 to IBM surface
    z_uvp = z_uvp[5:]
    z_uvp = z_uvp - z_uvp[0]
    z_w = z_w[5:]
    z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))
    nt = len(ke) // pcnt3
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

if PCON_flag and v_fracflag:
    v_frac = get_var(opath + 'cut_cell/', 'v_frac', nx, ny, nzTot, 1, iintf, mpiProc)
    a_cut = get_var(opath + 'cut_cell/', 'a_cut', nx, ny, nzTot, 1, iintf, mpiProc)

#%% plot the topography or IBM

# def find_coordinates(array):
#     coordinates = []
#     for i in range(len(array)):
#         for j in range(len(array[i])):
#             if array[i][j] > 0.101:
#                 coordinates.append((i, j))
#     return coordinates

# coord = find_coordinates(intf)

zi = 1000
fig, ax = plt.subplots()
contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
# for i in range(len(coord)):
    # ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
# contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
plt.xlabel('$x$', usetex=True)
plt.ylabel('$y$', usetex=True)
plt.colorbar(contour)
ax.set_aspect('auto')
plt.gca().set_facecolor('white')
# plt.savefig(figPath+'topo.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Compute the slopes
slope = np.zeros((nx))
for i in range(0,nx-1):
    slope[i] = math.atan(abs((intf[i+1,1]-intf[i,1])/dx))
    
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

nt_tot = nt * actLaunch - incskip
print('nt_tot is', nt_tot)

avgT = int(avg_time / pcnt3)
print('avgT is', avgT)

print('startavg is', startavg)
print('endavg is', endavg)

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
        data[var[i]] = get_var(ipath,var[i],nx,ny,nzTot,nt_tot,iintf,mpiProc,avgT)
        data_tavg[var[i]] = np.mean(data[var[i]],axis=3)

if save_data:
    with open(pathOUT+'data.pkl', 'wb') as f:
        pickle.dump(data, f)
  
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
  'pturb_t', 'pturb_s', 'canopy', 'dissip')

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

#%%Compute the uu variance tranport eqution terms

#ADVECTION
terms_bdg['adv_h'] = - data_tavg['u']*get_dphidx(terms_ptb['w2_t'], wn_x) - data_tavg['v']*get_dphidy(terms_ptb['w2_t'], wn_y)
terms_bdg['adv_v'] = - data_tavg['w']*wnode2uvpnode(get_dphidz(terms_ptb['w2_t'], dz))

terms_bdg['adv_h_SGS'] = - data_tavg['u']*get_dphidx(data_tavg['tzz'], wn_x) - data_tavg['v']*get_dphidy(data_tavg['tzz'], wn_y)
terms_bdg['adv_v_SGS'] = - data_tavg['w']*wnode2uvpnode(get_dphidz(data_tavg['tzz'], dz))

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['adv_h_SGS'] + terms_bdg['adv_v_SGS']

#PRODUCTION
terms_bdg['prod_h'] = - 2*(terms_ptb['uw_t']*data_tavg['dwdx'] + terms_ptb['vw_t']*data_tavg['dwdy'])
terms_bdg['prod_v'] = - 2*(terms_ptb['w2_t']*data_tavg['dwdz'])

terms_bdg['prod_h_SGS'] = - 2*(data_tavg['txz']*data_tavg['dwdx'] + data_tavg['tyz']*data_tavg['dwdy'])
terms_bdg['prod_v_SGS'] = - 2*(data_tavg['tzz']*data_tavg['dwdz'])

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v'] + terms_bdg['prod_h_SGS'] + terms_bdg['prod_v_SGS']

#TURBULENT TRANSPORT
terms_ptb['w3_t'] = data_tavg['www'] - 3*data_tavg['w']*data_tavg['ww'] + 2*(data_tavg['w']**3)
terms_ptb['wtzz_t'] = data_tavg['wtzz'] - data_tavg['w']*data_tavg['tzz']

terms_bdg['uturb_v'] = - wnode2uvpnode(get_dphidz(terms_ptb['w3_t'], dz)) - wnode2uvpnode(get_dphidz(terms_ptb['wtzz_t'], dz))

terms_ptb['w2u_t'] = data_tavg['wwu'] - 2*data_tavg['w']*data_tavg['uw']\
  + 2*data_tavg['u']*(data_tavg['w']**2) - data_tavg['u']*data_tavg['ww']
terms_ptb['wtxz_t'] = data_tavg['wtxz'] - data_tavg['w']*data_tavg['txz']
terms_ptb['w2v_t'] = data_tavg['wwv'] - 2*data_tavg['w']*data_tavg['vw']\
  + 2*data_tavg['v']*(data_tavg['w']**2) - data_tavg['v']*data_tavg['ww']
terms_ptb['wtyz_t'] = data_tavg['wtyz'] - data_tavg['w']*data_tavg['tyz']

terms_bdg['uturb_h'] = - get_dphidx(terms_ptb['w2u_t'], wn_x) - get_dphidy(terms_ptb['w2v_t'], wn_y) - get_dphidx(terms_ptb['wtxz_t'], wn_x)\
    - get_dphidy(terms_ptb['wtyz_t'], wn_y)

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

#PRESSURE TRANSPORT
terms_ptb['pw_t'] = data_tavg['pw'] - data_tavg['p']*data_tavg['w']
terms_bdg['pturb_t'] = - 2*(wnode2uvpnode(get_dphidz(terms_ptb['pw_t'],dz)))
terms_ptb['pdwdz_t'] = data_tavg['pdwdz'] - data_tavg['p']*data_tavg['dwdz']
terms_bdg['pturb_s'] = 2*terms_ptb['pdwdz_t']

terms_bdg['ptrans'] = terms_bdg['pturb_t'] + terms_bdg['pturb_s']

#DISSIPATION
# terms_ptb['dvdx2_t'] = data_tavg['dvdx2'] - data_tavg['dvdx']*data_tavg['dvdx']
# terms_ptb['dvdy2_t'] = data_tavg['dvdy2'] - data_tavg['dvdy']*data_tavg['dvdy']
# terms_ptb['dvdz2_t'] = data_tavg['dvdz2'] - data_tavg['dvdz']*data_tavg['dvdz']

# terms_bdg['dissip'] = - 2*(terms_ptb['dvdx2_t'] + terms_ptb['dvdy2_t'] + terms_ptb['dvdz2_t'])

terms_drv['S11'] = data_tavg['dudx']
terms_drv['S12'] = 0.5*(data_tavg['dudy'] + data_tavg['dvdx'])
terms_drv['S13'] = 0.5*(data_tavg['dudz'] + data_tavg['dwdx'])
terms_drv['S22'] = data_tavg['dvdy']
terms_drv['S23'] = 0.5*(data_tavg['dvdz'] + data_tavg['dwdy'])
terms_drv['S33'] = data_tavg['dwdz']

data_tavg['dissip'] = data_tavg['dzz'] + data_tavg['dxz'] + data_tavg['dyz']

terms_bdg['dissip'] = 2*(data_tavg['dissip']\
  - data_tavg['tzz']*terms_drv['S33'] - data_tavg['txz']*terms_drv['S13'] - data_tavg['tyz']*terms_drv['S23'])
    
#CANOPY
terms_bdg['canopy'] = 2*(data_tavg['wfdz']-data_tavg['w']*data_tavg['fdz'])

terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

#RESIDUAL
terms_bdg['res'] = terms_bdg['adv'] + terms_bdg['prod'] + terms_bdg['ttrans'] + terms_bdg['pturb_s'] - terms_bdg['dissip'] - terms_bdg['canopy']
terms_bdg['res_v'] =  terms_bdg['prod_v'] + terms_bdg['uturb_v'] + terms_bdg['pturb_s'] - terms_bdg['dissip'] - terms_bdg['canopy']

# np.save(pathOUT + 'WW_terms.npy', terms_bdg) 

var_bdg = ['adv_h', 'adv_v', 'adv_h_SGS', 'adv_v_SGS', 'uturb_h', 'uturb_v', 'pturb_t', 'pturb_s', 'dissip', 
            'prod_h', 'prod_v','adv','res','ptrans','ttrans','prod','canopy','totdis']

for i in range(len(var_bdg)):
    terms_bdg[var_bdg[i]][(dist < 0)] = float("nan")

#%%Pcolor plots

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy','pturb_s']

i = 0

tmp = terms_bdg[terms[i]][:,:,5:]

yslice = 161
fig,axs = plt.subplots(1,1,figsize=(10,6),tight_layout=True)
pc = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),tmp[:,yslice,:].T,cmap='coolwarm')
axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_c$')
axs.set_title(terms[i])
cbar = plt.colorbar(pc)

#%%Profile plots

z_ax = (np.arange(0,nz-5)*dz+dz/2)/(39/zi)

plt.figure(figsize=(8,8))
plt.plot(np.nanmean(terms_bdg['res'][:,:,5:],axis=(0,1)),z_ax,c='black',marker='o',markevery=8,label='Res')
plt.plot(np.nanmean(terms_bdg['adv'][:,:,5:],axis=(0,1)),z_ax,c='pink',marker='o',markevery=8,label='Adv')
plt.plot(np.nanmean(terms_bdg['prod'][:,:,5:],axis=(0,1)),z_ax,c='red',marker='s',markevery=8,label='Prod')
plt.plot(np.nanmean((terms_bdg['ttrans'] + terms_bdg['pturb_t'])[:,:,5:],axis=(0,1)),z_ax,c='purple',marker='^',markevery=8,label='T_trans')
plt.plot(np.nanmean(terms_bdg['pturb_s'][:,:,5:],axis=(0,1)),z_ax,c='brown',marker='D',markevery=8,label='Return')
plt.plot(np.nanmean(terms_bdg['pturb_t'][:,:,5:],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='P_trans')
# plt.plot(np.nanmean(terms_bdg['dissip'][:,:,5:],axis=(0,1)),z_ax,c='green',marker='v',markevery=8,label='Diss')
# plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='Can')
plt.plot(np.nanmean(-terms_bdg['totdis'][:,:,5:],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='TotDis')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod')
# plt.plot(np.nanmean(terms_bdg['prod']-terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod-Diss')
# plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')
plt.hlines(1,-80,80,linestyle='--',colors='k')
plt.hlines(2,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,50,linestyle='--',colors='grey')
plt.ylim(z_ax[0],z_ax[-1])
plt.xlim(-30,30)
plt.title(r'Vertical Variance')
plt.xlabel(r'$\left \langle duu/dt \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/h_c$')
plt.legend(loc='upper right')

# plt.savefig(pathOUT +'Figures/WW_budget_prof_flat.png',dpi=300,facecolor='white', edgecolor='white')


plt.show()