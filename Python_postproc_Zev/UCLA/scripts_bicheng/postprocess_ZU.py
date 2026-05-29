#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:14:41 2023

This script plots basic statistics such has shear stress, MKE,uavgz, and pcolor 
of u and w  
"""

import lespy as lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

## User-specified Variable
# Case
case = ['amazon_canopy_hill']
lab_case = ('hill')

# File format
fmt_lespath = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}'
fn_budget = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/figures.test.png'

# Topography parameters
zpad = (0.5, 5)
amp = (25,0)
wl = 1000

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 18000
te = 25200
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
#items_bdg = ('prod', 'dissip', 'canopy', 'res', 'adv_h', 'adv_v', 'adv', 'uturb', 'pturb', 'res+uturb', 'sum')
#labs_bdg = (r'$P$', r'$\epsilon$', r'$\epsilon_c$', r'$R$', r'$-A^h_e$',
#  r'$-A^v_e$', r'$A_e$', r'$-T_e$', r'$-\Pi_e$', r'$-R-T_e$', 'sum')
ind_plot = [6, 9]

# Ae cancelation
#cri_divfree = 0.1

## Function
def get_topo(x, wl, amp):
  return (amp*np.cos(2*np.pi/wl*x+np.pi)+amp)

def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = np.complex(0, 1) * wn[np.newaxis, :] * phi_c
  dphidx_c[:, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidx3d(phi, wn):
    # Ensure the input array is 3D
    if len(phi.shape) != 3:
        raise ValueError("The input array must be 3D")

    # Perform the FFT along the last axis
    phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")

    # Compute the derivative using the wavenumbers
    dphidx_c = np.complex(0, 1) * wn[np.newaxis, np.newaxis, :] * phi_c

    # Set the last element to zero
    dphidx_c[:, :, -1] = 0

    # Compute the inverse FFT to get the derivative in real space
    return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1, :] = (phi[1:, :]-phi[:-1, :]) / (dm.dz/dm.zi)
  dphidz[-1, :] = dphidz[-2, :]
  return dphidz

def uvpnode2wnode(phi_uv):
  phi_w = np.zeros(phi_uv.shape)
  phi_w[0, :] = 0
  phi_w[1:, :] = 0.5*(phi_uv[:-1, :]+phi_uv[1:, :])
  return phi_w

def wnode2uvpnode(phi_w):
  phi_uv = np.zeros(phi_w.shape)
  phi_uv[:-1, :] = 0.5*(phi_w[:-1, :]+phi_w[1:, :])
  phi_uv[-1, :] = phi_uv[-2, :]
  return phi_uv




# Plot MKE: For some reason not actually MKE not sure where MKE is outputted 
# =============================================================================
# lespath = fmt_lespath.format(case=case)
# path = lespath + '/output/check_ke.out'
# MKE = np.loadtxt(path, dtype=np.float64)
# fig, ax = plt.subplots(dpi = 100)
# ax.plot(MKE)
# ax.set(title = 'MKE',
#        xlabel = 'timesteps')
# =============================================================================
       
#Load Data 
## Initialization
nca = len(case)
data = [dict.fromkeys(items, None) for ic in range(nca)]

print('Reading Params')
## Read data and calculate the temporal mean
param = [None for ic in range(nca)]
dm = [None for ic in range(nca)]
for ic in range(nca):
  lespath = fmt_lespath.format(case=case[ic])
  param[ic] = lp.lesParam.lesClass.param(lespath)
  dm[ic] = param[ic].domain

print('Reading Data')
for ic in range(nca):
  for item in items:
    print('loading ' + str(item))
    data[ic][item], time = lp.io.io_averFile.loadLES_averFile(param[ic],
      qtype=item, tss=ts, tes=te)
    data[ic][item] = np.mean(data[ic][item], axis=0)

print('Done loading items getting coords')
# %%
## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]

for ic in range(nca):
  z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
  z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
  z_gd[ic] -= zpad[ic]
  Z_gd[ic] = z_gd[ic]-z_tpg[ic]



# %% Plot Shear Stress
z = np.array(z_gd)
x = np.array(x_gd)
z = z[0,:,1]
x = x[0,:,1]

# Reynolds shear stress 
R13 = data[0]['uw']-data[0]['u']*data[0]['w']
R13avgz = np.mean(R13, axis = 1)

#Dispersive (thouhg not really because its already been averaged in y)

D13 = np.mean(data[0]['u']*data[0]['w'], axis=1) \
    - np.mean(data[0]['u'],axis = 1)*np.mean(data[0]['w'], axis=1)
    
txzavgz = np.mean(data[0]['txz'],axis = 1)

shear_stress = R13avgz+D13+txzavgz

# %%
uscale = .4**2
fig, ax = plt.subplots(dpi = 100, figsize= (7.5,10))
ax.plot(-R13avgz*uscale, z, label='R13')
ax.plot(-D13*uscale, z, label='D13')
ax.plot(-txzavgz*uscale, z, label = 'txz')
ax.plot(-shear_stress*uscale,z,label = 'tauxz')
ax.set(title = 'Shear Stress non-dimensional',
       xlabel= 'Tauxz',
       ylabel = 'z')
ax.grid()
ax.set_ylim([0, 550])
ax.legend()

# %%
# Now plot contourfs of u and w 
nz = param[0].domain.nz
dz = param[0].domain.dz

kcut = 100 

data[0]['u_s'] = gaussian_filter(data[0]['u'], sigma=(1, 1), mode='wrap')
data[0]['w_s'] = gaussian_filter(data[0]['w'], sigma=(1, 1), mode='wrap')

fig, (ax0,ax1) = plt.subplots(nrows=2, ncols = 1, dpi = 100, figsize= (10,7.5))

#ax0.contourf([np.squeeze(x_gd[0]),np.squeeze(z_gd[0])], np.squeeze(data[0]['u']) )
cf1 = ax0.contourf(x_gd[0][1:kcut,:],z_gd[0][1:kcut,:], data[0]['u'][1:kcut,:], 30)
ax0.plot(x_gd[0][0, :], z_tpg[0][0, :], ls='-',  c='k')
ax0.set(ylabel='z',
        title = 'u & w vel')
ax.set_ylim([0, kcut*dz])
cbar1 = fig.colorbar(cf1, ax = ax0)

cf2 = ax1.contourf(x_gd[0][1:kcut,:],z_gd[0][1:kcut,:], data[0]['w'][1:kcut,:], 30)
ax1.plot(x_gd[0][0, :], z_tpg[0][0, :], ls='-',  c='k')
ax1.set(ylabel='z',
        xlabel = 'x')
ax.set_ylim([0, kcut*dz])
cbar2 = fig.colorbar(cf2, ax = ax1)
# %%
def build_intf(phi,dz):
    nz,ny,nx = np.shape(phi)
    intf = np.zeros((nx,ny))
    iintf = np.zeros((nx,ny))
    init = 0
    for j in range(0,ny):
        for i in range(0,nx):
            for k in range(0,nz-1):
                if phi[k,j,i]*phi[k+1,j,i] <= 0.0 and init == 0:
                    intf[i,j] = (k-1)*dz-phi[k,j,i]
                    iintf[i,j] = k
                    init = 1
            init = 0
    return intf, iintf


# %%
# intrinsic and superficial averaging
def avgx(var,phi_uv):
    A = phi_uv < 0.0  # logical array for inside interface nodes
    var[A]=0.0 # zero inside interface
    tmp = np.sum(phi_uv>=0.0, axis = 1)
    var_xy = np.sum(var,axis = 1)/tmp  # intrinsic averaging
    var_xy[np.isnan(var_xy)]=0.0  # fix NaN
    return var_xy 
def avgx2(var, phi_uv, dm):
    nm_outside = np.zeros(dm.nz)
    nm_sum = np.zeros(dm.nz)
    for k in range(dm.nz):
        for i in range(dm.nx):
            if phi_uv[k,i] > 0.0:
                nm_outside[k] += 1
                nm_sum[k] += var[k,i]
    var_xy = nm_sum/nm_outside 
    var_xy[np.isnan(var_xy)]=0.0
    return var_xy

# %% This section of codes requires that I've already run paperraw_crossAeEstHill
#    .py or another similar script 
# Calculates Shear stress budget with instrinsic and superficial averaging 

# First Calculate the Dispersive Flux 

ibm = lp.domain.dmClass.immersedbdy(param[0])
phi_2d = np.mean(ibm.phi_uv,axis = 1)
intf, iintf = build_intf(ibm.phi_uv,dm[0].dz)
# %%
# Reynolds shear stress 
R13 = data[0]['uw']-data[0]['u']*data[0]['w']
Rt1 = avgx2(data[0]['u']*data[0]['w'], phi_2d,dm[0])
Rt2 = avgx2(data[0]['uw'], phi_2d,dm[0])
R13avgz = avgx(R13, phi_2d)
#R13avgz = np.mean(R13, axis = 1)

#Dispersive (thouhg not really because its already been averaged in y)

D13 = avgx(data[0]['u']*data[0]['w'], phi_2d) \
    - avgx(data[0]['u'],phi_2d)*avgx(data[0]['w'], phi_2d)
    
txzavgz = avgx(data[0]['txz'],phi_2d)

shear_stress = R13avgz+D13+txzavgz

# %%
z = np.array(z_gd)
x = np.array(x_gd)
z = z[0][:,1]
x = x[0][:,1]

fig, ax = plt.subplots(dpi = 100, figsize= (7.5,10))
ax.plot(-R13avgz, z, label='R13')
ax.plot(-D13, z, label='D13')
ax.plot(-txzavgz, z, label = 'txz')
ax.plot(-shear_stress,z,label = 'tauxz')
ax.set(title = 'Shear Stress non-dimensional',
       xlabel= 'Tauxz',
       ylabel = 'z')
ax.grid()
ax.set_ylim([0, 580])
ax.legend()



 