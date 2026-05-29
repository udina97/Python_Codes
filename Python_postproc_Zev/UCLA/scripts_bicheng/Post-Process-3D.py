#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:01:27 2023

@author: Zev Underwood
"""

import numpy as np
import lespy as lp
import os.path
import time
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.colors as colors
from scipy.fft import fftn, ifftn
from scipy.interpolate import RegularGridInterpolator
import useful_functions as uf
from scipy.interpolate import interpn

## User-specified Variables
case = 'amazon_canopy_hill'
# File
fmt_lespath = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}'
#fmt_data = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_hill_3D/'
fn2 = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_hill05h_3D/'    

tt_all = (180100, 252000+1, 100)
ts = 18000
te = 25200
print('#'*80)
print('   Reading the setup parameters of the simulation...')
path_les = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(path_les)
dm = param.domain
dm.show()

# Define the variables and coordinates
print('   Initializing the data reading...')
ntt = len(range(*tt_all))
z, y, x = lp.domain.dmFun.xyzcoord(dm)
ibm = lp.domain.dmClass.immersedbdy(param)
z_tpg = ibm.z_phi0 * dm.zi
Z_gd = z-z_tpg
y_tpg, x_tpg = lp.domain.dmFun.xycoord(dm)

nca = 1
hc = 39.0
amp = 'hill'


# %%

var_list2 = ['u','v','w','p','u2','v2','w2',\
'uv','uw','vw','p2','pu','pv','pw','pdudx','pdvdy','pdwdz','txx','txy','txz','tyy','tyz',\
'tzz','u3','v3','w3','u2v','u2w','uv2','v2w','uw2','vw2','uvw','utxx','utxy','utxz',\
'vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','dissip','Fcx','Fcy','Fcz','uFcx','vFcy','wFcz']

# Load 3D data
data = {}
for var in var_list2:
    data[var] = np.load(fn2 + var + '.npy')
   # data[var] = gaussian_filter(data[var], sigma=(0, 2 ,2), mode='wrap')
 # %% Test terrain following average
intf, iintf = uf.build_intf(ibm.phi_uv, dm.dz)
zt = np.arange(np.min(intf), dm.lz - np.max(intf), dm.dz)
# %%
# t = time.time()
# uavg1 = uf.tavg_xy(data['uw'], intf.T, dm.dz, dm.lz)
# elapsed = time.time()-t
#%%
# def get_vprof2(term, Z_gd, amp):
#   if amp==0:
#     prof = term.mean(axis=(1,2))
#     return prof
#   else:
#     term2d = np.mean(term,axis = 1)
#     Z_gd2d = np.mean(Z_gd,axis = 1)
#     Z_prof = np.squeeze(Z_gd2d[:,0])
#     prof = term2d[:, 0] / term2d.shape[-1]
#     for ix in range(1, term2d.shape[-1]):
#             prof += np.interp(Z_prof, Z_gd2d[:, ix], term2d[:, ix]) / term2d.shape[-1]
#     return prof

# t2 = time.time()
# uavg2 = get_vprof2(data['uw'], Z_gd, 50)
# elapsed2 = time.time()-t2
# %%
# t3 = time.time()
# uavg3 = uf.tavg_xy2(data['uw'], intf.T, dm.dz, dm.lz)
# elapsed2 = time.time()-t3
# # %%
# zv= np.arange(0,dm.lz,dm.dz)
# plt.figure()
# #plt.plot(uavg1,zt,label = '1')
# plt.plot(uavg3,zt+min(zt),label = '2')
# plt.plot(uavg2,zv,label = '3')
# plt.legend()
# plt.show()
#%%
items = ['Fcx', 'Fcz', 'Fcy', 'uFcx','wFcz', 'vFcy', 'dissip']
data_2d = {}
for item in items:
  print('loading ' + str(item))
  data_2d[item], time = lp.io.io_averFile.loadLES_averFile(param,
    qtype=item, tss=ts, tes=te)
  data_2d[item] = np.mean(data_2d[item], axis=0)
    
# %%
# Now plot contourfs of u and w 
nz = param.domain.nz
dz = param.domain.dz

kcut = 100 

data['u_s'] = gaussian_filter(np.mean(data['u'], axis = 1), sigma=(1, 1), mode='wrap')
data['w_s'] = gaussian_filter(np.mean(data['w'], axis = 1), sigma=(1, 1), mode='wrap')

fig, (ax0,ax1) = plt.subplots(nrows=2, ncols = 1, dpi = 100, figsize= (10,7.5))

#ax0.contourf([np.squeeze(x_gd[0]),np.squeeze(z_gd[0])], np.squeeze(data[0]['u']) )
cf1 = ax0.contourf(x[1:kcut,1,:],z[1:kcut,1,:], data['u_s'][1:kcut,:], 30)
ax0.plot(np.squeeze(x[0,1, :]), z_tpg[0, :], ls='-',  c='k')
ax0.set(ylabel='z',
        title = 'u & w vel')
ax0.set_ylim([0, kcut*dz])
cbar = fig.colorbar(cf1, ax = ax0)

cf2 = ax1.contourf(x[1:kcut,1,:],z[1:kcut,1,:], data['w_s'][1:kcut,:], 30)
ax1.plot(np.squeeze(x[0,1, :]), z_tpg[0, :], ls='-',  c='k')
ax1.set(ylabel='z',
        xlabel = 'x')
ax1.set_ylim([0, kcut*dz])
cbar2 = fig.colorbar(cf2, ax = ax1)

# %%
# Now calculate Shear and Dispersive Stress

def avgxy(fluid_vars, Z_gd):
    # Get boolean mask for nodes above surface
    #above_surface_mask = phi >= 0.0
    
    # Set nodes below surface to NaN to exclude them from averaging
    #tmp[~above_surface_mask] = np.nan
    mask = Z_gd<=0
    tmp = np.ma.array(fluid_vars, mask=mask)
    # Average along x and y dimensions, ignoring NaN values
    avg_vars = np.nanmean(tmp, axis=(1, 2))
    avg_vars[np.isnan(avg_vars)]=0.0
    return avg_vars

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
  phi_w[0,:,:] = 0
  phi_w[1:,:,:] = 0.5*(phi_uv[:-1,:,:]+phi_uv[1:,:,:])
  return phi_w

def wnode2uvpnode(phi_w):
  phi_uv = np.zeros(phi_w.shape)
  phi_uv[:-1, :] = 0.5*(phi_w[:-1, :]+phi_w[1:, :])
  phi_uv[-1, :] = phi_uv[-2, :]
  return phi_uv

def get_vprof(term, Z_gd, amp):
  if amp=='flat':
    prof = term.mean(axis=(1,2))
    return prof
  elif amp == 'hill':
    term2d = np.mean(term,axis = 1)
    Z_gd2d = np.mean(Z_gd,axis = 1)
    Z_prof = np.squeeze(Z_gd2d[:,0])
    prof = term2d[:, 0] / term2d.shape[-1]
    for ix in range(1, term2d.shape[-1]):
            prof += np.interp(Z_prof, Z_gd2d[:, ix], term2d[:, ix]) / term2d.shape[-1]
    return prof

  elif amp == 'real':
    #flat_index =  np.argmin(Z_gd)
    #j, i = np.unravel_index(flat_index, Z_gd.shape)
    Z_prof = np.squeeze(Z_gd[:,91,31])
    prof = np.zeros(term.shape[0])
    for ix in range(0, term.shape[-1]):
        for jy in range(0,term.shape[-2]):
            prof += np.interp(Z_prof, Z_gd[:,jy, ix], term[:,jy, ix]) / (term.shape[-1]*term.shape[-2])              
    return prof





# %%
phi = ibm.phi_uv


# %%
u_scale = .4
# Reynolds shear stress 
R13 = data['uw']-data['u']*data['w']
# Rt1 = avgx(data[0]['u']*data[0]['w'], phi)
# Rt2 = avgx(data[0]['uw'], phi)
#R13avgz = avgxy(R13, phi)
R13avgz = np.mean(R13, axis = (1,2))
#R13avgz = avgxy(R13, Z_gd)


#Dispersive 

D13 = avgxy(data['u']*data['w'], phi) \
    - avgxy(data['u'],phi)*avgxy(data['w'], phi)
    

D13 = np.mean(data['u']*data['w'], axis = (1,2)) \
    - np.mean(data['u'], axis = (1,2))*np.mean(data['w'], axis = (1,2))
    
# D13_3 = uf.average_isocontours(data['u']*data['w'], intf, dm.dz,dm.lz)\
#       - uf.average_isocontours(data['u'], intf, dm.dz,dm.lz)\
#       * uf.average_isocontours(data['w'], intf, dm.dz,dm.lz)
    
txzavgz = avgxy(data['txz'],phi)

shear_stress = R13avgz+D13+txzavgz

z1 = z[:,1,1]
x1 = x[1,1,:]

fig, ax = plt.subplots(dpi = 100, figsize= (7.5,10))
ax.plot(-R13avgz*u_scale**2, z1, label='R13')
#ax.plot(-R13avgz2, z1, label='R13-2')
ax.plot(-D13*u_scale**2, z1, label='D13')
#ax.plot(-D13_2, z1, label='D13-2')
ax.plot(-txzavgz*u_scale**2, z1, label = 'txz')
ax.plot(-shear_stress*u_scale**2,z1,label = 'tauxz')
ax.set(title = 'Shear Stress non-dimensional',
       xlabel= 'Tauxz',
       ylabel = 'z')
ax.grid()
ax.set_ylim([0, 600])
ax.axhline(np.max(z_tpg), linestyle = '--', color = 'k', alpha = .5)
ax.axhline(np.max(z_tpg)+hc, linestyle = '--', color = 'k', alpha = .5)
ax.legend()
# %% Now Calculate the Dispersive Flux Contribution Percent 

R13Int = np.trapz(-R13avgz, x = z1, dx= dm.dz)
D13Int = np.trapz(-D13, x = z1, dx= dm.dz)
txzInt = np.trapz(-txzavgz, x = z1, dx= dm.dz )

D13perc = D13Int/(D13Int+R13Int+txzInt)
print('Dispersive Flux Contribution ' + str(D13perc*100))

# %% Calculate MKE Budget 
wn = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
## Horizontal Contribution 
termsx_mke_list = ['tt_x', 'sgst_x', 'disst_x','dissd_x', 'sgsd_x','adv_x','press_x']
termsx_mke = {}

## Vertical Contribution 
termsz_mke_list = ['tt_z', 'sgst_z', 'disst_z','dissd_z', 'sgsd_z','adv_z','press_z']
# termsx_mke_lab = ['$-\frac{\partial}{partial x} \langle \overline{u^{\prime}u^{\prime}} \
#                   \rangle$', '$-\frac{\partial}{partial x} \langle \overline{u^{''}u^{''}} \
#                   \rangle$','$-\frac{\partial}{partial x} \langle \tau_{xx}u \rangle$', \
#                   '$-\frac{\partial u}{partial x} \langle \overline{u^{\prime}u^{\prime}} \
#                   \rangle$', '$-\frac{\partial u}{partial x} \langle \overline{u^{''}u^{''}} \
#                   \rangle$', '$-\frac{\partial u}{partial x} \langle \tau_{xx}\rangle$' \
#                   , '$- 1/3 \frac{\partial \langle u \rangle}{partial x}$', \
#                     '$\langle u \rangle \frac{\partial \langle p^* \rangle}{partial x}']
termsz_mke = {}

# MKE Transport Due to Turbulence 
#x component 
#avgxy2(fluid_vars, phi)
uu_varu = (data['u2']-data['u']*data['u'])*data['u']
_,_,duu_varudx  = np.gradient(uu_varu,dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
duu_varudx2 = uf.dvardx(uu_varu, dm.dx/dm.zi)

tt_x = avgxy(duu_varudx,phi)
#t_x = -np.mean(duu_varudx, axis= (1,2))
#tt_x2 = -np.mean(duu_varudx2, axis= (1,2))
termsx_mke['tt_x'] = -tt_x

# z component 
uw_varu = (data['uw']-data['u']*data['w'])*data['u']
duw_varudz,_,_ = np.gradient(uw_varu, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)

tt_z = avgxy(duw_varudz, phi)
#tt_z = -np.mean(duw_varudz, axis= (1,2))
termsz_mke['tt_z'] = -tt_z


# MKE Transport Due to Dispersive Flux  
# x component 
#avgxy2(fluid_vars, phi)
# uu_disp = (avgxy(data['u']*data['u'], phi) \
#     - avgxy(data['u'],phi)*avgxy(data['w'], phi))*data['u']
# duu_dispudx,_,_  = np.gradient(uu_varu,dm.dx,dm.dy,dm.dz)

# dt_x = avgxy(duu_varudx,phi)

# z component 

uw_dispu = (avgxy(data['u']*data['w'], phi) \
    - avgxy(data['u'],phi)*avgxy(data['w'], phi))\
    *avgxy(data['u'], phi)
#uw_dispu = (np.mean(data['u']*data['w'], axis = (1,2)) \
#    - np.mean(data['u'], axis = (1,2))*np.mean(data['w'], axis = (1,2)))\
#   * np.mean(data['u'], axis = (1,2))

dt_z = -np.gradient(uw_dispu, dm.dz/dm.zi)
termsz_mke['dt_z'] = dt_z

#  SGS Transport of MKE 

# x component 
utxx = data['u']*data['txx']
_,_,dutxxdx = np.gradient(utxx,dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)

sgst_x = avgxy(dutxxdx,phi)
#sgst_x = -np.mean(dutxxdx, axis= (1,2))
termsx_mke['sgst_x'] = -sgst_x

# z component 
utxz = data['u']*data['txz']
dutxzdz,_,_ = np.gradient(utxz, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)

sgst_z = avgxy(dutxzdz,phi)
#sgst_z = -np.mean(dutxzdz,axis = (1,2))

termsz_mke['sgst_z'] = -sgst_z
# z component test
# utxza = np.mean(data['u'], axis= (1,2))*np.mean(data['txz'],axis = (1,2))
# sgst_za = -np.gradient(utxza, dm.dz/dm.zi)
#   MKE dissipation from turbulence 
# x component 
_,_,dudx = np.gradient(data['u'],dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
dudxa = avgxy(dudx,phi)
#dudxa = np.mean(dudx, axis= (1,2))

uu_vara = avgxy(data['u2']-data['u']*data['u'],phi)
#uu_vara = np.mean(data['u2']-data['u']*data['u'], axis= (1,2))

disst_x = dudxa*uu_vara
termsx_mke['disst_x'] = disst_x

# z compoent 

dudz,_,_ = np.gradient(data['u'],dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
dudza = avgxy(dudz,phi)
#dudza = np.mean(dudz, axis= (1,2))

uw_vara = avgxy(data['uw']-data['u']*data['w'],phi)
#uw_vara = np.mean(data['uw']-data['u']*data['w'], axis= (1,2))
disst_z = dudza*uw_vara

termsz_mke['disst_z'] = disst_z
#  MKE dissipation from dispersive flux 

# x component 

# dudx,_,_ = np.gradient(data['u'],dm.dz,dm.dy,dm.dx)
# #dudxa = avgxy(dudx,phi)
# dudxa = np.mean(dudx, axis= (1,2))

uu_disp = (avgxy(data['u']*data['u'], phi) \
     - avgxy(data['u'],phi)*avgxy(data['u'], phi))

#uu_disp = (np.mean(data['u']*data['u'], axis = (1,2)) \
#     - np.mean(data['u'],axis = (1,2))*np.mean(data['u'], axis= (1,2)))

dissd_x = dudxa*uu_disp 
termsx_mke['dissd_x'] = dissd_x

# z componet 

uw_disp = (avgxy(data['u']*data['w'], phi) \
     - avgxy(data['u'],phi)*avgxy(data['w'], phi))

#uw_disp = (np.mean(data['u']*data['w'], axis = (1,2)) \
#     - np.mean(data['u'],axis = (1,2))*np.mean(data['w'], axis= (1,2)))

dissd_z = dudza*uw_disp 
termsz_mke['dissd_z'] = dissd_z
#  MKE dissipation from sgs flux 

# x component 
txxa = avgxy(data['txx'], phi)
#txxa = np.mean(data['txx'], axis = (1,2))
sgsd_x = dudxa*txxa
termsx_mke['sgsd_x'] = sgsd_x

# z component 
txza = avgxy(data['txz'], phi)
#txza = np.mean(data['txz'], axis = (1,2))
sgsd_z = dudza*txza
termsz_mke['sgsd_z'] = sgsd_z

#  Advection of MKE 

_,_,du3dx = np.gradient(data['u3'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)

du3dxa = avgxy(du3dx,phi)
#du3dxa = np.mean(du3dx, axis = (1,2))
adv_x = -1/3*du3dxa
termsx_mke['adv_x'] = adv_x
#  Mean Pressure Contribution 

# x component 

_,_,dpdx = np.gradient(data['p'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
press_x =  avgxy(dpdx,phi)*avgxy(data['u'],phi)
#press_x =  np.mean(dpdx,axis = (1,2))*np.mean(data['u'],axis = (1,2))
termsx_mke['press_x'] = press_x

# z component 

dpdz,_,_ = np.gradient(data['p'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
press_z =  avgxy(dpdz,phi)*avgxy(data['u'],phi)
#press_z =  np.mean(dpdx,axis = (1,2))*np.mean(data['w'],axis = (1,2))
termsz_mke['press_z'] = press_z

#  Pressure Gradient Contribution 

# x component 
# dpdx = np.mean(np.ones(data['u'].shape), axis = (1,2))
# uavgz = avgxy(data['u'], phi)
# #uavgz = np.mean(data['u'], axis = (1,2))

# Fpx = dpdx*uavgz
# termsx_mke['Fpx'] = Fpx

# %% Plot MKE x component contribution 
# fig, ax = plt.subplots(dpi = 100, figsize= (7.5,10))
# it = 0
# for key in termsx_mke.keys():
#     ax.plot(termsx_mke[key]*param.flow.uScal**3/dm.zi, z1, label = key, \
#             linewidth = 2.0)
#     it += 1

# ax.legend()
# ax.set(xlabel = 'MKEx',
#        ylabel = 'z')
# ax.grid()
# ax.axhline(np.max(z_tpg), linestyle = '--', color = 'k', alpha = .5)
# ax.axhline(np.max(z_tpg)+hc, linestyle = '--', color = 'k', alpha = .5)

# fig, ax = plt.subplots(dpi = 100, figsize= (7.5,10))

# for key in termsz_mke.keys():
#     ax.plot(termsz_mke[key]*param.flow.uScal**3/dm.zi, z1, label = key, \
#             linewidth = 2.0)

# ax.legend()
# ax.set(xlabel = '$MKE_z$',
#        ylabel = 'z')
# ax.grid()
# ax.axhline(np.max(z_tpg), linestyle = '--', color = 'k', alpha = .5)
# ax.axhline(np.max(z_tpg)+hc, linestyle = '--', color = 'k', alpha = .5)

# %%  3D TKE Budget  ##########################################################
terms_bdg = {}
terms_bdg2 = {}
# Gradient function
def gradient(array, dx, dy, dz):
    return np.gradient(array, dx, dy, dz, axis=(0, 1, 2))

######################### Production term
#data['dwdx'] = uf.dvardx_f(data['w'], dm.dx/dm.zi)
_,_,data['dwdx'] = np.gradient(data['w'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dwdx'] = gaussian_filter(data['dwdx'], sigma=(1, 1 ,1), mode='wrap')

#data['dudx'] = uf.dvardx_f(data['u'], dm.dx/dm.zi)
_,_,data['dudx'] = np.gradient(data['u'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dudx'][1:, :, :] = 0.5 * (data['dudx'][:-1, :, :] + data['dudx'][1:, : :])
data['dudx'][0, : :] = 0
data['dudx'] = gaussian_filter(data['dudx'], sigma=(1, 1 , 1), mode='wrap')

data['dudz'] = np.zeros(data['u'].shape) 
data['dudz'][1:, : :] = (data['u'][1:,:, :] - data['u'][:-1,:, :]) / (dm.dz / dm.zi)
data['dudz'] = gaussian_filter(data['dudz'], sigma=(1, 1, 1), mode='wrap')

data['dwdz'] = np.zeros(data['w'].shape) 
data['dwdz'][1:-1,:, :] = (data['w'][2:,:, :] - data['w'][:-2,:, :]) / (2 * dm.dz / dm.zi)
data['dwdz'] = gaussian_filter(data['dwdz'], sigma=(1, 1, 1), mode='wrap')

data['uw_t'] = data['uw'] - data['u'] * data['w']
data['uw_t'] = gaussian_filter(data['uw_t'], sigma=(1, 1, 1), mode='wrap')
data['uu_t'] = data['u2'] - data['u']**2
data['uu_t'] = gaussian_filter(data['uu_t'], sigma=(1, 1, 1), mode='wrap')
data['vv_t'] = data['v2'] - data['v']**2
data['vv_t'] = gaussian_filter(data['vv_t'], sigma=(1, 1, 1), mode='wrap')
data['ww_t'] = data['w2'] - data['w']**2
data['ww_t'] = gaussian_filter(data['ww_t'], sigma=(1, 1, 1), mode='wrap')
data['uv_t'] = data['uv']  - data['u']*data['v']
data['vw_t'] = data['vw']  - data['v']*data['w']

# ####################### Wake Production term #################################
# %%calc avg for dispersive terms
u_xy=get_vprof(data['u'], Z_gd,amp)
v_xy=get_vprof(data['v'], Z_gd,amp)
w_xy=get_vprof(data['w'], Z_gd,amp)
uu_xy=get_vprof(data['uu_t'], Z_gd,amp)
vv_xy=get_vprof(data['vv_t'], Z_gd,amp)
ww_xy=get_vprof(data['ww_t'], Z_gd,amp)
uv_xy=get_vprof(data['uv_t'], Z_gd,amp)
uw_xy=get_vprof(data['uw_t'], Z_gd,amp)
vw_xy=get_vprof(data['vw_t'], Z_gd,amp)
dudx_xy=get_vprof(dudx, Z_gd,amp)

#data['dvdx'] = uf.dvardx_f(data['v'], dm.dx/dm.zi)
_,_,data['dvdx'] = np.gradient(data['v'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dvdx'][1:, :, :] = 0.5 * (data['dvdx'][:-1, :, :] + data['dvdx'][1:, : :])
data['dvdx'][0, : :] = 0

#data['dwdx'] = uf.dvardx_f(data['w'], dm.dx/dm.zi)
data['dwdx'][1:, :, :] = 0.5 * (data['dwdx'][:-1, :, :] + data['dwdx'][1:, : :])
data['dwdx'][0, : :] = 0

dvdx_xy=get_vprof(data['dvdx'], Z_gd,amp)
dwdx_xy=get_vprof(data['dwdx'], Z_gd,amp)

#data['dudy'] = uf.dvardy_f(data['u'], dm.dx/dm.zi)
_,data['dudy'],_ = np.gradient(data['u'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dudy'][1:, :, :] = 0.5 * (data['dudx'][:-1, :, :] + data['dudx'][1:, : :])
data['dudy'][0, : :] = 0

#data['dvdy'] = uf.dvardy_f(data['v'], dm.dx/dm.zi)
_,data['dvdy'],_ = np.gradient(data['v'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dvdy'][1:, :, :] = 0.5 * (data['dvdx'][:-1, :, :] + data['dvdx'][1:, : :])
data['dvdy'][0, : :] = 0

#data['dwdy'] = uf.dvardx_f(data['w'], dm.dx/dm.zi)
_,data['dwdy'],_ = np.gradient(data['u'], dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
data['dwdy'][1:, :, :] = 0.5 * (data['dwdx'][:-1, :, :] + data['dwdx'][1:, : :])
data['dwdy'][0, : :] = 0

dudy_xy=get_vprof(data['dudy'], Z_gd,amp)
dvdy_xy=get_vprof(data['dvdy'], Z_gd,amp)
dwdy_xy=get_vprof(data['dwdy'], Z_gd,amp)
dudz_xy=get_vprof(data['dudz'], Z_gd,amp)

data['dvdz'] = np.zeros(data['v'].shape) 
data['dvdz'][1:-1,:, :] = (data['v'][2:,:, :] - data['v'][:-2,:, :]) / (2 * dm.dz / dm.zi)

dvdz_xy=get_vprof(data['dvdz'], Z_gd,amp)
dwdz_xy=get_vprof(data['dwdz'], Z_gd,amp)

ud = np.zeros_like(data['u'])
vd = np.zeros_like(data['u'])
wd = np.zeros_like(data['u'])
uud = np.zeros_like(data['u'])
vvd = np.zeros_like(data['u'])
uvd = np.zeros_like(data['u'])
wwd = np.zeros_like(data['u'])
uwd = np.zeros_like(data['u'])
vwd = np.zeros_like(data['u'])
duddx = np.zeros_like(data['u'])
dvddx = np.zeros_like(data['u'])
dwddx = np.zeros_like(data['u'])
duddy = np.zeros_like(data['u'])
dvddy = np.zeros_like(data['u'])
dwddy = np.zeros_like(data['u'])
duddz = np.zeros_like(data['u'])
dvddz = np.zeros_like(data['u'])
dwddz = np.zeros_like(data['u'])


for i in range(dm.nz):
    ud[i, :, :] = data['u'][i, :, :] - u_xy[i]
    vd[i, :, :] = data['v'][i, :, :] - v_xy[i]
    wd[i, :, :] = data['v'][i, :, :] - w_xy[i]
    uud[i, :, :] = data['uu_t'][i, :, :] - uu_xy[i]
    vvd[i, :, :] = data['vv_t'][i, :, :] - vv_xy[i]
    uvd[i, :, :] = data['uv_t'][i, :, :] - uv_xy[i]
    wwd[i, :, :] = data['ww_t'][i, :, :] - ww_xy[i]
    uwd[i, :, :] = data['uw_t'][i, :, :] - uw_xy[i]
    vwd[i, :, :] = data['vw_t'][i, :, :] - vw_xy[i]
    duddx[i, :, :] = data['dudx'][i, :, :] - dudx_xy[i]
    dvddx[i, :, :] = data['dvdx'][i, :, :] - dvdx_xy[i]
    dwddx[i, :, :] = data['dwdx'][i, :, :] - dwdx_xy[i]
    duddy[i, :, :] = data['dudy'][i, :, :] - dudy_xy[i]
    dvddy[i, :, :] = data['dvdy'][i, :, :] - dvdy_xy[i]
    dwddy[i, :, :] = data['dwdy'][i, :, :] - dwdy_xy[i]
    duddz[i, :, :] = data['dudz'][i, :, :] - dudz_xy[i]
    dvddz[i, :, :] = data['dvdz'][i, :, :] - dvdz_xy[i]
    dwddz[i, :, :] = data['dwdz'][i, :, :] - dwdz_xy[i]
    
    spd = -uud*duddx-vvd*dvddy-wwd*dwddz-uwd*(duddz+dwddx)-vwd*(dvddz+dwddy)-uvd*(duddy+dvddx);
    terms_bdg2['disp_prod'] = get_vprof(spd, Z_gd,amp)
terms_bdg['disp_prod'] = spd
# %% 
#################################### TKE
tke = (data['uu_t'] + data['vv_t'] + data['ww_t']) / 2
tkeavg = get_vprof(tke, Z_gd,amp)
#################################### Production
terms_bdg['prod'] = -data['uu_t'] * data['dudx']\
    - data['ww_t'] * data['dwdz']\
    - data['uw_t'] * (data['dudz'] + data['dwdx'])

## Shear Production Finnigan Procedure
#uw_varz = avgxy(data['uw_t'],Z_gd)
#ww_varz = avgxy(data['ww_t'],Z_gd)
#vw_varz = avgxy(data['vw_t'],Z_gd)
#u_z = avgxy(data['u'],Z_gd)
#w_z = avgxy(data['w'],Z_gd)
dudz = np.gradient(u_xy,dm.dz/dm.zi)
dwdz = np.gradient(w_xy,dm.dz/dm.zi)
dvdz = np.gradient(v_xy,dm.dz/dm.zi)

terms_bdg2['prod'] = -uw_xy*dudz-ww_xy*dwdz-vw_xy*dvdz



#  ################################### Advection term
ue = data['u']*tke
#duedx = uf.dvardx_f(ue, dm.dx)
wn = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
#_,_,duedx = np.gradient(ue,dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
duedx = get_dphidx3d(ue, wn)
tkez = np.zeros(tke.shape)
tkez[0,:,:] = 0
tkez[1:, :, :] = 0.5*(tke[:-1,:,:] + tke[1:,:,:])
we = data['w']*tkez
dwedz = uf.dvardz(we, dm.dz/dm.zi)
terms_bdg['adv_h'] = -duedx
terms_bdg['adv_v'] = dwedz
terms_bdg['adv'] = -duedx-dwedz

terms_bdg2['adv'] = -np.gradient(w_xy*tkeavg,dm.dz/dm.zi)



# %%
#################################### Turbulent transport term
turb_u3 = data['u3'] - 3 * data['u'] * data['u2'] + 2 * data['u']**3
turb_uv2 = data['uv2'] - 2 * data['v'] * data['uv'] + 2 * data['u'] * data['v']**2 - data['u'] * data['v2']
uw = wnode2uvpnode(data['uw'])
w_c = wnode2uvpnode(data['w'])
w2_c = wnode2uvpnode(data['w2'])
turb_uw2 = data['uw2'] - 2 * w_c * uw + 2 * data['u'] * w_c**2 - data['u'] * w2_c
ue = 0.5 * (turb_u3 + turb_uv2 + turb_uw2)
#duedx = uf.dvardx_f(ue, dm.dx/dm.dz)
_,_,duedx = np.gradient(ue, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
u_w = uvpnode2wnode(data['u'])
u2_w = uvpnode2wnode(data['u2'])
turb_wu2 = data['u2w'] - 2 * u_w * data['uw'] + 2 * data['w'] * u_w**2 - data['w'] * u2_w
v_w = uvpnode2wnode(data['v'])
v2_w = uvpnode2wnode(data['v2'])
turb_wv2 = data['v2w'] - 2 * v_w * data['vw'] + 2 * data['w'] * v_w**2 - data['w'] * v2_w
turb_w3 = data['w3'] - 3 * data['w'] * data['w2'] + 2 * data['w']**3
we = 0.5 * (turb_wu2 + turb_wv2 + turb_w3)
weavg = get_vprof(we, Z_gd,amp)
dwedz,_,_ = np.gradient(we, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
dwedzavg = np.gradient(weavg, dm.dz/dm.zi)
terms_bdg2['uturb'] = -dwedzavg
terms_bdg['uturb'] = -duedx - dwedz
##################################### SGS transport added to Turbulent 
utxx = data['utxx'] - data['u'] * data['txx']
vtxy = data['vtxy'] - data['v'] * data['txy']
wtxz = data['wtxz'] - data['w'] * data['txz']
wtxz_uv = wnode2uvpnode(wtxz)
#dutaudx = uf.dvardx_f(utxx + vtxy + wtxz_uv, dm.dx/dm.zi)
_,_,dutaudx = np.gradient(utxx+vtxy+wtxz, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
utxz = data['utxz'] - u_w * data['txz']
vtyz = data['vtyz'] - v_w * data['tyz']
wtzz = data['wtzz'] - data['w'] * data['tzz']
dutaudz = uf.dvardz(utxz + vtyz + wtzz, dm.dz/dm.zi)
dutaudzavg = np.gradient(get_vprof(utxz + vtyz + wtzz,Z_gd,amp), dm.dz/dm.zi)
terms_bdg['uturb'] -= dutaudx + dutaudz
terms_bdg2['uturb'] -= dutaudzavg

#################################### Dispersive Transport

ttdx=uud*ud+vvd*ud+wwd*ud;
ttdy=uud*vd+vvd*vd+wwd*vd;
ttdz=uud*wd+vvd*wd+wwd*wd;
#ttdxx = uf.dvardx_f(ttdx,dm.dx/dm.zi)
_,_,ttdxx = np.gradient(ttdx, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
#ttdyy = uf.dvardy_f(ttdy,dm.dy/dm.zi)
_,ttdyy,_ = np.gradient(ttdy, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
ttdzz = uf.dvardz(ttdz,dm.dz/dm.zi)
ttdzzavg = np.gradient(get_vprof(ttdz, Z_gd,amp),dm.dz/dm.zi)
dft = -.5*(ttdxx+ttdyy+ttdzz);

terms_bdg['disptrans'] = dft
terms_bdg2['disptrans'] = -.5*ttdzzavg #Here check neg sign 

#################################### Pressure transport term
pu = data['pu'] - data['p'] * data['u']
#dpudx = uf.dvardx_f(pu, dm.dx/dm.zi)
_,_,dpudx = np.gradient(pu, dm.dz/dm.zi,dm.dy/dm.zi,dm.dx/dm.zi)
p_w = uvpnode2wnode(data['p'])
pw = data['pw'] - p_w * data['w']
dpwdz = uf.dvardz(pw, dm.dz/dm.zi)
dpwdzavg = np.gradient(get_vprof(pw, Z_gd,amp), dm.dz/dm.zi)
terms_bdg['pturb'] = -dpudx - dpwdz
terms_bdg2['pturb'] = -dpwdzavg

#################################### Dissipation
terms_bdg['canopy'] = data_2d['wFcz'] - np.mean(data['w'],axis=1) * data_2d['Fcz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data_2d['uFcx'] - np.mean(data['u'],axis=1) * data_2d['Fcx']) + (data_2d['vFcy'] - np.mean(data['v'],axis=1) * data_2d['Fcy'])
terms_bdg['canopy'] = np.repeat(terms_bdg['canopy'].reshape([dm.nz,1,dm.nx]), repeats = 160, axis = 1)
terms_bdg2['canopy'] = get_vprof(gaussian_filter(terms_bdg['canopy'],[1,2,2]), Z_gd,amp)
terms_bdg['dissip'] = np.repeat(data_2d['dissip'].reshape([dm.nz,1,dm.nx]), repeats = 160, axis = 1)
terms_bdg2['dissip'] = get_vprof(terms_bdg['dissip'], Z_gd,amp)


# %%

##################################### Residual term
terms_bdg['res_lhs'] = terms_bdg['prod'] + terms_bdg['dissip'] + terms_bdg['canopy']
terms_bdg['res_rhs'] = -terms_bdg['adv'] - terms_bdg['pturb'] - terms_bdg['uturb']

##################################### R-T_e term
terms_bdg['res+uturb'] = terms_bdg['res_lhs'] + terms_bdg['uturb']

##################################### Sum of above terms
terms_bdg['sum'] = terms_bdg['prod'] + terms_bdg['dissip'] + terms_bdg['canopy'] + terms_bdg['adv'] + terms_bdg['uturb'] + terms_bdg['pturb']
terms_bdg['sum_disp'] = terms_bdg['disp_prod']+terms_bdg['disptrans']+terms_bdg['prod'] + terms_bdg['dissip'] + terms_bdg['canopy'] + terms_bdg['adv'] + terms_bdg['uturb'] + terms_bdg['pturb']
# %%
terms_bdg['res_lhs+disp_prod']= terms_bdg['res_lhs']+terms_bdg['disp_prod']
terms_bdg['res_rhs-disp_trans']=terms_bdg['res_rhs']-terms_bdg['disptrans']

#%% Terms BDG Procedure 2 

# %%
## Smooth terms
for key in terms_bdg.keys():
  print(key)
  terms_bdg[key] = gaussian_filter(terms_bdg[key], sigma=(1, 2,2), mode='wrap')
  print(terms_bdg[key].min(), terms_bdg[key].max())




## Mask the data[ic]

mask = Z_gd<=0
for key in terms_bdg.keys():
    terms_bdg[key] = np.ma.array(terms_bdg[key], mask=mask)
    

items_bdg = []
for key in terms_bdg.keys():
    items_bdg.append(key)
    
items_bdg2 = []
for key in terms_bdg2.keys():
    items_bdg2.append(key)
    
## Renormalize the data[ic]

total_dissip = -(terms_bdg['dissip'] + terms_bdg['canopy'])
total_dissipv = get_vprof(total_dissip, Z_gd, amp)
#for item in items_bdg:
#  terms_bdg[item] /= total_dissip


# %%
from scipy.ndimage import gaussian_filter1d
total_dissip = gaussian_filter1d(-(terms_bdg2['dissip']+terms_bdg2['canopy']),3)
#mask2 = np.abs(total_dissip) < .1
#total_dissip[mask2] = np.nan
# %%
for item in items_bdg2:
    terms_bdg2[item] = gaussian_filter1d(terms_bdg2[item],1)
    terms_bdg2[item] /= total_dissipv
    


# %% 
terms_bdgavg = {}
terms_bdgdiff = {}

for key in terms_bdg2.keys():
     terms_bdgavg[key] = get_vprof(terms_bdg[key], Z_gd,amp)
    
    # # Calculate the difference between terms 
    # terms_bdgdiff[key] = np.zeros([dm.nz,dm.ny,dm.nx])
    # for i in range(dm.nx):
    #     for j in range(dm.ny):   
    #         terms_bdgdiff[key][:,j,i] = terms_bdg[key][:,j,i]-terms_bdg2[key]


# %% Profiles of Residual LHS RHS + dispersive terms
plt.rcParams['text.usetex'] = True
items = ['adv', 'uturb', 'pturb','prod', 'dissip','canopy']
items2 = ['adv', 'uturb', 'pturb','prod', 'dissip','canopy', 'disp_prod', 'disptrans']
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,8), dpi = 100)

for item in items:
    ax[0].plot(terms_bdgavg[item], z[:,0,0]/hc-.1, label = item)

ax[0].set(title= 'Avg Method 1')
ax[0].set(xlabel = r'$\frac{\partial e}{\partial t}/(\epsilon+\epsilon_c)$')
ax[0].set(ylabel = r'$z/h_c$')
ax[0].axhline(1.0, linestyle = '--', color = 'k', alpha = .5)
ax[0].axhline(2.0, linestyle = '-.', color = 'k', alpha = .5)
ax[0].set_ylim([0,5]) 
for item in items2:
    ax[1].plot(terms_bdg2[item],z[:,0,0]/hc-.1, label = item)
plt.legend()
ax[1].set(xlabel = r'$\frac{\partial e}{\partial t}/(\epsilon+\epsilon_c)$')
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].set(title = 'Avg Method 2 (Finnigan proc)' )
ax[1].axhline(1.0, linestyle = '--', color = 'k', alpha = .5)
ax[1].axhline(2.0, linestyle = '-.', color = 'k', alpha = .5)
ax[1].set_ylim([0,5])  
#ax[1].set_ylim([-3,3]) 
##ax[1].set_ylim([-2,2]) 
ax[1].set_yticks([])


# %% Plot the Results 


# Figure inputs 
sz = dict(left=0.08, right=0.98, bottom=0.17, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
ylim_z = 400
levels_bdg = np.arange(-3.5, 3.5+0.1, 0.05)
levels_ccl = np.arange(0, 1.01, 0.1)
ind_plot = [0, 6]
hc = 39.0
jcut = 100

## Plot the results
# Initlize the figure
cmap_bdg = cm.get_cmap("seismic")
cmap_bdg = cmap_bdg(np.linspace(0, 1, levels_bdg.size))
half = levels_bdg.size // 2
cmap_bdg[half-1:half+2, :] = 1
#print(cmap_bdg)
cmap_bdg = colors.ListedColormap(cmap_bdg)
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)
ifig = 0
fig = plt.figure(ifig, figsize=(12, 8))
nplot = len(ind_plot) * nca
gs = gridspec.GridSpec(1, nplot*2, width_ratios=[100, 20] * nplot) # Here is where width of plots are defined 
gs.update(**sz)


for inum, ind in enumerate(ind_plot):
  isub = 2*inum
  ax = fig.add_subplot(gs[isub])
  if ind == 9:
    cax=ax.contourf(x[:,1,:], z[:,1,:], -np.mean(terms_bdg[items_bdg[ind]],axis = 1)*hc/dm.zi,
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    #cax=ax.contourf(x[:,1,:], z[:,1,:], np.mean(terms_bdg[items_bdg[ind]]*hc/dm.zi, axis = 1),
    #  levels_bdg, cmap=cmap_bdg, extend='both')
    cax=ax.contourf(x[:,1,:], z[:,1,:], terms_bdg[items_bdg[ind]][:,jcut,:]*hc/dm.zi,
      levels_bdg, cmap=cmap_bdg, extend='both')
  #if lab_case[ic] == 'hill':
  #  cs=ax.contour(x_gd[ic], z_gd[ic], lvl_ccl[ic], levels=levels_ccl,
  #    linewidths=lw-0.5, linestyles='--')
  #  ax.clabel(cs, inline=True, fontsize='small')
  #ax.streamplot(x_gd, z_gd,
  #  data[ic]['u'], data[ic]['w'], color='gray', density=2, linewidth=lw-0.5,
  #  arrowsize=ars)
  ax.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
  ax.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
  ax.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow(~mask[:,1,:], extent=(0, dm.lx, 0, dm.lz), alpha=0.5,
    interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')
  
  # Axis property
  ax.set_xlim([0, dm.lx])
  ax.set_ylim([0, ylim_z/1.5])

  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)

  # Axis label
  
  ax.set_title(items_bdg[ind])
  ax.set_xlabel(r'$x (m)$')

  if isub == 0:
      ax.set_ylabel('$z (m)$')
  else:
      ax.set_yticklabels([])

  #ax.set_xticks([])
  #ax.set_yticks([])


  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+inum)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.12
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, .5)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

# %%

# Figure inputs 
sz = dict(left=0.08, right=0.98, bottom=0.17, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
ylim_z = 400
levels_bdg = np.arange(-2.0, 2.0+0.1, 0.1)
levels_ccl = np.arange(0, 1.01, 0.1)
ind_plot = [0, 6, 1, 4]
hc = 39.0

## Plot the results
# Initlize the figure
cmap_bdg = cm.get_cmap("seismic")
cmap_bdg = cmap_bdg(np.linspace(0, 1, levels_bdg.size))
half = levels_bdg.size // 2
cmap_bdg[half-1:half+2, :] = 1
#print(cmap_bdg)
cmap_bdg = colors.ListedColormap(cmap_bdg)
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)
ifig = 0
fig = plt.figure(ifig, figsize=(12, 8))
nplot = 2
gs = gridspec.GridSpec(2, 2, width_ratios=[100] * nplot) # Here is where width of plots are defined 
gs.update(**sz)


for inum, ind in enumerate(ind_plot):
  isub = inum
  ax = fig.add_subplot(gs[isub])
  if ind == 9:
    cax=ax.contourf(x[:,1,:], z[:,1,:], -np.mean(terms_bdg[items_bdg[ind]],axis = 1)*hc/dm.zi,
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    cax=ax.contourf(x[1:100,1,:], z[1:100,1,:], np.mean(terms_bdg[items_bdg[ind]][1:100,:,:]*hc/dm.zi, axis = 1),
      levels_bdg, cmap=cmap_bdg, extend = 'both')
  #if lab_case[ic] == 'hill':
  #  cs=ax.contour(x_gd[ic], z_gd[ic], lvl_ccl[ic], levels=levels_ccl,
  #    linewidths=lw-0.5, linestyles='--')
  #  ax.clabel(cs, inline=True, fontsize='small')
  #ax.streamplot(x_gd, z_gd,
  #  data[ic]['u'], data[ic]['w'], color='gray', density=2, linewidth=lw-0.5,
  #  arrowsize=ars)
  ax.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
  ax.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
  ax.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow(~mask[:,1,:], extent=(0, dm.lx, 0, dm.lz), alpha=0.5,
    interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')
  
  # Axis property
  ax.set_xlim([0, dm.lx])
  ax.set_ylim([0, ylim_z])

  # ax.xaxis.set_major_locator(xmaxLocator)
  # ax.yaxis.set_major_locator(ymaxLocator)

  # Axis label
  
  ax.set_title(items_bdg[ind])
  ax.set_xlabel(r'$x (m)$')

  if isub == 0:
      ax.set_ylabel('$z (m)$')
  else:
      ax.set_yticklabels([])

  # if isub == 0 or isub == 1:
  #     ax.set_xticks([])
  # if isub == 1 or isub == 3:
  #     ax.set_yticks([])


  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+inum)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.12
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, 0.5)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

# %% 

#levels_bdg = np.arange(-2.0, 2.0+0.1, 0.1)

fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows = 2, ncols = 2, dpi = 100, figsize = (14,8))

ax0.contourf(x[1:150,1,:], z[1:150,1,:], np.mean(terms_bdg[items_bdg[10]][1:150,:,:]*hc/dm.zi, axis = 1),
      levels_bdg, cmap=cmap_bdg, extend = 'both')
ax0.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
ax0.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
ax0.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
ax0.set(title = items_bdg[10],
        ylabel = 'z (m)')

ax1.contourf(x[1:150,1,:], z[1:150,1,:], np.mean(terms_bdg[items_bdg[4]][1:150,:,:]*hc/dm.zi, axis = 1),
      levels_bdg, cmap=cmap_bdg, extend = 'both')
ax1.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
ax1.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
ax1.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
ax1.set(title = items_bdg[4])

ax2.contourf(x[1:150,1,:], z[1:150,1,:], -np.mean(terms_bdg[items_bdg[5]][1:150,:,:]*hc/dm.zi, axis = 1),
      levels_bdg, cmap=cmap_bdg, extend = 'both')
ax2.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
ax2.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
ax2.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
ax2.set(title = items_bdg[5],
         ylabel = 'z (m)',
         )

ax3.contourf(x[1:150,1,:], z[1:150,1,:], -np.mean(terms_bdg[items_bdg[7]][1:150,:,:]*hc/dm.zi, axis = 1),
      levels_bdg, cmap=cmap_bdg, extend = 'both')
ax3.plot(x1, z_tpg[0, :], ls='-', lw=lw*2, c='k')
ax3.plot(x1, z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
ax3.plot(x1, z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
ax3.set(title = items_bdg[7], 
        )

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.12
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, .5) #.5
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')



# # Avg-poin for each individual term then sum up the differences
# # and compare those to the dispersive terms

# L = 250
# x1L = x1/L
# xv = [0, 1.0, 2.0, 3.0, 4.0]
# jcut = 80
# #terms = ['res_lhs', 'res_rhs']
# #terms = ['res_lhs+disp_prod', 'res_rhs-disp_trans']
# terms = ['prod','adv','uturb','pturb','canopy',\
#                   'dissip']
# fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (15,8), dpi = 100)
# for i in range(len(xv)):
#     if  i < len(xv):
#         it = np.where(x1L == xv[i])[0][0]
        
#         # ax[i].plot((terms_bdg['res_rhs'][:,jcut,it]-terms_bdgavg['res_rhs'])*hc/dm.zi,z1, label = 'res_lhs-res_lhs_avg')
#         # ax[i].plot((terms_bdg['Disp_prod'][:,jcut,it]+terms_bdg['disptrans'][:,jcut,it])*hc/dm.zi,z1,label='Disp Trans')
#         # ax[i].axhline(np.max(z_tpg[:,it]), linestyle = '--', color = 'k', alpha = .5)
#         # ax[i].axhline(np.max(z_tpg[:,it])+hc, linestyle = '--', color = 'k', alpha = .5)
#         # ax[i].set(title = 'Loc '+str(xv[i]))
#         # if i == len(xv)-1:
#         #     ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))  
#         diff = np.zeros(dm.nz)
#         for j in range(len(terms)):
#             diff[:] += terms_bdgdiff[terms[j]][:,jcut,it]*hc/dm.zi
#         ax[i].plot(diff,z1, \
#                    label = 'Diff', linewidth = 2)
#         ax[i].plot((terms_bdg2['disp_prod']+terms_bdg2['disptrans'])*hc/dm.zi,z1,\
#                    label = 'Disp', linewidth = 2)
#         ax[i].set(title = 'Loc '+str(xv[i])) 
#         ax[i].axhline(np.max(z_tpg[:,it]), linestyle = '--', color = 'k', alpha = .5)
#         ax[i].axhline(np.max(z_tpg[:,it])+hc, linestyle = '--', color = 'k', alpha = .5)  
#         ax[i].set_ylim([0,580])  
#         if i != 0: ax[i].set_yticks([])
#         if i == 4: ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     # else:            
#     #     for j in range(len(terms)):
#     #         ax[i].plot(terms_bdgavg[terms[j]]*hc/dm.zi,z1, \
#     #                     label = terms[j], linewidth = 2)
#     #     ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #     ax[i].set(title = 'Avg' )
#     #     ax[i].axhline(np.max(z_tpg), linestyle = '--', color = 'k', alpha = .5)
#     #     ax[i].axhline(np.max(z_tpg)+hc, linestyle = '--', color = 'k', alpha = .5)
#     #     ax[i].set_ylim([0,580])  
#     #     ax[i].set_yticks([])
    


# #%% Profiles at various x locations 

# terms_bdg_list = ['Disp_prod','prod','adv','uturb','disptrans','pturb','canopy',\
#                   'dissip']
    
# terms_bdg_avg2 = ['prod','canopy',\
#                   'dissip','disp_prod'] 
# terms_bdg_tower2 = ['prod','canopy',\
#                   'dissip'] 
    
# terms_bdg_avg2 = ['adv','uturb','pturb','disptrans'] 
# terms_bdg_tower2 = ['adv','uturb','pturb'] 
    
# L = 250
# x1L = x1/L
# xv = [0, 1.0, 2.0, 3.0, 5.0]
# jcut = 80


# fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (15,8), dpi = 100)
# for i in range(len(xv)):
#     if  i < len(xv)-1:
#         it = np.where(x1L == xv[i])[0][0]
        
#         for j in range(len(terms_bdg_tower2)):
#             ax[i].plot(terms_bdg[terms_bdg_tower2[j]][:,jcut,it]*hc/dm.zi,z1, \
#                        label = terms_bdg_tower2[j], linewidth = 2)
#         ax[i].set(title = 'Loc '+str(xv[i])) 
#         ax[i].axhline(np.max(z_tpg[:,it]), linestyle = '--', color = 'k', alpha = .5)
#         ax[i].axhline(np.max(z_tpg[:,it])+hc, linestyle = '--', color = 'k', alpha = .5)  
#         ax[i].set_ylim([0,580])  
#         if i != 0: ax[i].set_yticks([])
#         #if i == 4: ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     else:            
#         for j in range(len(terms_bdg_avg2)):
#             ax[i].plot(terms_bdg2[terms_bdg_avg2[j]]*hc/dm.zi,z1, \
#                        label = terms_bdg_avg2[j], linewidth = 2)
#         ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         ax[i].set(title = 'Avg' )
#         ax[i].axhline(np.max(z_tpg), linestyle = '--', color = 'k', alpha = .5)
#         ax[i].axhline(np.max(z_tpg)+hc, linestyle = '--', color = 'k', alpha = .5)
#         ax[i].set_ylim([0,580])  
#         ax[i].set_yticks([])
    