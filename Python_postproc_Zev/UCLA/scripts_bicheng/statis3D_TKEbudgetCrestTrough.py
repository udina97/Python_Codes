#!/usr/bin/env python3

"""
Program: statis3D_TKEbudgetCrestTrough
"""

### Histories:
### 8/1/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created.



## Prerequisite Module
import lespy as lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import itertools
import sys



## User-specified Variable
# Case
case = 'amazon_canopy_real'
lab_case = 'real'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fmt_base = './data/statistics3D_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_ptb = './data/statistics_pertubation_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_drv = './data/statistics_derivative_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_bdg = './data/statistics_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_CTstat = './data/statistics_budget_CrestTrough_alpha{alpha:01d}_{case:s}_tt{tts:08d}-{tte:08d}.npz'

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
tts = 144100
tte = 324000
#tte = 216000
Z_prof = np.arange(1, 3*hc+1, 2)
dz = 2
#wid_ct = 4
items = ('u', 'v', 'w', 'p', 'u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_vel = ('u', 'v', 'w', 'u2', 'v2', 'w2')
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'canopy', 'dissip', 'sum', 'prod_dudz')
items_h = ('adv_h', 'uturb_h', 'pturb_h')
items_vt = ('uturb_v',)
items_vo = ('adv_v', 'pturb_v')
items_R = ('P-(eps+eps_c)', 'R^h', '-T_e^v', '-pi_e^v', '-A_e^v', 'R')

# Filter criteria
alpha_flt = 2

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.1, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X', 'h', '+', '8', 'p', '*', 'P', 'd', 'H')
xspacing = 1
yspacing = 0.5
xlim = (-2, 3)
ylim = (0, 3)



## Functions
def get_dphidz(phi, dz, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1] = (phi[1:]-phi[:-1]) / (dz/dm.zi)
  dphidz[-1] = dphidz[-2]
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

def get_vprof(terms, items, Z_gd, Z_prof, iy, ix):
  prof = dict()
  for item in items:
    prof[item] = np.interp(Z_prof, Z_gd[:, iy, ix], terms[item][:, iy, ix])

  return prof

def getCrest_line(z_tpg):
  ix_crest = []
  iy_crest = []
  hgt_crest = []
  size_crest = []
  sizeWind_crest = []
  sizeLee_crest = []
  dhWind_crest = []
  dhLee_crest = []
  for iy in range(z_tpg.shape[0]):
    for ix in range(z_tpg.shape[-1]):
      ixs = ix - 1
      ixe = int(np.mod(ix + 1, z_tpg.shape[-1]))
      if np.all(z_tpg[iy, ix] >= z_tpg[iy, [ixs, ixe]]):
        iy_crest.append(iy)
        ix_crest.append(ix)
        hgt_crest.append(z_tpg[iy, ix])
        # Get crest size
        size_crest.append(0)
        sizeWind_crest.append(0)
        sizeLee_crest.append(0)
        dhWind_crest.append(0)
        dhLee_crest.append(0)
        ind = len(size_crest) - 1
  
        ixs = ix_crest[ind]-1
        while z_tpg[iy_crest[ind], ixs]<=z_tpg[iy_crest[ind], ixs+1]:
          ixs -= 1
          sizeWind_crest[ind] += 1
        dhWind_crest[ind] =\
          z_tpg[iy_crest[ind], ix_crest[ind]]\
          - z_tpg[iy_crest[ind], ixs+1]
      
        ixe = int(np.mod(ix_crest[ind] + 1, z_tpg.shape[-1]))
        while z_tpg[iy_crest[ind], ixe]<=z_tpg[iy_crest[ind], ixe-1]:
          ixe += 1
          if ixe >= z_tpg.shape[-1]:
            ixe -= z_tpg.shape[-1]
          sizeLee_crest[ind] += 1
        dhLee_crest[ind] =\
          z_tpg[iy_crest[ind], ix_crest[ind]]\
          - z_tpg[iy_crest[ind], ixe-1]
  
        size_crest[ind] = sizeWind_crest[ind] + sizeLee_crest[ind] + 1
      
  
  #ind_sorted = np.argsort(hgt_crest)[::-1]
  ind_sorted = np.argsort(dhWind_crest)[::-1]
  iy_crest = np.array(iy_crest)[ind_sorted]
  ix_crest = np.array(ix_crest)[ind_sorted]
  hgt_crest = np.array(hgt_crest)[ind_sorted]
  size_crest = np.array(size_crest)[ind_sorted]
  sizeWind_crest = np.array(sizeWind_crest)[ind_sorted]
  sizeLee_crest = np.array(sizeLee_crest)[ind_sorted]
  dhWind_crest = np.array(dhWind_crest)[ind_sorted]
  dhLee_crest = np.array(dhLee_crest)[ind_sorted]
  n_crest = len(hgt_crest)
  
  return iy_crest, ix_crest, hgt_crest, size_crest,\
    sizeWind_crest, sizeLee_crest, dhWind_crest, dhLee_crest, n_crest

def getTrough_line(z_tpg):
  ix_trough = []
  iy_trough = []
  hgt_trough = []
  size_trough = []
  sizeWind_trough = []
  sizeLee_trough = []
  dhWind_trough = []
  dhLee_trough = []
  for iy in range(z_tpg.shape[0]):
    for ix in range(z_tpg.shape[-1]):
      ixs = ix - 1
      ixe = int(np.mod(ix + 1, z_tpg.shape[-1]))
      if np.all(z_tpg[iy, ix] <= z_tpg[iy, [ixs, ixe]]):
        iy_trough.append(iy)
        ix_trough.append(ix)
        hgt_trough.append(z_tpg[iy, ix])
        # Get trough size
        size_trough.append(0)
        sizeWind_trough.append(0)
        sizeLee_trough.append(0)
        dhWind_trough.append(0)
        dhLee_trough.append(0)
        ind = len(size_trough) - 1
  
        ixs = ix_trough[ind]-1
        while z_tpg[iy_trough[ind], ixs]>=z_tpg[iy_trough[ind], ixs+1]:
          ixs -= 1
          sizeWind_trough[ind] += 1
        dhWind_trough[ind] =\
          z_tpg[iy_trough[ind], ix_trough[ind]]\
          - z_tpg[iy_trough[ind], ixs+1]
  
        ixe = int(np.mod(ix_trough[ind] + 1, z_tpg.shape[-1]))
        while z_tpg[iy_trough[ind], ixe]>=z_tpg[iy_trough[ind], ixe-1]:
          ixe += 1
          if ixe >= z_tpg.shape[-1]:
            ixe -= z_tpg.shape[-1]
          sizeLee_trough[ind] += 1
        dhLee_trough[ind] =\
          z_tpg[iy_trough[ind], ix_trough[ind]]\
          - z_tpg[iy_trough[ind], ixe-1]
  
        size_trough[ind] = sizeWind_trough[ind] + sizeLee_trough[ind] + 1
  
  #ind_sorted = np.argsort(hgt_trough)
  ind_sorted = np.argsort(dhWind_trough)[::-1]
  iy_trough = np.array(iy_trough)[ind_sorted]
  ix_trough = np.array(ix_trough)[ind_sorted]
  hgt_trough = np.array(hgt_trough)[ind_sorted]
  size_trough = np.array(size_trough)[ind_sorted]
  sizeWind_trough = np.array(sizeWind_trough)[ind_sorted]
  sizeLee_trough = np.array(sizeLee_trough)[ind_sorted]
  dhWind_trough = np.array(dhWind_trough)[ind_sorted]
  dhLee_trough = np.array(dhLee_trough)[ind_sorted]
  n_trough = len(hgt_trough)

  return iy_trough, ix_trough, hgt_trough, size_trough,\
    sizeWind_trough, sizeLee_trough, dhWind_trough, dhLee_trough, n_trough

def getCrestTrough(z_tpg, alpha_flt):
  ix_crest = []
  iy_crest = []
  hgt_crest = []
  ix_trough = []
  iy_trough = []
  hgt_trough = []
  dh_flt = alpha_flt*z_tpg.std()
  for iy in range(z_tpg.shape[0]):
    ix_max = np.argmax(z_tpg[iy, :])
    z_max = z_tpg[iy, ix_max]
    ix_min = np.argmin(z_tpg[iy, :])
    z_min = z_tpg[iy, ix_min]
    if z_max - z_min < dh_flt:
      continue
    else:
      ix_cur = ix_max
      tp_cur = 'crest'
      for ind in range(ix_max+1, ix_max+1+z_tpg.shape[-1]):
        ix_chk = int(np.mod(ind, z_tpg.shape[-1]))
        ixs = ix_chk - 1
        ixe = int(np.mod(ix_chk+1, z_tpg.shape[-1]))
        if tp_cur == 'crest': # current potential point is a crest
          if all(z_tpg[iy, ix_chk]<=z_tpg[iy, [ixs, ixe]]): # Local trough
            if (z_tpg[iy, ix_cur] - z_tpg[iy, ix_chk] >= dh_flt):
              ix_crest.append(ix_cur)
              iy_crest.append(iy)
              hgt_crest.append(z_tpg[iy, ix_cur])
              ix_cur = ix_chk
              tp_cur = 'trough'
          elif all(z_tpg[iy, ix_chk]>=z_tpg[iy, [ixs, ixe]]): # Local crest
            if (z_tpg[iy, ix_chk]>z_tpg[iy, ix_cur]):
              ix_cur = ix_chk # Update current potential crest
        elif tp_cur == 'trough': # current potential point is a trough
          if all(z_tpg[iy, ix_chk]>=z_tpg[iy, [ixs, ixe]]): # Local Crest
            if (z_tpg[iy, ix_cur] - z_tpg[iy, ix_chk] <= -dh_flt):
              ix_trough.append(ix_cur)
              iy_trough.append(iy)
              hgt_trough.append(z_tpg[iy, ix_cur])
              ix_cur = ix_chk
              tp_cur = 'crest'
          elif all(z_tpg[iy, ix_chk]<=z_tpg[iy, [ixs, ixe]]): # Local trough
            if (z_tpg[iy, ix_chk]<z_tpg[iy, ix_cur]):
              ix_cur = ix_chk # Update current potential crest

  ind_sorted = np.argsort(hgt_crest)[::-1]
  iy_crest = np.array(iy_crest)[ind_sorted]
  ix_crest = np.array(ix_crest)[ind_sorted]
  hgt_crest = np.array(hgt_crest)[ind_sorted]

  ind_sorted = np.argsort(hgt_trough)
  iy_trough = np.array(iy_trough)[ind_sorted]
  ix_trough = np.array(ix_trough)[ind_sorted]
  hgt_trough = np.array(hgt_trough)[ind_sorted]

  n_crest = len(hgt_crest)
  n_trough = len(hgt_trough)
  print('#'*80)
  print('Numter of crest,', n_crest)
  print('Numter of trough,', n_trough)
  print('#'*80)

  return iy_crest, ix_crest, hgt_crest, n_crest, iy_trough, ix_trough, hgt_trough, n_trough



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

fn_base = fmt_base.format(case=case, tts=tts, tte=tte)
fn_ptb = fmt_ptb.format(case=case, tts=tts, tte=tte)
fn_drv = fmt_drv.format(case=case, tts=tts, tte=tte)
fn_bdg = fmt_bdg.format(case=case, tts=tts, tte=tte)

dset_base = np.load(fn_base, mmap_mode='r', allow_pickle=True)
dset_ptb = np.load(fn_ptb, mmap_mode='r', allow_pickle=True)
dset_drv = np.load(fn_drv, mmap_mode='r')
dset_bdg = np.load(fn_bdg, mmap_mode='c', allow_pickle=True)

z_tpg = dset_base['z_tpg']
z_gd = dset_base['z']
terms_base = dset_base['data'].item()
terms_bdg = dset_bdg['terms'].item()
terms_ptb = dset_ptb['terms'].item()



## Smooth the data
for key in terms_bdg.keys():
  if key != 'sum':
    terms_bdg[key] = gaussian_filter(terms_bdg[key], sigma=(1, 3, 3),
      mode='wrap')



## Sum of some terms
# Total balance
terms_bdg['sum'] = terms_bdg['prod_h'] + terms_bdg['prod_v']\
  + terms_bdg['adv_h'] + terms_bdg['adv_v']\
  + terms_bdg['uturb_h'] + terms_bdg['uturb_v']\
  + terms_bdg['pturb_h'] + terms_bdg['pturb_v']\
  + terms_bdg['canopy'] + terms_bdg['dissip']
# Total dissipation rate
terms_bdg['dissip_ttl'] = terms_bdg['dissip'] + terms_bdg['canopy']



## Extract the data from the crest
iy_crest, ix_crest, hgt_crest, n_crest, iy_trough, ix_trough, hgt_trough, n_trough =\
  getCrestTrough(z_tpg, alpha_flt)
# Find crests
#iy_crest, ix_crest, hgt_crest, size_crest, sizeWind_crest, sizeLee_crest,\
#  dhWind_crest, dhLee_crest, n_crest = getCrest_line(z_tpg)

# Find troughs
#iy_trough, ix_trough, hgt_trough, size_trough, sizeWind_trough, sizeLee_trough,\
#  dhWind_trough, dhLee_trough, n_trough = getTrough_line(z_tpg)



## Extract the profile
bdg_crest = [dict.fromkeys(items_bdg, None) for ic in range(n_crest)]
bdg_trough = [dict.fromkeys(items_bdg, None) for ic in range(n_trough)]
vel_crest = [dict.fromkeys(items_vel, None) for ic in range(n_crest)]
vel_trough = [dict.fromkeys(items_vel, None) for ic in range(n_trough)]
uw_crest = [None for ic in range(n_crest)]
uw_trough = [None for ic in range(n_trough)]
txz_crest = [None for ic in range(n_crest)]
txz_trough = [None for ic in range(n_trough)]

Z_gd = z_gd - z_tpg[np.newaxis, :, :]

for ic in range(n_crest):
  bdg_crest[ic] = get_vprof(terms_bdg, items_bdg, Z_gd, Z_prof,
    iy_crest[ic], ix_crest[ic])
  vel_crest[ic] = get_vprof(terms_base, items_vel, Z_gd, Z_prof,
    iy_crest[ic], ix_crest[ic])
  uw_crest[ic] = get_vprof(terms_ptb, ['uw_t',], Z_gd, Z_prof,
    iy_crest[ic], ix_crest[ic])['uw_t']
  txz_crest[ic] = get_vprof(terms_base, ['txz',], Z_gd, Z_prof,
    iy_crest[ic], ix_crest[ic])['txz']

for ic in range(n_trough):
  bdg_trough[ic] = get_vprof(terms_bdg, items_bdg, Z_gd, Z_prof,
    iy_trough[ic], ix_trough[ic])
  vel_trough[ic] = get_vprof(terms_base, items_vel, Z_gd, Z_prof,
    iy_trough[ic], ix_trough[ic])
  uw_trough[ic] = get_vprof(terms_ptb, ['uw_t',], Z_gd, Z_prof,
    iy_trough[ic], ix_trough[ic])['uw_t']
  txz_trough[ic] = get_vprof(terms_base, ['txz',], Z_gd, Z_prof,
    iy_trough[ic], ix_trough[ic])['txz']



## Calculate terms
# Residual terms
res_crest = [dict.fromkeys(items_R, None) for ic in range(n_crest)]
res_trough = [dict.fromkeys(items_R, None) for ic in range(n_trough)]

for ic in range(n_crest):
  res_crest[ic]['P-(eps+eps_c)'] =\
    bdg_crest[ic]['prod_h'] + bdg_crest[ic]['prod_v']\
    + bdg_crest[ic]['dissip'] + bdg_crest[ic]['canopy']

  res_crest[ic]['R^h'] = 0
  for item in items_h:
    res_crest[ic]['R^h'] -= bdg_crest[ic][item]
  
  res_crest[ic]['-T_e^v'] = 0
  for item in items_vt:
    res_crest[ic]['-T_e^v'] -= bdg_crest[ic][item]
  
  res_crest[ic]['-pi_e^v'] = -bdg_crest[ic]['pturb_v']
  
  res_crest[ic]['-A_e^v'] = -bdg_crest[ic]['adv_v']
  
  res_crest[ic]['R'] = res_crest[ic]['R^h'] + res_crest[ic]['-T_e^v']\
    + res_crest[ic]['-pi_e^v'] + res_crest[ic]['-A_e^v']

for ic in range(n_trough):
  res_trough[ic]['P-(eps+eps_c)'] =\
    bdg_trough[ic]['prod_h'] + bdg_trough[ic]['prod_v']\
    + bdg_trough[ic]['dissip'] + bdg_trough[ic]['canopy']

  res_trough[ic]['R^h'] = 0
  for item in items_h:
    res_trough[ic]['R^h'] -= bdg_trough[ic][item]
  
  res_trough[ic]['-T_e^v'] = 0
  for item in items_vt:
    res_trough[ic]['-T_e^v'] -= bdg_trough[ic][item]
  
  res_trough[ic]['-pi_e^v'] = -bdg_trough[ic]['pturb_v']
  
  res_trough[ic]['-A_e^v'] = -bdg_trough[ic]['adv_v']
  
  res_trough[ic]['R'] = res_trough[ic]['R^h'] + res_trough[ic]['-T_e^v']\
    + res_trough[ic]['-pi_e^v'] + res_trough[ic]['-A_e^v']

# displacement height
d0_crest = [None for ic in range(n_crest)]
d0_trough = [None for ic in range(n_trough)]

for ic in range(n_crest):
  uw = uw_crest[ic] + txz_crest[ic]
  duwdz = get_dphidz(uw, dz, dm)
  mask_can = np.logical_and(Z_prof>=0, Z_prof<=hc+dm.dz/2)
  uw_top = np.interp(hc, Z_prof, uw)
  uw = uw[mask_can]
  duwdz = duwdz[mask_can]
  z_can = Z_prof[mask_can]
  integrate = np.trapz(duwdz/dm.zi*z_can, z_can)
  d0_crest[ic] = integrate/uw_top

for ic in range(n_trough):
  uw = uw_trough[ic] + txz_trough[ic]
  duwdz = get_dphidz(uw, dz, dm)
  mask_can = np.logical_and(Z_prof>=0, Z_prof<=hc+dm.dz/2)
  uw_top = np.interp(hc, Z_prof, uw)
  uw = uw[mask_can]
  duwdz = duwdz[mask_can]
  z_can = Z_prof[mask_can]
  integrate = np.trapz(duwdz/dm.zi*z_can, z_can)
  d0_trough[ic] = integrate/uw_top



## Save the data
#print(size_crest)
#print(len(size_trough))
#print(len(hgt_trough))
fn_CTstat = fmt_CTstat.format(alpha=alpha_flt, case=case, tts=tts, tte=tte)
#np.savez(fn_CTstat,
#  ix_crest=ix_crest, iy_crest=iy_crest, hgt_crest=hgt_crest,
#  size_crest=size_crest,
#  sizeWind_crest=sizeWind_crest, sizeLee_crest=sizeLee_crest,
#  dhWind_crest=dhWind_crest, dhLee_crest=dhLee_crest,
#  ix_trough=ix_trough, iy_trough=iy_trough, hgt_trough=hgt_trough,
#  size_trough = size_trough,
#  sizeWind_trough=sizeWind_trough, sizeLee_trough=sizeLee_trough,
#  dhWind_trough=dhWind_trough, dhLee_trough=dhLee_trough,
#  bdg_crest=bdg_crest, bdg_trough=bdg_trough,
#  vel_crest=vel_crest, vel_trough=vel_trough,
#  res_crest=res_crest, res_trough=res_trough,
#  uw_crest=uw_crest, uw_trough=uw_trough,
#  txz_crest=txz_crest, txz_trough=txz_trough,
#  d0_crest=d0_crest, d0_trough=d0_trough,
#  Z_prof=Z_prof)
np.savez(fn_CTstat,
  ix_crest=ix_crest, iy_crest=iy_crest, hgt_crest=hgt_crest,
  ix_trough=ix_trough, iy_trough=iy_trough, hgt_trough=hgt_trough,
  bdg_crest=bdg_crest, bdg_trough=bdg_trough,
  vel_crest=vel_crest, vel_trough=vel_trough,
  res_crest=res_crest, res_trough=res_trough,
  uw_crest=uw_crest, uw_trough=uw_trough,
  txz_crest=txz_crest, txz_trough=txz_trough,
  d0_crest=d0_crest, d0_trough=d0_trough,
  Z_prof=Z_prof)

fn_paper = "./data/forpaper/fig1_topography.npz"
np.savez(fn_paper,
  ix_crest=ix_crest, iy_crest=iy_crest, hgt_crest=hgt_crest,
  ix_trough=ix_trough, iy_trough=iy_trough, hgt_trough=hgt_trough)
