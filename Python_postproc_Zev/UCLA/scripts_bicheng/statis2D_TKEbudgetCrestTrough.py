#!/usr/bin/env python3

"""
Program: statis2D_TKEbudgetCrestTrough
"""

### Histories:
### 10/21/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created.



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



## User-specified Variable
# Case
case = 'amazon_canopy_hill2h'
lab_case = 'hill2h'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_2D/{case:s}'
fmt_CTstat = './data/statistics_budget_CrestTrough_{case:s}_tt{tts:08d}-{tte:08d}.npz'

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 18000
te = 25200
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
ix_crest = (80, 240)
ix_trough = (0, 160)

# Topography parameters
zpad = 15
amp = 50
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

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
def get_topo(x, wl, amp):
  return (amp*np.cos(2*np.pi/wl*x+np.pi)+amp)

def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = complex(0, 1) * wn[np.newaxis, :] * phi_c
  dphidx_c[:, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

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

def get_vprof(terms, items, Z_gd, Z_prof, ix):
  prof = dict()
  for item in items:
    if len(ix) == 1:
      prof[item] = np.interp(Z_prof, Z_gd[:, ix], terms[item][:, ix])
    elif len(ix) > 1:
      prof[item] = np.interp(Z_prof, np.mean(Z_gd[:, ix], axis=-1),
        np.mean(terms[item][:, ix], axis=-1))

  return prof



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

data = dict.fromkeys(items, None)
terms_bdg = dict.fromkeys(items_bdg, None)
for item in items:
  data[item], time = lp.io.io_averFile.loadLES_averFile(param,
    qtype=item, tss=ts, tes=te)
  data[item] = np.mean(data[item], axis=0)



## Get the coordinates
z_gd, x_gd = lp.domain.dmFun.xzcoord(dm, ztype='staggered')
z_tpg = get_topo(x_gd, wl, amp)
z_gd -= zpad
Z_gd = z_gd-z_tpg



## Calculate the budget terms
field_bdg = dict.fromkeys(items, None)
field_R = dict.fromkeys(items_R, None)
wn = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)

# Calculate the production
data['w'] = gaussian_filter(data['w'], sigma=(0, 1), mode='wrap')
data['dwdx'] = get_dphidx(data['w'], wn)
data['dwdx'] = gaussian_filter(data['dwdx'], sigma=(1, 1),
  mode='wrap')

data['u'] = gaussian_filter(data['u'], sigma=(0, 1), mode='wrap')
data['dudx'] = get_dphidx(data['u'], wn)
data['dudx'] = uvpnode2wnode(data['dudx'])
data['dudx'] = gaussian_filter(data['dudx'], sigma=(1, 1),
  mode='wrap')

data['dudz'] = np.zeros(data['u'].shape) 
data['dudz'][1:, :] =\
  (data['u'][1:, :]-data['u'][:-1, :]) / (dm.dz/dm.zi)
data['dudz'] = gaussian_filter(data['dudz'], sigma=(1, 1),
  mode='wrap')

data['dwdz'] = np.zeros(data['w'].shape) 
data['dwdz'] = uvpnode2wnode(data['dwdz'])
data['dwdz'] = gaussian_filter(data['dwdz'], sigma=(1, 1),
  mode='wrap')

data['uw_t'] = data['uw'] - data['u']*data['w']
data['uw_t'] = gaussian_filter(data['uw_t'], sigma=(1, 1),
  mode='wrap')
data['uu_t'] = data['u2'] - data['u']**2
data['uu_t'] = gaussian_filter(data['uu_t'], sigma=(1, 1),
  mode='wrap')
data['vv_t'] = data['v2'] - data['v']**2
data['vv_t'] = gaussian_filter(data['vv_t'], sigma=(1, 1),
  mode='wrap')
data['ww_t'] = data['w2'] - data['w']**2
data['ww_t'] = gaussian_filter(data['ww_t'], sigma=(1, 1),
  mode='wrap')

# TKE
tke = (data['uu_t'] + data['vv_t'] + data['ww_t']) / 2

# Production
field_bdg['prod_h'] = -data['uu_t']*data['dudx']\
  - data['uw_t']*data['dwdx']
field_bdg['prod_v'] = - data['ww_t']*data['dwdz']\
  - data['uw_t']*data['dudz']
field_bdg['prod'] =\
  field_bdg['prod_h'] + field_bdg['prod_v']
field_bdg['prod_dudz'] = -data['uw_t']*data['dudz']

# Strain terms
data['dvdx'] = get_dphidx(data['v'], wn)
data['dvdx'] = uvpnode2wnode(data['dvdx'])
data['dvdx'] = gaussian_filter(data['dvdx'], sigma=(1, 1), mode='wrap')

data['dvdz'] = np.zeros(data['v'].shape) 
data['dvdz'][1:, :] =\
  (data['v'][1:, :]-data['v'][:-1, :]) / (dm.dz/dm.zi)
data['dvdz'] = gaussian_filter(data['dvdz'], sigma=(1, 1), mode='wrap')
ux = data['dudx']
uz = data['dudz']
vx = data['dvdx']
vz = data['dvdz']
wx = data['dwdx']
wz = data['dwdz']

S11 = ux
S12 = 0.5 * vx
S13 = 0.5 * (uz + wx)
S22 = 0
S23 = 0.5 * vz
S33 = wz

# Advection term
ue = data['u']*tke
duedx = get_dphidx(ue, wn)
tkez = np.zeros(tke.shape)
tkez[0, :] = 0
tkez[1:, :] = 0.5*(tke[:-1, :] + tke[1:, :])
we = data['w']*tkez
dwedz = get_dphidz(we, dz, dm)
field_bdg['adv_h'] = -duedx
field_bdg['adv_v'] = -dwedz
field_bdg['adv'] =\
  field_bdg['adv_h'] + field_bdg['adv_v']

# Turbulent transport term in horizontal
turb_u3 = data['u3'] - 3*data['u']*data['u2']+2*data['u']**3
turb_uv2 = data['uv2'] - 2*data['v']*data['uv']\
  + 2*data['u']*data['v']**2-data['u']*data['v2']
uw = wnode2uvpnode(data['uw'])
w_c = wnode2uvpnode(data['w'])
w2_c = wnode2uvpnode(data['w2'])
turb_uw2 = data['uw2'] - 2*w_c*uw\
  + 2*data['u']*w_c**2-data['u']*w2_c
ue = 0.5*(turb_u3 + turb_uv2 + turb_uw2)
duedx = get_dphidx(ue, wn)
utxx = data['utxx'] - data['u']*data['txx']
vtxy = data['vtxy'] - data['v']*data['txy']
wtxz = data['wtxz'] - data['w']*data['txz']
wtxz_uv = wnode2uvpnode(wtxz)
dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn)
field_bdg['uturb_h'] = -duedx-dutaudx

# Turbulent transport term in vertical
u_w = uvpnode2wnode(data['u'])
u2_w = uvpnode2wnode(data['u2'])
turb_wu2 = data['u2w'] - 2*u_w*data['uw']\
  + 2*data['w']*u_w**2 - data['w']*u2_w
v_w = uvpnode2wnode(data['v'])
v2_w = uvpnode2wnode(data['v2'])
turb_wv2 = data['v2w'] - 2*v_w*data['vw']\
  + 2*data['w']*v_w**2 - data['w']*v2_w
turb_w3 = data['w3'] - 3*data['w']*data['w2']\
  + 2*data['w']**3
we = 0.5*(turb_wu2 + turb_wv2 + turb_w3)
dwedz = get_dphidz(we, dz, dm)
utxz = data['utxz'] - u_w*data['txz']
vtyz = data['vtyz'] - v_w*data['tyz']
wtzz = data['wtzz'] - data['w']*data['tzz']
dutaudz = get_dphidz(utxz+vtyz+wtzz, dz, dm)
field_bdg['uturb_v'] = -dwedz-dutaudz
field_bdg['uturb'] =\
  field_bdg['uturb_h'] + field_bdg['uturb_v']

# Pressure transport term
pu = data['pu'] - data['p']*data['u']
dpudx = get_dphidx(pu, wn)
p_w = uvpnode2wnode(data['p'])
pw = data['pw'] - p_w*data['w']
dpwdz = get_dphidz(pw, dz, dm)
field_bdg['pturb_h'] = -dpudx
field_bdg['pturb_v'] = -dpwdz
field_bdg['pturb'] =\
  field_bdg['pturb_h'] + field_bdg['pturb_v']

# Dissipation
field_bdg['canopy'] = (data['wFcz']-data['w']*data['Fcz'])
field_bdg['canopy'] = wnode2uvpnode(field_bdg['canopy'])
field_bdg['canopy'] += (data['uFcx']-data['u']*data['Fcx'])\
  +(data['vFcy']-data['v']*data['Fcy'])
field_bdg['dissip'] = data['dissip']\
  - data['txx']*S11 - data['tyy']*S22 - data['tzz']*S33\
  - 2*data['txy']*S12 - 2*data['txz']*S13 - 2*data['tyz']*S23

# Sum of above terms
field_bdg['sum'] = field_bdg['prod_h'] + field_bdg['prod_v']\
  + field_bdg['adv_h'] + field_bdg['adv_v']\
  + field_bdg['uturb_h'] + field_bdg['uturb_v']\
  + field_bdg['pturb_h'] + field_bdg['pturb_v']\
  + field_bdg['canopy'] + field_bdg['dissip'] 

# Residual terms
field_R['P-(eps+eps_c)'] = field_bdg['prod_h']\
  + field_bdg['prod_v'] + field_bdg['dissip']\
  + field_bdg['canopy']

field_R['R^h'] = 0
for item in items_h:
  field_R['R^h'] -= field_bdg[item]

field_R['-T_e^v'] = 0
for item in items_vt:
  field_R['-T_e^v'] -= field_bdg[item]

field_R['-pi_e^v'] = -field_bdg['pturb_v']

field_R['-A_e^v'] = -field_bdg['adv_v']

field_R['R'] = field_R['R^h'] + field_R['-T_e^v']\
  + field_R['-pi_e^v'] + field_R['-A_e^v']



## Smooth the data
for item in items_bdg:
  field_bdg[item] = gaussian_filter(field_bdg[item],
    sigma=(1, 2), mode='wrap')
for item in items_R:
  field_R[item] = gaussian_filter(field_R[item],
    sigma=(1, 2), mode='wrap')



## Mask the data
mask = Z_gd<=0
for item in items_bdg:
  field_bdg[item] = np.ma.array(field_bdg[item], mask=mask)
for item in items_R:
  field_R[item] = np.ma.array(field_R[item], mask=mask)



## Get vertical profiles
bdg_crest = get_vprof(field_bdg, items_bdg, Z_gd, Z_prof, ix_crest)
vel_crest = get_vprof(data, items_vel, Z_gd, Z_prof, ix_crest)
uw_crest = get_vprof(data, ['uw_t', ], Z_gd, Z_prof, ix_crest)['uw_t']
txz_crest = get_vprof(data, ['txz', ], Z_gd, Z_prof, ix_crest)['txz']

bdg_trough = get_vprof(field_bdg, items_bdg, Z_gd, Z_prof, ix_trough)
vel_trough = get_vprof(data, items_vel, Z_gd, Z_prof, ix_trough)
uw_trough = get_vprof(data, ['uw_t', ], Z_gd, Z_prof, ix_trough)['uw_t']
txz_trough = get_vprof(data, ['txz', ], Z_gd, Z_prof, ix_trough)['txz']



## Calculate terms
# Residual terms
res_crest = dict.fromkeys(items_R, None)
res_trough = dict.fromkeys(items_R, None)

res_crest['P-(eps+eps_c)'] = bdg_crest['prod_h'] + bdg_crest['prod_v']\
  + bdg_crest['dissip'] + bdg_crest['canopy']

res_crest['R^h'] = 0
for item in items_h:
  res_crest['R^h'] -= bdg_crest[item]

res_crest['-T_e^v'] = 0
for item in items_vt:
  res_crest['-T_e^v'] -= bdg_crest[item]

res_crest['-pi_e^v'] = -bdg_crest['pturb_v']

res_crest['-A_e^v'] = -bdg_crest['adv_v']

res_crest['R'] = res_crest['R^h'] + res_crest['-T_e^v']\
  + res_crest['-pi_e^v'] + res_crest['-A_e^v']

res_trough['P-(eps+eps_c)'] =\
  bdg_trough['prod_h'] + bdg_trough['prod_v']\
  + bdg_trough['dissip'] + bdg_trough['canopy']

res_trough['R^h'] = 0
for item in items_h:
  res_trough['R^h'] -= bdg_trough[item]

res_trough['-T_e^v'] = 0
for item in items_vt:
  res_trough['-T_e^v'] -= bdg_trough[item]

res_trough['-pi_e^v'] = -bdg_trough['pturb_v']

res_trough['-A_e^v'] = -bdg_trough['adv_v']

res_trough['R'] = res_trough['R^h'] + res_trough['-T_e^v']\
  + res_trough['-pi_e^v'] + res_trough['-A_e^v']

uw = uw_crest + txz_crest
duwdz = get_dphidz(uw, dz, dm)
mask_can = np.logical_and(Z_prof>=0, Z_prof<=hc+dm.dz/2)
uw_top = np.interp(hc, Z_prof, uw)
uw = uw[mask_can]
duwdz = duwdz[mask_can]
z_can = Z_prof[mask_can]
integrate = np.trapz(duwdz/dm.zi*z_can, z_can)
d0_crest = integrate/uw_top

uw = uw_trough + txz_trough
duwdz = get_dphidz(uw, dz, dm)
mask_can = np.logical_and(Z_prof>=0, Z_prof<=hc+dm.dz/2)
uw_top = np.interp(hc, Z_prof, uw)
uw = uw[mask_can]
duwdz = duwdz[mask_can]
z_can = Z_prof[mask_can]
integrate = np.trapz(duwdz/dm.zi*z_can, z_can)
d0_trough = integrate/uw_top



## Save the data
tts = ts*10 + 100
tte = te*10
fn_CTstat = fmt_CTstat.format(case=case, tts=tts, tte=tte)
np.savez(fn_CTstat,
  bdg_crest=bdg_crest, bdg_trough=bdg_trough,
  vel_crest=vel_crest, vel_trough=vel_trough,
  res_crest=res_crest, res_trough=res_trough,
  uw_crest=uw_crest, uw_trough=uw_trough,
  txz_crest=txz_crest, txz_trough=txz_trough,
  d0_crest=d0_crest, d0_trough=d0_trough,
  Z_prof=Z_prof)
