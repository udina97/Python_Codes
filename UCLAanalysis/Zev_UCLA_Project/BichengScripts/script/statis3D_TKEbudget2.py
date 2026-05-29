#!/usr/bin/env python3

"""
Program: statis3D_TKEbudget2
Calculate the TKE buget terms from 3D space to 1D profile
"""

### Histories:
### 2020/05/13 -- Bicheng Chen (chabby@berkeley.edu) -- First create



## Prerequisite Module
import numpy as np
import lespy as lp



## User-specified Variable
# File
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fmt_i = './data/statistics3D_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_ptb = './data/statistics_pertubation_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_drv = './data/statistics_derivative_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_bdg = './data/statistics_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'

# Data range
case = 'amazon_canopy_real'
tts = 144100
tte = 324000
#tte = 216000
#tte = 180000

# Items
items_i = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uv', 'uw', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')



## Function
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = complex(0, 1) * wn[np.newaxis, np.newaxis, :] * phi_c
  dphidx_c[:, :, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-2, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=-2, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1, :] = (phi[1:, :]-phi[:-1, :]) / (dm.dz/dm.zi)
  dphidz[-1, :] = dphidz[-2, :]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[0, :, :] = 0
  phi_h[1:, :, :] = 0.5*(phi_c[:-1, :, :] + phi_c[1:, :, :])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:-1, :, :] = 0.5*(phi_h[:-1, :, :]+phi_h[1:, :, :])
  phi_c[-1, :, :] = phi_c[-2, :, :]
  return phi_c



## Read the data
# Read statistics
fn = fmt_i.format(case=case, tts=tts, tte=tte)
print("Reading the basic statistics from {:s}...".format(fn))
dset = np.load(fn, allow_pickle=True)
#dset = np.load(fn)
x_gd = dset['x']
y_gd = dset['y']
z_gd = dset['z']
x_tpg = dset['x_tpg']
y_tpg = dset['y_tpg']
z_tpg = dset['z_tpg']
data = dset['data'].item()

# Read LES parameters
print('Reading the setup parameters of the simulation...')
path_les = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(path_les)
dm = param.domain
dm.show()



## Get the coordinate
Z_gd = z_gd - z_tpg[np.newaxis, :, :]
mask = Z_gd<0



## Calculate the budget terms
# Initialization
terms_bdg = dict.fromkeys(items_bdg, None)
terms_ptb = dict()
terms_drv = dict()
wn_x = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
wn_y = 2*np.pi*np.fft.rfftfreq(dm.ny, dm.dy/dm.zi)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data['u'])
v_h = uvpnode2wnode(data['v'])

# Calculate the pertubation (all on uvp-nodes)
#print(data['u2'][:50, 0, 0])
#print(data['uv'][:50, 0, 0])
#print(data['u'][:50, 0, 0])
#print(data['v'][:50, 0, 0])
terms_ptb['u2_t'] = data['u2'] - data['u']**2
terms_ptb['uv_t'] = data['uv'] - data['u']*data['v']
terms_ptb['uw_t'] = data['uw'] - u_h*data['w']
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data['v2'] - data['v']**2
terms_ptb['vw_t'] = data['vw'] - v_h*data['w']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data['w2'] -data['w']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2

# Calculate the derivatives (all on uvp-nodes)
terms_drv['dudx'] = get_dphidx(data['u'], wn_x)
terms_drv['dudy'] = get_dphidy(data['u'], wn_y)
terms_drv['dudz'] = get_dphidz(data['u'], dm)
terms_drv['dudz'] = wnode2uvpnode(terms_drv['dudz'])

terms_drv['dvdx'] = get_dphidx(data['v'], wn_x)
terms_drv['dvdy'] = get_dphidy(data['v'], wn_y)
terms_drv['dvdz'] = get_dphidz(data['v'], dm)
terms_drv['dvdz'] = wnode2uvpnode(terms_drv['dvdz'])

terms_drv['dwdx'] = get_dphidx(data['w'], wn_x)
terms_drv['dwdx'] = wnode2uvpnode(terms_drv['dwdx'])
terms_drv['dwdy'] = get_dphidy(data['w'], wn_y)
terms_drv['dwdy'] = wnode2uvpnode(terms_drv['dwdy'])
terms_drv['dwdz'] = get_dphidz(data['w'], dm)

# Calculate the advection term (all on uvp-nodes)
ue = data['u']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
ve = data['v']*terms_ptb['tke']
dvedy = get_dphidy(ve, wn_y)
tkez = uvpnode2wnode(terms_ptb['tke'])
we = data['w']*tkez
dwedz = get_dphidz(we, dm)
terms_bdg['adv_h'] = -duedx - dvedy
terms_bdg['adv_v'] = -dwedz

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
uw_c = wnode2uvpnode(data['uw'])
vw_c = wnode2uvpnode(data['vw'])
w_c = wnode2uvpnode(data['w'])
w2_c = wnode2uvpnode(data['w2'])

terms_ptb['u3_t'] = data['u3'] - 3*data['u']*data['u2'] + 2*data['u']**3
terms_ptb['uv2_t'] = data['uv2'] - 2*data['v']*data['uv']\
  + 2*data['u']*data['v']**2 - data['u']*data['v2']
terms_ptb['uw2_t'] = data['uw2'] - 2*w_c*uw_c + 2*data['u']*w_c**2\
  - data['u']*w2_c
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data['utxx'] - data['u']*data['txx']
terms_ptb['vtxy_t'] = data['vtxy'] - data['v']*data['txy']
terms_ptb['wtxz_t'] = data['wtxz'] - data['w']*data['txz']
terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(terms_ptb['utxx_t']+terms_ptb['vtxy_t']
  +terms_ptb['wtxz_t'], wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

terms_ptb['u2v_t'] = data['u2v'] - 2*data['u']*data['uv']\
  + 2*data['v']*data['u']**2 - data['v']*data['u2']
terms_ptb['v3_t'] = data['v3'] - 3*data['v']*data['v2'] + 2*data['v']**3
terms_ptb['vw2_t'] = data['vw2'] - 2*w_c*vw_c + 2*data['v']*w_c**2\
  - data['v']*w2_c
ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
dvedy = get_dphidy(ve, wn_y)
terms_ptb['utxy_t'] = data['utxy'] - data['u']*data['txy']
terms_ptb['vtyy_t'] = data['vtyy'] - data['v']*data['tyy']
terms_ptb['wtyz_t'] = data['wtyz'] - data['w']*data['tyz']
terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
dutaudy = get_dphidy(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
  +terms_ptb['wtyz_t'], wn_y)
terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
u2_h = uvpnode2wnode(data['u2'])
v2_h = uvpnode2wnode(data['v2'])
terms_ptb['wu2_t'] = data['u2w'] - 2*u_h*data['uw'] + 2*data['w']*u_h**2\
  - data['w']*u2_h
terms_ptb['wv2_t'] = data['v2w'] - 2*v_h*data['vw'] + 2*data['w']*v_h**2\
  - data['w']*v2_h
terms_ptb['w3_t'] = data['w3'] - 3*data['w']*data['w2'] + 2*data['w']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dm)
terms_ptb['utxz_t'] = data['utxz'] - u_h*data['txz']
terms_ptb['vtyz_t'] = data['vtyz'] - v_h*data['tyz']
terms_ptb['wtzz_t'] = data['wtzz'] - data['w']*data['tzz']
dutaudz = get_dphidz(terms_ptb['utxz_t']+terms_ptb['vtyz_t']
  +terms_ptb['wtzz_t'], dm)
terms_bdg['uturb_v'] = -dwedz-dutaudz

# Calculate the pressure transport term (all on uvp-nodes)
terms_ptb['pu_t'] = data['pu'] - data['p']*data['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

terms_ptb['pv_t'] = data['pv'] - data['p']*data['v']
dpvdy = get_dphidx(terms_ptb['pv_t'], wn_y)

p_h = uvpnode2wnode(data['p'])
terms_ptb['pw_t'] = data['pw'] - p_h*data['w']
dpwdz = get_dphidz(terms_ptb['pw_t'], dm)

terms_bdg['pturb_h'] = -dpudx-dpvdy
terms_bdg['pturb_v'] = -dpwdz

# Calculate the dissipation rate (all on uvp-nodes)
terms_drv['S11'] = terms_drv['dudx']
terms_drv['S12'] = 0.5*(terms_drv['dudy'] + terms_drv['dvdx'])
terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
terms_drv['S22'] = terms_drv['dvdy']
terms_drv['S23'] = 0.5*(terms_drv['dvdz'] + terms_drv['dwdy'])
terms_drv['S33'] = terms_drv['dwdz']

terms_bdg['dissip'] = data['dissip']\
  - data['txx']*terms_drv['S11'] - data['tyy']*terms_drv['S22']\
  - data['tzz']*terms_drv['S33']\
  - 2*data['txy']*terms_drv['S12'] - 2*data['txz']*terms_drv['S13']\
  - 2*data['tyz']*terms_drv['S23']

terms_bdg['canopy'] = data['wFcz'] - data['w']*data['Fcz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data['uFcx']-data['u']*data['Fcx'])\
  +(data['vFcy']-data['v']*data['Fcy'])

# Calculate the production (all on uvp-nodes)
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] + terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] + terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] + terms_ptb['vw_t']*terms_drv['dwdy']
  #+ data['txx']*terms_drv['S11'] + data['tyy']*terms_drv['S22']
  #+ 2*data['txy']*terms_drv['S12'] + data['txz']*terms_drv['dwdx']
  #+ data['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  #+ data['txz']*terms_drv['dudz'] + data['tyz']*terms_drv['dvdz']
  #+ data['tzz']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']
print('*'*80)
#print(terms_bdg['prod_h'][:, 176, 176]/terms_bdg['prod_v'][:, 176, 176])

# Save the data
fn = fmt_ptb.format(case=case, tts=tts, tte=tte)
np.savez(fn, terms=terms_ptb)
fn = fmt_drv.format(case=case, tts=tts, tte=tte)
np.savez(fn, terms=terms_drv)
fn = fmt_bdg.format(case=case, tts=tts, tte=tte)
np.savez(fn, terms=terms_bdg)
