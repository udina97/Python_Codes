#!/usr/bin/env python3

"""
Program: paperraw_profTKEbudgetTowers
Plot the TKE budget profiles over crests and troughs. The terms are normalized by the friction velocity and the canopy height.
"""

### Histories:
### 7/31/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created.



## Prerequisite Module
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
case = ('amazon_canopy_nohill', 'amazon_canopy_hill', 'amazon_canopy_real')
lab_case = ('flat', 'hill', 'real')
ix_tower = ((0, ), (80, 0), (None, None, None))
type_CT = ((None,), ('crest', 'trough'), ('crest', 'crest', 'trough'))
#ind_CT = ((None,), (None, None), (1, 4, 0))
#ind_CT = ((None,), (None, None), (1, 34, 0))
ind_CT = ((None,), (None, None), (28, 183, 0))
#title_case = (('Flat', ), ('Hill crest', 'Hill trough'),
#  ('Real crest #1', 'Real crest #2', 'Real trough #1'))
title_case = (('flat', ), ('idealized crest', 'idealized trough'),
  ('real ridge', 'real hill', 'real trough'))

# File format
fmt_lespath = ('/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_2D/{case:s}', '/data/2/bzc/LES/amazon_3D/{case:s}')
fmt_real = './data/statistics_budget_CrestTrough_alpha2_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_mp = './data/meanProf_{case:s}_tt{tts:08d}-{tte:08d}.npz'
flag_outflat = False
fn_flat = './data/meanProf_flat.npz'
fn_bdg1 = './figure/paperraw/paperraw_profTKEbudgetByCanopy.png'
fn_bdg2 = './figure/paperraw/paperraw_profTKEbudgetByDissip.png'
fn_res1 = './figure/paperraw/paperraw_profResidualByCanopy.png'
fn_res2 = './figure/paperraw/paperraw_profResidualByDissip.png'

# Topography parameters
zpad = (5, 0, None)
amp = (0, 25, None)
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 14400
te = 32400
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
#items_bdg = ('prod_h', 'prod_v', 'dissip', 'canopy', 'adv_h', 'adv_v',
#  'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'sum')
#labs_bdg = (r'$P^h$', r'$P^v$', r'$\epsilon$', r'$\epsilon_c$', r'$A^h_e$',
#  r'$A^v_e$', r'$T^h_e$', r'$T^v_e$', r'$\Pi^h_e$', r'$\Pi^v_e$', 'sum')
items_bdg = ('prod', 'dissip', 'canopy', 'adv', 'uturb', 'pturb')
labs_bdg = (r'$P$', r'$\epsilon$', r'$\epsilon_c$', r'$A_e$',
  r'$T_e$', r'$\Pi_e$')
items_h = ('adv_h', 'uturb_h', 'pturb_h')
items_vt = ('uturb_v',)
items_vo = ('adv_v', 'pturb_v')
#items_R = ('P-(eps+eps_c)', 'R^h', '-T_e^v', '-pi_e^v', '-A_e^v', 'R')
#labs_R = (r'$P-(\epsilon+\epsilon_c)$', r'$R^h$',
#  r'$-T_e^v$', r'$-\Pi_e^v$', r'$-A_e^v$', r'$R$')
items_R = ('R', 'R^h', '-T_e^v', '-pi_e^v', '-A_e^v')
labs_R = (r'$R$', r'$R^h$', r'$-T_e^v$', r'$-\Pi_e^v$', r'$-A_e^v$')
dz_real = 2

# Figure
sz = dict(left=0.07, right=0.89, bottom=0.1, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X')
xspacing = 1
xspacing_minor = 0.5
yspacing = 0.5
xlim_bdg1 = (-5, 7)
xlim_bdg2 = (-1.1, 2.6)
xlim_R1 = (-3, 5)
xlim_R2 = (-1.2, 2)
ylim = (0, 3)



## Functions
def get_topo(x, wl, amp):
  return (amp*np.cos(2*np.pi/wl*x+np.pi)+amp)

def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = np.complex(0, 1) * wn[np.newaxis, :] * phi_c
  dphidx_c[:, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1, :] = (phi[1:, :]-phi[:-1, :]) / (dm.dz/dm.zi)
  dphidz[-1, :] = dphidz[-2, :]
  return dphidz

def get_dphidzReal(phi, dz, dm):
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

def get_vprof(term, ix, amp):
  if amp==0:
    prof = term.mean(axis=-1)
    return prof
  else:
    prof = term[:, ix] 
    return prof



## Initialization
nca = len(case) # number of case
nsc = [len(ix_tower[ind]) for ind in range(nca)] # number of sub case
nttl = np.sum(nsc)
data = [{item: None for item in items} for ic in range(nca)]
tke = [[None for isc in range(nsc[ic])] for ic in range(nca)]
res = [[None for isc in range(nsc[ic])] for ic in range(nca)]
ratio = [[None for isc in range(nsc[ic])] for ic in range(nca)]
field_bdg = [dict.fromkeys(items_bdg, None) for ic in range(nca)]
field_R = [dict.fromkeys(items_R, None) for ic in range(nca)]
field_dissipttl = [None for ic in range(nca)]
terms_bdg = [[dict.fromkeys(items_bdg, None) for isc in range(nsc[ic])]
  for ic in range(nca)]
terms_R = [[dict.fromkeys(items_R, None) for isc in range(nsc[ic])]
  for ic in range(nca)]
uw_t = [[None for isc in range(nsc[ic])] for ic in range(nca)]
txz = [[None for isc in range(nsc[ic])] for ic in range(nca)]
dissip_ttl = [[None for isc in range(nsc[ic])] for ic in range(nca)]
d0 = [[None for isc in range(nsc[ic])] for ic in range(nca)]



## Read the mean profiles
meanProf = [None for ic in range(nca)]
Z_mp = [None for ic in range(nca)]
for ic in range(nca):
  fn_mp = fmt_mp.format(case=case[ic], tts=ts*10+100, tte=te*10)
  dset = np.load(fn_mp, allow_pickle=True)
  meanProf[ic] = dset['terms'].item()['prod']
  Z_mp[ic] = dset['Z_prof'] / hc



## Read data and calculate the temporal mean
param = [None for ic in range(nca)]
dm = [None for ic in range(nca)]
for ic in range(nca):
  lespath = fmt_lespath[ic].format(case=case[ic])
  param[ic] = lp.lesParam.lesClass.param(lespath)
  dm[ic] = param[ic].domain

for ic in range(nca):
  if lab_case[ic] != 'real':
    for item in items:
      data[ic][item], time = lp.io.io_averFile.loadLES_averFile(param[ic],
        qtype=item, tss=ts, tes=te)
      data[ic][item] = np.mean(data[ic][item], axis=0)
  else:
    fn_real = fmt_real.format(case=case[ic], tts=ts*10+100, tte=te*10)
    dset = np.load(fn_real, allow_pickle=True)
    Z_prof = dset['Z_prof']
    for isc in range(nsc[ic]):
      if type_CT[ic][isc] == 'crest':
        terms_bdg[ic][isc] = dset['bdg_crest'][ind_CT[ic][isc]]
        terms_R[ic][isc] = dset['res_crest'][ind_CT[ic][isc]]
        d0[ic][isc] = dset['d0_crest'][ind_CT[ic][isc]]
      elif type_CT[ic][isc] == 'trough':
        terms_bdg[ic][isc] = dset['bdg_trough'][ind_CT[ic][isc]]
        terms_R[ic][isc] = dset['res_trough'][ind_CT[ic][isc]]
        d0[ic][isc] = dset['d0_trough'][ind_CT[ic][isc]]
      dissip_ttl[ic][isc] = terms_bdg[ic][isc]['dissip']\
        + terms_bdg[ic][isc]['canopy']



## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'real':
    z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
    
    z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
    z_gd[ic] -= zpad[ic]
    Z_gd[ic] = z_gd[ic]-z_tpg[ic]



wn = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'real':
    wn[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)

for ic in range(nca):
  if lab_case[ic] != 'real':
    # Calculate the production
    data[ic]['w'] = gaussian_filter(data[ic]['w'], sigma=(0, 1), mode='wrap')
    data[ic]['dwdx'] = get_dphidx(data[ic]['w'], wn[ic])
    data[ic]['dwdx'] = gaussian_filter(data[ic]['dwdx'], sigma=(1, 1),
      mode='wrap')
    
    #data[ic]['u'] = gaussian_filter(data[ic]['u'], sigma=(0, 1), mode='wrap')
    data[ic]['dudx'] = get_dphidx(data[ic]['u'], wn[ic])
    data[ic]['dudx'] = uvpnode2wnode(data[ic]['dudx'])
    data[ic]['dudx'] = gaussian_filter(data[ic]['dudx'], sigma=(1, 1),
      mode='wrap')
    
    data[ic]['dudz'] = np.zeros(data[ic]['u'].shape) 
    data[ic]['dudz'][1:, :] =\
      (data[ic]['u'][1:, :]-data[ic]['u'][:-1, :]) / (dm[ic].dz/dm[ic].zi)
    data[ic]['dudz'] = gaussian_filter(data[ic]['dudz'], sigma=(1, 1),
      mode='wrap')

    data[ic]['dwdz'] = np.zeros(data[ic]['w'].shape) 
    data[ic]['dwdz'] = uvpnode2wnode(data[ic]['dwdz'])
    data[ic]['dwdz'] = gaussian_filter(data[ic]['dwdz'], sigma=(1, 1),
      mode='wrap')
    
    data[ic]['uw_t'] = data[ic]['uw'] - data[ic]['u']*data[ic]['w']
    data[ic]['uw_t'] = gaussian_filter(data[ic]['uw_t'], sigma=(1, 1),
      mode='wrap')
    data[ic]['uu_t'] = data[ic]['u2'] - data[ic]['u']**2
    data[ic]['uu_t'] = gaussian_filter(data[ic]['uu_t'], sigma=(1, 1),
      mode='wrap')
    data[ic]['vv_t'] = data[ic]['v2'] - data[ic]['v']**2
    data[ic]['vv_t'] = gaussian_filter(data[ic]['vv_t'], sigma=(1, 1),
      mode='wrap')
    data[ic]['ww_t'] = data[ic]['w2'] - data[ic]['w']**2
    data[ic]['ww_t'] = gaussian_filter(data[ic]['ww_t'], sigma=(1, 1),
      mode='wrap')

    # TKE
    tke[ic] = (data[ic]['uu_t'] + data[ic]['vv_t'] + data[ic]['ww_t']) / 2

    # Production
    field_bdg[ic]['prod_h'] = -data[ic]['uu_t']*data[ic]['dudx']\
      - data[ic]['uw_t']*data[ic]['dwdx']
    field_bdg[ic]['prod_v'] = - data[ic]['ww_t']*data[ic]['dwdz']\
      - data[ic]['uw_t']*data[ic]['dudz']
    field_bdg[ic]['prod'] =\
      field_bdg[ic]['prod_h'] + field_bdg[ic]['prod_v']
    field_bdg[ic]['prod_dudz'] = -data[ic]['uw_t']*data[ic]['dudz']

    # Strain terms
    data[ic]['dvdx'] = get_dphidx(data[ic]['v'], wn[ic])
    data[ic]['dvdx'] = uvpnode2wnode(data[ic]['dvdx'])
    data[ic]['dvdx'] = gaussian_filter(data[ic]['dvdx'], sigma=(1, 1), mode='wrap')
    
    data[ic]['dvdz'] = np.zeros(data[ic]['v'].shape) 
    data[ic]['dvdz'][1:, :] =\
      (data[ic]['v'][1:, :]-data[ic]['v'][:-1, :]) / (dm[ic].dz/dm[ic].zi)
    data[ic]['dvdz'] = gaussian_filter(data[ic]['dvdz'], sigma=(1, 1), mode='wrap')
    ux = data[ic]['dudx']
    uz = data[ic]['dudz']
    vx = data[ic]['dvdx']
    vz = data[ic]['dvdz']
    wx = data[ic]['dwdx']
    wz = data[ic]['dwdz']

    S11 = ux
    S12 = 0.5 * vx
    S13 = 0.5 * (uz + wx)
    S22 = 0
    S23 = 0.5 * vz
    S33 = wz
    
    # Advection term
    ue = data[ic]['u']*tke[ic]
    duedx = get_dphidx(ue, wn[ic])
    tkez = np.zeros(tke[ic].shape)
    tkez[0, :] = 0
    tkez[1:, :] = 0.5*(tke[ic][:-1, :] + tke[ic][1:, :])
    we = data[ic]['w']*tkez
    dwedz = get_dphidz(we, dm[ic])
    field_bdg[ic]['adv_h'] = -duedx
    field_bdg[ic]['adv_v'] = -dwedz
    field_bdg[ic]['adv'] =\
      field_bdg[ic]['adv_h'] + field_bdg[ic]['adv_v']

    # Turbulent transport term in horizontal
    turb_u3 = data[ic]['u3'] - 3*data[ic]['u']*data[ic]['u2']+2*data[ic]['u']**3
    turb_uv2 = data[ic]['uv2'] - 2*data[ic]['v']*data[ic]['uv']\
      + 2*data[ic]['u']*data[ic]['v']**2-data[ic]['u']*data[ic]['v2']
    uw = wnode2uvpnode(data[ic]['uw'])
    w_c = wnode2uvpnode(data[ic]['w'])
    w2_c = wnode2uvpnode(data[ic]['w2'])
    turb_uw2 = data[ic]['uw2'] - 2*w_c*uw\
      + 2*data[ic]['u']*w_c**2-data[ic]['u']*w2_c
    ue = 0.5*(turb_u3 + turb_uv2 + turb_uw2)
    duedx = get_dphidx(ue, wn[ic])
    utxx = data[ic]['utxx'] - data[ic]['u']*data[ic]['txx']
    vtxy = data[ic]['vtxy'] - data[ic]['v']*data[ic]['txy']
    wtxz = data[ic]['wtxz'] - data[ic]['w']*data[ic]['txz']
    wtxz_uv = wnode2uvpnode(wtxz)
    dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn[ic])
    field_bdg[ic]['uturb_h'] = -duedx-dutaudx

    # Turbulent transport term in vertical
    u_w = uvpnode2wnode(data[ic]['u'])
    u2_w = uvpnode2wnode(data[ic]['u2'])
    turb_wu2 = data[ic]['u2w'] - 2*u_w*data[ic]['uw']\
      + 2*data[ic]['w']*u_w**2 - data[ic]['w']*u2_w
    v_w = uvpnode2wnode(data[ic]['v'])
    v2_w = uvpnode2wnode(data[ic]['v2'])
    turb_wv2 = data[ic]['v2w'] - 2*v_w*data[ic]['vw']\
      + 2*data[ic]['w']*v_w**2 - data[ic]['w']*v2_w
    turb_w3 = data[ic]['w3'] - 3*data[ic]['w']*data[ic]['w2']\
      + 2*data[ic]['w']**3
    we = 0.5*(turb_wu2 + turb_wv2 + turb_w3)
    dwedz = get_dphidz(we, dm[ic])
    utxz = data[ic]['utxz'] - u_w*data[ic]['txz']
    vtyz = data[ic]['vtyz'] - v_w*data[ic]['tyz']
    wtzz = data[ic]['wtzz'] - data[ic]['w']*data[ic]['tzz']
    dutaudz = get_dphidz(utxz+vtyz+wtzz, dm[ic])
    field_bdg[ic]['uturb_v'] = -dwedz-dutaudz
    field_bdg[ic]['uturb'] =\
      field_bdg[ic]['uturb_h'] + field_bdg[ic]['uturb_v']

    # Pressure transport term
    pu = data[ic]['pu'] - data[ic]['p']*data[ic]['u']
    dpudx = get_dphidx(pu, wn[ic])
    p_w = uvpnode2wnode(data[ic]['p'])
    pw = data[ic]['pw'] - p_w*data[ic]['w']
    dpwdz = get_dphidz(pw, dm[ic])
    field_bdg[ic]['pturb_h'] = -dpudx
    field_bdg[ic]['pturb_v'] = -dpwdz
    field_bdg[ic]['pturb'] =\
      field_bdg[ic]['pturb_h'] + field_bdg[ic]['pturb_v']

    # Dissipation
    field_bdg[ic]['canopy'] = (data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz'])
    field_bdg[ic]['canopy'] = wnode2uvpnode(field_bdg[ic]['canopy'])
    field_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
      +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
    field_bdg[ic]['dissip'] = data[ic]['dissip']\
      - data[ic]['txx']*S11 - data[ic]['tyy']*S22 - data[ic]['tzz']*S33\
      - 2*data[ic]['txy']*S12 - 2*data[ic]['txz']*S13 - 2*data[ic]['tyz']*S23
    field_dissipttl[ic] = data[ic]['dissip'] + field_bdg[ic]['canopy']

    # Sum of above terms
    field_bdg[ic]['sum'] = field_bdg[ic]['prod_h'] + field_bdg[ic]['prod_v']\
      + field_bdg[ic]['adv_h'] + field_bdg[ic]['adv_v']\
      + field_bdg[ic]['uturb_h'] + field_bdg[ic]['uturb_v']\
      + field_bdg[ic]['pturb_h'] + field_bdg[ic]['pturb_v']\
      + field_bdg[ic]['canopy'] + field_bdg[ic]['dissip'] 

    # Residual terms
    field_R[ic]['P-(eps+eps_c)'] = field_bdg[ic]['prod_h']\
      + field_bdg[ic]['prod_v'] + field_bdg[ic]['dissip']\
      + field_bdg[ic]['canopy']

    field_R[ic]['R^h'] = 0
    for item in items_h:
      field_R[ic]['R^h'] -= field_bdg[ic][item]

    field_R[ic]['-T_e^v'] = 0
    for item in items_vt:
      field_R[ic]['-T_e^v'] -= field_bdg[ic][item]

    field_R[ic]['-pi_e^v'] = -field_bdg[ic]['pturb_v']

    field_R[ic]['-A_e^v'] = -field_bdg[ic]['adv_v']

    field_R[ic]['R'] = field_R[ic]['R^h'] + field_R[ic]['-T_e^v']\
      + field_R[ic]['-pi_e^v'] + field_R[ic]['-A_e^v']

    # uw and displacement height
    u_w = uvpnode2wnode(data[ic]['u']) 
    uw_2D = data[ic]['uw'] - u_w*data[ic]['w'] + data[ic]['txz']
    for isc in range(nsc[ic]):
      duwdz = get_dphidz(uw_2D, dm[ic])
      if amp[ic]==0:
        uw = uw_2D.mean(axis=-1)
        duwdz = duwdz.mean(axis=-1)
      else:
        uw = uw_2D[:, ix_tower[ic][isc]]
        duwdz = duwdz[:, ix_tower[ic][isc]]
      mask_can = np.logical_and(Z_gd[ic][:, ix_tower[ic][isc]]>=0,
        Z_gd[ic][:, ix_tower[ic][isc]]<=hc+dm[ic].dz/2)
      z_can = Z_gd[ic][:, ix_tower[ic][isc]][mask_can]
      uw_top = np.interp(hc, Z_gd[ic][:, ix_tower[ic][isc]]-dm[ic].dz/2, uw)
      uw = uw[mask_can]
      duwdz = duwdz[mask_can]
      integrate = np.trapz(duwdz/dm[ic].zi*z_can, z_can)
      d0[ic][isc] = integrate/uw_top



## Smooth the data
for ic in range(nca):
  if lab_case[ic] != 'real':
    field_dissipttl[ic] = gaussian_filter(field_dissipttl[ic], sigma=(1, 2),
      mode='wrap')
    for item in items_bdg:
      field_bdg[ic][item] = gaussian_filter(field_bdg[ic][item],
        sigma=(1, 2), mode='wrap')
    for item in items_R:
      field_R[ic][item] = gaussian_filter(field_R[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_bdg:
      field_bdg[ic][item] = np.ma.array(field_bdg[ic][item], mask=mask[ic])
    for item in items_R:
      field_R[ic][item] = np.ma.array(field_R[ic][item], mask=mask[ic])



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'real':
    for isc in range(nsc[ic]):
      dissip_ttl[ic][isc] = get_vprof(field_dissipttl[ic], ix_tower[ic][isc],
        amp[ic])
      for item in items_bdg:
        terms_bdg[ic][isc][item] = get_vprof(field_bdg[ic][item],
          ix_tower[ic][isc], amp[ic])
      for item in items_R:
        terms_R[ic][isc][item] = get_vprof(field_R[ic][item], ix_tower[ic][isc],
          amp[ic])
  else:
    for isc in range(nsc[ic]):
      for item in items_bdg:
        if not (item in ('dissip', 'canopy')):
          item_h = item + '_h'
          item_v = item + '_v'
          terms_bdg[ic][isc][item] =\
            terms_bdg[ic][isc][item_h] + terms_bdg[ic][isc][item_v]



## Save the flat profile
if flag_outflat:
  np.savez(fn_flat, terms=terms_bdg[0][0], Z_prof=z_gd[0][:, 0])



##Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_bdg:
      terms_bdg[ic][isc][item] *= hc/dm[ic].lz
    dissip_ttl[ic][isc] *= hc/dm[ic].lz
  meanProf[ic] *= hc/dm[ic].lz



## Plot the budget normalized by u* and hc
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, (nttl+1)//2)
gs.update(**sz)

isub = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    isub += 1
    # Plot the budget terms
    ax = fig.add_subplot(gs[isub])
    if lab_case[ic] != 'real':
      for it, item in enumerate(items_bdg):
        ax.plot(terms_bdg[ic][isc][item], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
    else:
      for it, item in enumerate(items_bdg):
        ax.plot(terms_bdg[ic][isc][item], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
    ax.plot(meanProf[ic], Z_mp[ic], lw=lw+1, ls=':', color='k',
      label=r'$\langle P\rangle$')

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
    ax.axvline(x=0, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim_bdg1)
    ax.set_ylim(ylim)

    # Axis label
    if isub >= nttl/2:
      ax.set_xlabel(
        r'$\frac{\partial\overline{e}}{\partial t}\frac{h_c}{u_*^3}$')
    else:
      ax.set_xticklabels([])

    if isub==0 or isub==nttl/2:
      ax.set_ylabel('$z/h_c$')
    else:
      ax.set_yticklabels([])

    if isub==nttl-1:
      #ax.legend(loc='upper right')
      ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    ax.set_title(title_case[ic][isc], fontsize='large')

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
fig.savefig(fn_bdg1, dpi=400)



##Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_bdg:
      terms_bdg[ic][isc][item] /= -dissip_ttl[ic][isc]



## Identify source and sink ranges
for item in items_bdg:
  print(80*'*')
  print(item, ":")
  for ic in range(nca):
    for isc in range(nsc[ic]):
      print(10*'=')
      print('Case', title_case[ic][isc])
      print('source')
      ind = terms_bdg[ic][isc][item]>=0.1
      if lab_case[ic] != 'real':
        ind2 = Z_gd[ic][:, ix_tower[ic][isc]]/hc <= 3.0
        ind = np.logical_and(ind, ind2)
        print(Z_gd[ic][:, ix_tower[ic][isc]][ind]/hc)
      else:
        ind2 = Z_prof/hc <= 3.0
        ind = np.logical_and(ind, ind2)
        print(Z_prof[ind]/hc)
      print('sink')
      ind = terms_bdg[ic][isc][item]<=-0.1
      if lab_case[ic] != 'real':
        ind2 = Z_gd[ic][:, ix_tower[ic][isc]]/hc <= 3.0
        ind = np.logical_and(ind, ind2)
        print(Z_gd[ic][:, ix_tower[ic][isc]][ind]/hc)
      else:
        ind2 = Z_prof/hc <= 3.0
        ind = np.logical_and(ind, ind2)
        print(Z_prof[ind]/hc)



## Plot the budget normalized by dissipation rate
# Initlize the figure
ifig += 1
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, (nttl+1)//2)
gs.update(**sz)

isub = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    isub += 1
    # Plot the budget terms
    ax = fig.add_subplot(gs[isub])
    if lab_case[ic] != 'real':
      for it, item in enumerate(items_bdg):
        ax.plot(terms_bdg[ic][isc][item], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
      fn_npz = "./data/forpaper/fig7_profTKEbudget_{ttl:s}".format(
        ttl=title_case[ic][isc].replace(" ", "_"))
      np.savez(fn_npz, z=Z_gd[ic][:, ix_tower[ic][isc]]/hc, budget=terms_bdg[ic][isc])
    else:
      for it, item in enumerate(items_bdg):
        ax.plot(terms_bdg[ic][isc][item], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
      fn_npz = "./data/forpaper/fig7_profTKEbudget_{ttl:s}".format(
        ttl=title_case[ic][isc].replace(" ", "_"))
      np.savez(fn_npz, z=Z_prof/hc, budget=terms_bdg[ic][isc])

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
    print(title_case[ic][isc], d0[ic][isc]/hc)
    ax.axvline(x=0, lw=lw, ls='--', color='gray')
    ax.axvline(x=1, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim_bdg2)
    ax.set_ylim(ylim)

    # Axis label
    if isub >= nttl/2:
      ax.set_xlabel(
        r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$')
    else:
      ax.set_xticklabels([])

    if isub==0 or isub==nttl/2:
      ax.set_ylabel('$z/h_c$')
    else:
      ax.set_yticklabels([])

    if isub==nttl-1:
      #ax.legend(loc='upper right')
      ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    ax.set_title(title_case[ic][isc], fontsize='large')

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
fig.savefig(fn_bdg2, dpi=400)



##Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_R:
      terms_R[ic][isc][item] *= hc/dm[ic].lz



## Plot the Residual terms normalized by u* and hc
ifig += 1
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, (nttl+1)//2)
gs.update(**sz)

# Plot the budget terms
isub = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    isub += 1
    # Plot the budget terms
    ax = fig.add_subplot(gs[isub])
    if lab_case[ic] != 'real':
      for it, item in enumerate(items_R):
        ax.plot(terms_R[ic][isc][item], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_R[it])
    else:
      for it, item in enumerate(items_R):
        ax.plot(terms_R[ic][isc][item], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_R[it])

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
    ax.axvline(x=0, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim_R1)
    ax.set_ylim(ylim)

    # Axis label
    if isub >= nttl/2:
      ax.set_xlabel(
        r'$\frac{\partial \overline{e}}{\partial t}\frac{h_c}{u_*^3}$')
    else:
      ax.set_xticklabels([])

    if isub==0 or isub==nttl/2:
      ax.set_ylabel('$z/h_c$')
    else:
      ax.set_yticklabels([])

    if isub==nttl-1:
      #ax.legend(loc='upper right')
      ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    ax.set_title(title_case[ic][isc], fontsize='large')

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
# Save figure
fig.savefig(fn_res1, dpi=600)



##Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_R:
      terms_R[ic][isc][item] /= -dissip_ttl[ic][isc]



## Plot the Residual terms normalized by dissipation rate
xmaxLocator_minor = MultipleLocator(xspacing_minor)
ifig += 1
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, (nttl+1)//2)
gs.update(**sz)

# Plot the budget terms
isub = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    isub += 1
    # Plot the budget terms
    ax = fig.add_subplot(gs[isub])
    if lab_case[ic] != 'real':
      for it, item in enumerate(items_R):
        ax.plot(terms_R[ic][isc][item], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_R[it])
      fn_npz = "./data/forpaper/fig8_profResidual_{ttl:s}".format(
        ttl=title_case[ic][isc].replace(" ", "_"))
      np.savez(fn_npz, z=Z_gd[ic][:, ix_tower[ic][isc]]/hc, budget=terms_R[ic][isc])
    else:
      for it, item in enumerate(items_R):
        ax.plot(terms_R[ic][isc][item], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_R[it])
      fn_npz = "./data/forpaper/fig8_profResidual_{ttl:s}".format(
        ttl=title_case[ic][isc].replace(" ", "_"))
      np.savez(fn_npz, z=Z_prof/hc, budget=terms_R[ic][isc])

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
    ax.axvline(x=0, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator_minor)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim_R2)
    ax.set_ylim(ylim)

    # Axis label
    if isub >= nttl/2:
      ax.set_xlabel(
        r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$')
    else:
      ax.set_xticklabels([])

    if isub==0 or isub==nttl/2:
      ax.set_ylabel('$z/h_c$')
    else:
      ax.set_yticklabels([])

    if isub==nttl-1:
      #ax.legend(loc='upper right')
      ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    ax.set_title(title_case[ic][isc], fontsize='large')

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
# Save figure
fig.savefig(fn_res2, dpi=600)
