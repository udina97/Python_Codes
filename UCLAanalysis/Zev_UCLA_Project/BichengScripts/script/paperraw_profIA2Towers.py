#!/usr/bin/env python3

"""
Program: paperraw_profIA2Towers
Plot the modified IA index profiles over selected locations and all crests and troughs.
"""

### Histories:
### 5/21/2022 -- Bicheng Chen (chabby@ucla.edu) -- First created.



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
case = ('amazon_canopy_nohill', 'amazon_canopy_hill05h',
  'amazon_canopy_hill', 'amazon_canopy_hill2h', 'amazon_canopy_real')
lab_case = ('Flat', 'Hill05h', 'Hill', 'Hill2h', 'Real')
ix_tower = ((0, ), (80, 0), (80, 0), (80, 0), (None, None, None))
type_CT = ((None,), ('crest', 'trough'), ('crest', 'trough'),
  ('crest', 'trough'), ('crest', 'crest', 'trough'))
#ind_CT = ((None,), (None, None), (None, None), (None, None), (1, 34, 0))
ind_CT = ((None,), (None, None), (None, None), (None, None), (28, 183, 0))
labs_prof = (('Flat', ), ('Half', 'Half'),
  ('Idealized', 'Idealized'),
  ('Double', 'Double'),
  ('Real-ridge', 'Real-hill', 'Real-trough'))

# File format
fmt_lespath = ('/data/2/bzc/LES/amazon_2D/{case:s}', '/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_2D/{case:s}', '/data/2/bzc/LES/amazon_3D/{case:s}')
fmt_real = './data/statistics_budget_CrestTrough_alpha2_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_IAdata = './data/IA2_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_IA = './figure/paperraw/paperraw_profIA2Tower.png'
flag_IAcal = False

# Topography parameters
zpad = (5, 0.5, 0, 15, None)
amp = (0, 12.5, 25, 50, None)
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 14400
te = 32400
#te = 21600
items = ('u', 'v', 'w', 'uw', 'u2', 'v2', 'w2', 'uv', 'vw', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('dissip', 'canopy', 'adv_h', 'adv_v') 
dz_real = 2
perc = (10, 50, 90)

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.1, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X')
xspacing = 1
xspacing_minor = 0.2
yspacing = 0.5
xlim = (-0.05, 0.8)
ylim = (0, 3)



## Functions
def get_topo(x, wl, amp):
  return (amp*np.cos(2*np.pi/wl*x+np.pi)+amp)

def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = complex(0, 1) * wn[np.newaxis, :] * phi_c
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
field_dissipttl = [None for ic in range(nca)]
terms_bdg = [[dict.fromkeys(items_bdg, None) for isc in range(nsc[ic])]
  for ic in range(nca)]
uw_t = [[None for isc in range(nsc[ic])] for ic in range(nca)]
txz = [[None for isc in range(nsc[ic])] for ic in range(nca)]
dissip_ttl = [[None for isc in range(nsc[ic])] for ic in range(nca)]
d0 = [[None for isc in range(nsc[ic])] for ic in range(nca)]



## Read data and calculate the temporal mean
param = [None for ic in range(nca)]
dm = [None for ic in range(nca)]
for ic in range(nca):
  lespath = fmt_lespath[ic].format(case=case[ic])
  param[ic] = lp.lesParam.lesClass.param(lespath)
  dm[ic] = param[ic].domain

for ic in range(nca):
  if lab_case[ic] != 'Real':
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
        d0[ic][isc] = dset['d0_crest'][ind_CT[ic][isc]]
      elif type_CT[ic][isc] == 'trough':
        terms_bdg[ic][isc] = dset['bdg_trough'][ind_CT[ic][isc]]
        d0[ic][isc] = dset['d0_trough'][ind_CT[ic][isc]]
      dissip_ttl[ic][isc] = terms_bdg[ic][isc]['dissip']\
        + terms_bdg[ic][isc]['canopy']



## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
    
    z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
    z_gd[ic] -= zpad[ic]
    Z_gd[ic] = z_gd[ic]-z_tpg[ic]



wn = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    wn[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)

for ic in range(nca):
  if lab_case[ic] != 'Real':
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

    # Dissipation
    field_bdg[ic]['canopy'] = (data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz'])
    field_bdg[ic]['canopy'] = wnode2uvpnode(field_bdg[ic]['canopy'])
    field_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
      +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
    field_bdg[ic]['dissip'] = data[ic]['dissip']\
      - data[ic]['txx']*S11 - data[ic]['tyy']*S22 - data[ic]['tzz']*S33\
      - 2*data[ic]['txy']*S12 - 2*data[ic]['txz']*S13 - 2*data[ic]['tyz']*S23
    field_dissipttl[ic] = data[ic]['dissip'] + field_bdg[ic]['canopy']

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
  if lab_case[ic] != 'Real':
    field_dissipttl[ic] = gaussian_filter(field_dissipttl[ic], sigma=(1, 2),
      mode='wrap')
    for item in items_bdg:
      field_bdg[ic][item] = gaussian_filter(field_bdg[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_bdg:
      field_bdg[ic][item] = np.ma.array(field_bdg[ic][item], mask=mask[ic])



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'Real':
    for isc in range(nsc[ic]):
      dissip_ttl[ic][isc] = get_vprof(field_dissipttl[ic], ix_tower[ic][isc],
        amp[ic])
      for item in items_bdg:
        terms_bdg[ic][isc][item] = get_vprof(field_bdg[ic][item],
          ix_tower[ic][isc], amp[ic])



## Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_bdg:
      terms_bdg[ic][isc][item] /= -dissip_ttl[ic][isc]



## Calculate the IA term
term_IA = [[None for isc in range(nsc[ic])] for ic in range(nca)]
for ic in range(nca):
  for isc in range(nsc[ic]):
    term_IA[ic][isc] =\
      np.abs(terms_bdg[ic][isc]['adv_h'] + terms_bdg[ic][isc]['adv_v'])



## Calculate the statistics of IA for all crests and troughs
# Calculate the IA index
if flag_IAcal:
  n_crest = len(dset['bdg_crest'])
  nz_prof = dset['Z_prof'].shape[0]
  IA_crest = np.zeros((n_crest, nz_prof))
  for ind in range(n_crest):
    print('### IA_crest', ind)
    IA_crest[ind, :] = (np.abs(dset['bdg_crest'][ind]['adv_h']
      + dset['bdg_crest'][ind]['adv_v'])) \
      / (-dset['bdg_crest'][ind]['dissip'] - dset['bdg_crest'][ind]['canopy'])
  
  n_trough = len(dset['bdg_trough'])
  nz_prof = dset['Z_prof'].shape[0]
  IA_trough = np.zeros((n_trough, nz_prof))
  for ind in range(n_trough):
    print('### IA_trough', ind)
    IA_trough[ind, :] = (np.abs(dset['bdg_trough'][ind]['adv_h']
      + dset['bdg_trough'][ind]['adv_v'])) \
      / (-dset['bdg_trough'][ind]['dissip'] - dset['bdg_trough'][ind]['canopy'])

  fn_IAdata = fmt_IAdata.format(case=case[-1], tts=ts*10+100, tte=te*10)
  np.savez(fn_IAdata, n_crest=n_crest, IA_crest=IA_crest,
      n_trough=n_trough, IA_trough=IA_trough, nz_prof=nz_prof)
else:
  fn_IAdata = fmt_IAdata.format(case=case[-1], tts=ts*10+100, tte=te*10)
  dset = np.load(fn_IAdata, allow_pickle=True)
  n_crest = dset['n_crest']
  IA_crest = dset['IA_crest']
  n_trough = dset['n_trough']
  IA_trough = dset['IA_trough']
  nz_prof = dset['nz_prof']

# Calculate the percentiles and transform them to errors
perc_crest = np.zeros((len(perc), nz_prof))
for iz in range(nz_prof):
  perc_crest[:, iz] = np.percentile(IA_crest[:, iz], perc, overwrite_input=True)
perc_crest[0, :] = perc_crest[1, :] - perc_crest[0, :]
perc_crest[2, :] = perc_crest[2, :] - perc_crest[1, :]

perc_trough = np.zeros((len(perc), nz_prof))
for iz in range(nz_prof):
  perc_trough[:, iz] =\
    np.percentile(IA_trough[:, iz], perc, overwrite_input=True)
perc_trough[0, :] = perc_trough[1, :] - perc_trough[0, :]
perc_trough[2, :] = perc_trough[2, :] - perc_trough[1, :]



## Plot the budget normalized by u* and hc
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

## Plot the Residual terms normalized by dissipation rate
xmaxLocator_minor = MultipleLocator(xspacing_minor)
ifig = 0
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, 2)
gs.update(**sz)

# Plot the IA index
isub = 0
ax = fig.add_subplot(gs[isub])

it = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    # Plot the budget terms
    if type_CT[ic][isc] == 'crest' or type_CT[ic][isc] == None:
      it += 1
      if lab_case[ic] != 'Real':
        ax.plot(term_IA[ic][isc], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_prof[ic][isc])
        fn_npz = "./data/forpaper/fig9_profIA_crest_{lab:s}".format(lab=labs_prof[ic][isc])
        np.savez(fn_npz, z=Z_gd[ic][:, ix_tower[ic][isc]]/hc, IA=term_IA[ic][isc])
      else:
        ax.plot(term_IA[ic][isc], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_prof[ic][isc])
        fn_npz = "./data/forpaper/fig9_profIA_crest_{lab:s}".format(lab=labs_prof[ic][isc])
        np.savez(fn_npz, z=Z_prof/hc, IA=term_IA[ic][isc])

ax.axhline(y=1, lw=lw, ls='--', color='k')
ax.axhline(y=2, lw=lw, ls='-.', color='k')
#ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
ax.axvline(x=0, lw=lw, ls='--', color='gray')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$IA$')
ax.set_ylabel('$Z/h_c$')

ax.legend(loc='lower right', fontsize='x-small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
# Plot the IA index
isub += 1
ax = fig.add_subplot(gs[isub])

it = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    # Plot the budget terms
    if type_CT[ic][isc] == 'trough' or type_CT[ic][isc] == None:
      it += 1
      if lab_case[ic] != 'Real':
        ax.plot(term_IA[ic][isc], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_prof[ic][isc])
        fn_npz = "./data/forpaper/fig9_profIA_trough_{lab:s}".format(lab=labs_prof[ic][isc])
        np.savez(fn_npz, z=Z_gd[ic][:, ix_tower[ic][isc]]/hc, IA=term_IA[ic][isc])
      else:
        ax.plot(term_IA[ic][isc], Z_prof/hc, lw=lw,
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=labs_prof[ic][isc])
        fn_npz = "./data/forpaper/fig9_profIA_trough_{lab:s}".format(lab=labs_prof[ic][isc])
        np.savez(fn_npz, z=Z_prof/hc, IA=term_IA[ic][isc])

ax.axhline(y=1, lw=lw, ls='--', color='k')
ax.axhline(y=2, lw=lw, ls='-.', color='k')
#ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
ax.axvline(x=0, lw=lw, ls='--', color='gray')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$IA$')
ax.set_yticklabels([])

ax.legend(loc='lower right', fontsize='x-small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
# Plot the percentiles of IA
isub += 1
ax = fig.add_subplot(gs[isub])

it = 0
ax.errorbar(perc_crest[1, :], Z_prof/hc, xerr=perc_crest[[0, 2], :], lw=lw,
  ls='-', marker='o', color='C00', capsize=2*lw, elinewidth=lw,
  markevery=(np.mod(it, 5), 5))
fn_npz = "./data/forpaper/fig9_percentileIA_crest_Real"
np.savez(fn_npz, z=Z_prof/hc, perc=perc, IA_crest=perc_crest)

ax.axhline(y=1, lw=lw, ls='--', color='k')
ax.axhline(y=2, lw=lw, ls='-.', color='k')
#ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
ax.axvline(x=0, lw=lw, ls='--', color='gray')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$IA$')
ax.set_ylabel('$Z/h_c$')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the percentiles of IA
isub += 1
ax = fig.add_subplot(gs[isub])

it = 0
ax.errorbar(perc_trough[1, :], Z_prof/hc, xerr=perc_trough[[0, 2], :], lw=lw,
  ls='-', marker='o', color='C00', capsize=2*lw, elinewidth=lw,
  markevery=(np.mod(it, 5), 5))
fn_npz = "./data/forpaper/fig9_percentileIA_trough_Real"
np.savez(fn_npz, z=Z_prof/hc, perc=perc, IA_trough=perc_trough)

ax.axhline(y=1, lw=lw, ls='--', color='k')
ax.axhline(y=2, lw=lw, ls='-.', color='k')
#ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
ax.axvline(x=0, lw=lw, ls='--', color='gray')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$IA$')
ax.set_yticklabels([])

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Save figure
fig.savefig(fn_IA, dpi=600)
