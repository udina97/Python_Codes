#!/usr/bin/env python3

"""
Program: paperraw_profProdTowers
Plot profiles of different production components over crests and troughs. The terms are normalized by the friction velocity and the canopy height.
"""

### Histories:
### 8/10/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created.



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
case = ('amazon_canopy_hill', 'amazon_canopy_real')
lab_case = ('hill', 'real')
ix_tower = ((80, 0), (None, None, None))
type_CT = (('crest', 'trough'), ('crest', 'crest', 'trough'))
#ind_CT = ((None, None), (1, 4, 0))
#ind_CT = ((None, None), (1, 34, 0))
ind_CT = ((None, None), (28, 183, 0))
title_case = (('idealized crest', 'idealized trough'),
  ('real ridge', 'real hill', 'real trough'))

# File format
fmt_lespath = ('/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_3D/{case:s}')
fmt_real = './data/statistics_budget_CrestTrough_alpha2_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_prod = './figure/paperraw/paperraw_profProdByCanopy.png'

# Topography parameters
zpad = (0, None)
amp = (25, None)
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
#items_bdg = ('prod', 'prod_h', 'prod_v', 'prod_dudz')
#labs_bdg = (r'$P$', r'$P^h$', r'$P^v$', r'$P_{homo}$')
items_bdg = ('prod', 'prod_dudz')
labs_bdg = (r'$P$', r'$P_{homo}$')
dz_real = 2

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.13, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X')
xspacing = 1
yspacing = 0.5
xlim = (-0.2, 7.2)
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
terms_bdg = [[dict.fromkeys(items_bdg, None) for isc in range(nsc[ic])]
  for ic in range(nca)]



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
      elif type_CT[ic][isc] == 'trough':
        terms_bdg[ic][isc] = dset['bdg_trough'][ind_CT[ic][isc]]



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

    # Production
    field_bdg[ic]['prod_h'] = -data[ic]['uu_t']*data[ic]['dudx']\
      - data[ic]['uw_t']*data[ic]['dwdx']
    field_bdg[ic]['prod_v'] = - data[ic]['ww_t']*data[ic]['dwdz']\
      - data[ic]['uw_t']*data[ic]['dudz']
    field_bdg[ic]['prod'] = field_bdg[ic]['prod_h'] + field_bdg[ic]['prod_v']
    field_bdg[ic]['prod_dudz'] = -data[ic]['uw_t']*data[ic]['dudz']



## Smooth the data
for ic in range(nca):
  if lab_case[ic] != 'real':
    for item in items_bdg:
      field_bdg[ic][item] = gaussian_filter(field_bdg[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_bdg:
      field_bdg[ic][item] = np.ma.array(field_bdg[ic][item], mask=mask[ic])



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'real':
    for isc in range(nsc[ic]):
      for item in items_bdg:
        terms_bdg[ic][isc][item] = get_vprof(field_bdg[ic][item],
          ix_tower[ic][isc], amp[ic])
  else:
    for isc in range(nsc[ic]):
      terms_bdg[ic][isc]['prod'] = terms_bdg[ic][isc]['prod_h']\
        + terms_bdg[ic][isc]['prod_v']



##Renormalize the data
for ic in range(nca):
  for isc in range(nsc[ic]):
    for item in items_bdg:
      terms_bdg[ic][isc][item] *= hc/dm[ic].lz



## Plot the budget normalized by u* and hc
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(8, 5))
gs = gridspec.GridSpec(1, nca)
gs.update(**sz)

for ic in range(nca):
  isub = ic
  ax = fig.add_subplot(gs[isub])
  it = -1
  for isc in range(nsc[ic]):
    # Plot the budget terms
    if lab_case[ic] != 'real':
      for it, item in enumerate(items_bdg):
        label = labs_bdg[it] + ', ' + title_case[ic][isc]
        ax.plot(terms_bdg[ic][isc][item], Z_gd[ic][:, ix_tower[ic][isc]]/hc,
          color = 'C0{:d}'.format((nca-ic)*isc),
          lw=lw, ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=label)
    else:
      for it, item in enumerate(items_bdg):
        label = labs_bdg[it] + ', ' + title_case[ic][isc]
        ax.plot(terms_bdg[ic][isc][item], Z_prof/hc, lw=lw,
          color = 'C0{:d}'.format(isc),
          ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
          markevery=(np.mod(it, 5), 5), label=label)

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axvline(x=0, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Axis label
    ax.set_xlabel(r'$P\frac{h_c}{u_*^3}$')
    ax.legend(loc='upper right')

    if isub==0:
      ax.set_ylabel('$Z/h_c$')
    else:
      ax.set_yticklabels([])

    #if isub==nttl-1:
    #  ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    #ax.set_title(title_case[ic][isc], fontsize='large')

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
fig.savefig(fn_prod, dpi=400)
