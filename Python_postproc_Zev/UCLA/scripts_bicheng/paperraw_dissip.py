#!/usr/bin/env python3

"""
Program: budget_TKE
Plot the budget of TKE
"""

### Histories:
### 11/05/2019 -- Bicheng Chen (chabby@ucla.edu) -- First created.
### 07/01/2020 -- Bicheng Chen (chabby@ucla.edu) -- Add the real case.



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
lab_case = ('Flat', 'Idealized', 'Real')

# File format
fmt_lespath = ('/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_2D/{case:s}', '/data/2/bzc/LES/amazon_3D/{case:s}')
fmt_real = './data/prof_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_fig = './figure/paperraw/paperraw_dissip.png'

# Topography parameters
zpad = (5, 0)
amp = (0, 25)
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 14400
te = 21600
#te = 18000
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('prod', 'canopy', 'dissip')

# Figure
sz = dict(left=0.15, right=0.98, bottom=0.1, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 2
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^')
xspacing = 1
yspacing = 0.5
xlim = (-0.2, 6)
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

def get_vprof(term, Z_gd, amp):
  if amp==0:
    prof = term.mean(axis=-1)
    return prof
  else:
    Z_prof = np.squeeze(Z_gd[:, 0])
    prof = term[:, 0] / term.shape[-1]
    for ix in range(1, term.shape[-1]):
      prof += np.interp(Z_prof, Z_gd[:, ix], term[:, ix]) / term.shape[-1]
    return prof



## Initialization
nca = len(case)
data = [{item: None for item in items} for ic in range(nca)]
term_bdg = [dict.fromkeys(items_bdg, None) for ic in range(nca)]



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
    term_bdg[ic] = dset['terms'].item()



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



wn_x = [None for ic in range(nca)]
wn_y = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    wn_x[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)
    wn_y[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].ny, dm[ic].dy/dm[ic].zi)

for ic in range(nca):
  if lab_case[ic] != 'Real':
    # Calculate the production
    #data[ic]['w'] = gaussian_filter(data[ic]['w'], sigma=(0, 1), mode='wrap')
    data[ic]['dwdx'] = get_dphidx(data[ic]['w'], wn_x[ic])
    data[ic]['dwdx'] = gaussian_filter(data[ic]['dwdx'], sigma=(1, 1), mode='wrap')
    
    #data[ic]['u'] = gaussian_filter(data[ic]['u'], sigma=(0, 1), mode='wrap')
    data[ic]['dudx'] = get_dphidx(data[ic]['u'], wn_x[ic])
    data[ic]['dudx'][1:, :] = 0.5 * (data[ic]['dudx'][:-1, :]
      +data[ic]['dudx'][1:, :])
    data[ic]['dudx'][0, :] = 0
    data[ic]['dudx'] = gaussian_filter(data[ic]['dudx'], sigma=(1, 1), mode='wrap')
    
    data[ic]['dudz'] = np.zeros(data[ic]['u'].shape) 
    data[ic]['dudz'][1:, :] =\
      (data[ic]['u'][1:, :]-data[ic]['u'][:-1, :]) / (dm[ic].dz/dm[ic].zi)
    data[ic]['dudz'] = gaussian_filter(data[ic]['dudz'], sigma=(1, 1), mode='wrap')

    data[ic]['dwdz'] = np.zeros(data[ic]['w'].shape) 
    data[ic]['dwdz'][1:-1, :] =\
      (data[ic]['w'][2:, :]-data[ic]['w'][:-2, :]) / (2*dm[ic].dz/dm[ic].zi)
    data[ic]['dwdz'] = gaussian_filter(data[ic]['dwdz'], sigma=(1, 1), mode='wrap')
    
    data[ic]['uw_t'] = data[ic]['uw'] - data[ic]['u']*data[ic]['w']
    data[ic]['uw_t'] = gaussian_filter(data[ic]['uw_t'], sigma=(1, 1), mode='wrap')
    data[ic]['uu_t'] = data[ic]['u2'] - data[ic]['u']**2
    data[ic]['uu_t'] = gaussian_filter(data[ic]['uu_t'], sigma=(1, 1), mode='wrap')
    data[ic]['vv_t'] = data[ic]['v2'] - data[ic]['v']**2
    data[ic]['vv_t'] = gaussian_filter(data[ic]['vv_t'], sigma=(1, 1), mode='wrap')
    data[ic]['ww_t'] = data[ic]['w2'] - data[ic]['w']**2
    data[ic]['ww_t'] = gaussian_filter(data[ic]['ww_t'], sigma=(1, 1), mode='wrap')
    
    # Production
    term_bdg[ic]['prod'] = -data[ic]['uu_t']*data[ic]['dudx']\
      - data[ic]['ww_t']*data[ic]['dwdz']\
      - data[ic]['uw_t']*(data[ic]['dudz']+data[ic]['dwdx'])
    term_bdg[ic]['prod'] = gaussian_filter(term_bdg[ic]['prod'], sigma=(0, 2), mode='wrap')

    # Dissipation
    term_bdg[ic]['canopy'] = (data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz'])
    term_bdg[ic]['canopy'] = wnode2uvpnode(term_bdg[ic]['canopy'])
    term_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
      +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
    term_bdg[ic]['dissip'] = data[ic]['dissip']



## Smooth the data
for ic in range(nca):
  if lab_case[ic] != 'Real':
    for item in items_bdg:
      term_bdg[ic][item] = gaussian_filter(term_bdg[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_bdg:
      term_bdg[ic][item] = np.ma.array(term_bdg[ic][item], mask=mask[ic])



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'Real':
    for item in items_bdg:
      term_bdg[ic][item] = get_vprof(term_bdg[ic][item], Z_gd[ic], amp[ic])



## Renormalize the data
for ic in range(nca):
  for item in items_bdg:
    term_bdg[ic][item] *= hc/dm[ic].lz



## Plot the results
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(4, 6))
gs = gridspec.GridSpec(1, 1)
gs.update(**sz)
ax = fig.add_subplot(gs[0])

# Plot the budget terms
for ic in range(nca):
  label = "$\epsilon+\epsilon_c$, " + lab_case[ic]
  if lab_case[ic] != 'Real':
    ax.plot(-term_bdg[ic]['dissip']-term_bdg[ic]['canopy'], Z_gd[ic][:, 0]/hc,
      color = 'C{:d}'.format(ic),
      lw=lw, ls=ls[0], marker=ms[np.mod(ic, len(ms))],
      markevery=(np.mod(ic, 5), 5), label=label)
    fn_npz = "./data/forpaper/fig3_profDissip_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_gd[ic][:, 0]/hc, dissip=-term_bdg[ic]['dissip']-term_bdg[ic]['canopy'])
  else:
    ax.plot(-term_bdg[ic]['dissip']-term_bdg[ic]['canopy'], Z_prof/hc, lw=lw,
      color = 'C{:d}'.format(ic),
      ls=ls[0], marker=ms[np.mod(ic, len(ms))],
      markevery=(np.mod(ic, 5), 5), label=label)
    fn_npz = "./data/forpaper/fig3_profDissip_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_prof/hc, dissip=-term_bdg[ic]['dissip']-term_bdg[ic]['canopy'])

for ic in range(nca):
  label = "$P$, " + lab_case[ic]
  if lab_case[ic] != 'Real':
    ax.plot(term_bdg[ic]['prod'], Z_gd[ic][:, 0]/hc,
      color = 'C{:d}'.format(ic),
      lw=lw-0.5, ls=ls[1], label=label)
    fn_npz = "./data/forpaper/fig3_profProd_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_gd[ic][:, 0]/hc, dissip=term_bdg[ic]['prod'])
  else:
    ax.plot(term_bdg[ic]['prod'], Z_prof/hc, lw=lw-0.5,
      color = 'C{:d}'.format(ic),
      ls=ls[1], label=label)
    fn_npz = "./data/forpaper/fig3_profProd_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_prof/hc, dissip=term_bdg[ic]['prod'])

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls=':', color='k')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$\langle\frac{\partial\overline{e}}{\partial t}\rangle\frac{h_c}{u_*^3}$')
ax.set_ylabel('$z/h_c$')
ax.legend(loc='upper right')
  
# Save figure
fig.savefig(fn_fig, dpi=600)
