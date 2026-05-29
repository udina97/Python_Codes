#!/usr/bin/env python3

"""
Program: paperraw_profVelTowers
Plot the velocity profiles over crests and troughs. The terms are normalized by the friction velocity and the canopy height.
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
ind_CT = ((None,), (None, None), (1, 4, 0))
title_case = (('flat', ), ('idealized crest', 'idealized trough'),
  ('real ridge', 'real hill', 'real trough'))

# File format
fmt_lespath = ('/data/2/bzc/LES/amazon_2D/{case:s}',
  '/data/2/bzc/LES/amazon_2D/{case:s}', '/data/2/bzc/LES/amazon_3D/{case:s}')
fmt_real = './data/statistics_budget_CrestTrough_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_mp = './data/meanProf_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_vel1 = './figure/paperraw/paperraw_profVelTowers1.png'
fn_vel2 = './figure/paperraw/paperraw_profVelTowers2.png'

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
te = 21600
items = ('u', 'v', 'w', 'u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_vel = ('ubar', 'wbar', 'tke')
labs_vel = (r'$\overline{u_h/u_*}$', r'$\overline{w_h/u_*}$',
  r'$tke^\frac{1}{2}/u_*$')
dz_real = 2
Z_fig = np.arange(0.05, 3.01, 0.05)

# Figure
sz = dict(left=0.07, right=0.89, bottom=0.1, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X')
xspacing = 1
xspacing_minor = 0.5
yspacing = 0.5
xlim_vel = (-0.5, 9)
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
ratio = [[None for isc in range(nsc[ic])] for ic in range(nca)]
field_vel = [dict.fromkeys(items_vel, None) for ic in range(nca)]
terms_vel = [[dict.fromkeys(items_vel, None) for isc in range(nsc[ic])]
  for ic in range(nca)]
uw_t = [[None for isc in range(nsc[ic])] for ic in range(nca)]
d0 = [[None for isc in range(nsc[ic])] for ic in range(nca)]



## Read the mean profiles
Z_mp = [None for ic in range(nca)]
for ic in range(nca):
  fn_mp = fmt_mp.format(case=case[ic], tts=ts*10+100, tte=te*10)
  dset = np.load(fn_mp, allow_pickle=True)
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
        terms_vel[ic][isc] = dset['bdg_crest'][ind_CT[ic][isc]]
        terms_vel[ic][isc].update(dset['vel_crest'][ind_CT[ic][isc]])
        d0[ic][isc] = dset['d0_crest'][ind_CT[ic][isc]]
      elif type_CT[ic][isc] == 'trough':
        terms_vel[ic][isc] = dset['bdg_trough'][ind_CT[ic][isc]]
        terms_vel[ic][isc].update(dset['vel_trough'][ind_CT[ic][isc]])
        d0[ic][isc] = dset['d0_trough'][ind_CT[ic][isc]]



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
    # Velocity field
    field_vel[ic]['ubar'] = np.sqrt(data[ic]['u']**2 + data[ic]['v']**2)
    field_vel[ic]['wbar'] = data[ic]['w']

    # TKE
    data[ic]['uu_t'] = data[ic]['u2'] - data[ic]['u']**2
    data[ic]['uu_t'] = gaussian_filter(data[ic]['uu_t'], sigma=(1, 1),
      mode='wrap')
    data[ic]['vv_t'] = data[ic]['v2'] - data[ic]['v']**2
    data[ic]['vv_t'] = gaussian_filter(data[ic]['vv_t'], sigma=(1, 1),
      mode='wrap')
    data[ic]['ww_t'] = data[ic]['w2'] - data[ic]['w']**2
    data[ic]['ww_t'] = gaussian_filter(data[ic]['ww_t'], sigma=(1, 1),
      mode='wrap')

    field_vel[ic]['tke'] = (data[ic]['uu_t'] + data[ic]['vv_t'] + data[ic]['ww_t']) / 2

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

  else:
    for isc in range(nsc[ic]):
      terms_vel[ic][isc]['ubar'] =\
        np.sqrt(terms_vel[ic][isc]['u']**2 + terms_vel[ic][isc]['v']**2)
      terms_vel[ic][isc]['wbar'] = terms_vel[ic][isc]['w']
      uu_t = terms_vel[ic][isc]['u2'] - terms_vel[ic][isc]['u']**2
      vv_t = terms_vel[ic][isc]['v2'] - terms_vel[ic][isc]['v']**2
      ww_t = terms_vel[ic][isc]['w2'] - terms_vel[ic][isc]['w']**2
      terms_vel[ic][isc]['tke'] = (uu_t + vv_t + ww_t) / 2



## Smooth the data
for ic in range(nca):
  if lab_case[ic] != 'real':
    for item in items_vel:
      field_vel[ic][item] = gaussian_filter(field_vel[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_vel:
      field_vel[ic][item] = np.ma.array(field_vel[ic][item], mask=mask[ic])



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'real':
    for isc in range(nsc[ic]):
      for item in items_vel:
        terms_vel[ic][isc][item] = get_vprof(field_vel[ic][item],
          ix_tower[ic][isc], amp[ic])
        if lab_case[ic] != 'real':
          terms_vel[ic][isc][item] =\
            np.interp(Z_fig, Z_gd[ic][:, ix_tower[ic][isc]]/hc,
              terms_vel[ic][isc][item])
  else:
    for isc in range(nsc[ic]):
      for item in items_vel:
        terms_vel[ic][isc][item] =\
          np.interp(Z_fig, Z_prof/hc, terms_vel[ic][isc][item])


## Get square root of TKE
for ic in range(nca):
  for isc in range(nsc[ic]):
    terms_vel[ic][isc]['tke'] = np.sqrt(terms_vel[ic][isc]['tke'])



##Renormalize the data
#for ic in range(nca):
#  for isc in range(nsc[ic]):
#    for item in items_vel:
#      if item in ('ubar', 'wbar'):
#        terms_vel[ic][isc][item] *= hc/dm[ic].lz
#      elif item is 'tke':
#  meanProf[ic] *= hc/dm[ic].lz



## Plot the velocity terms
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(2, (nttl+1)//2)
sz['right'] = 0.88
sz['top'] = 0.95
sz['bottom'] = 0.13
gs.update(**sz)

isub = -1
for ic in range(nca):
  for isc in range(nsc[ic]):
    isub += 1
    # Plot the budget terms
    ax = fig.add_subplot(gs[isub])
    for it, item in enumerate(items_vel):
      ax.plot(terms_vel[ic][isc][item], Z_fig, lw=lw,
        ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
        markevery=(np.mod(it, 5), 5), label=labs_vel[it])

    ax.axhline(y=1, lw=lw, ls='--', color='k')
    ax.axhline(y=2, lw=lw, ls='-.', color='k')
    ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
    ax.axvline(x=0, lw=lw, ls='--', color='gray')

    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim(xlim_vel)
    ax.set_ylim(ylim)

    # Axis label
    if isub >= nttl/2:
      ax.set_xlabel(r'$u\ (m/s)$')
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
  
fig.savefig(fn_vel1, dpi=400)



## Plot the velocity difference terms
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig += 1
fig = plt.figure(ifig, figsize=(8, 4))
gs = gridspec.GridSpec(1, 3)
sz['right'] = 0.98
gs.update(**sz)

for it, item in enumerate(items_vel):
  ax = fig.add_subplot(gs[it])
  # Plot the difference terms
  ind = -1
  for ic in range(nca):
    for isc in range(nsc[ic]):
      ind += 1
      ax.plot(terms_vel[ic][isc][item], Z_fig, lw=lw,
        ls=ls[np.mod(ind, len(ls))], marker=ms[np.mod(ind, len(ms))],
        markevery=(np.mod(ind, 5), 5), label=title_case[ic][isc])

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls='-.', color='k')
  ax.axhline(y=d0[ic][isc]/hc, lw=lw, ls='-.', color='gray')
  ax.axvline(x=0, lw=lw, ls='--', color='gray')

  # Axis label
  ax.set_xlabel(labs_vel[it])
  if np.mod(isub, 2)==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+it)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
  
fig.savefig(fn_vel2, dpi=400)
