#!/usr/bin/env python3

"""
Program: paperraw_TKEbudgetPercentilesCrestTrough
"""

### Histories:
### 2020/08/10 -- Bicheng Chen (chabby@ucla.edu) -- First created.



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
case = 'amazon_canopy_real'
lab_case = 'real'
case_flat = 'amazon_canopy_nohill'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fmt_real = './data/statistics_budget_CrestTrough_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_flat = './data/meanProf_flat.npz'
fn_crest1 = './figure/paperraw/paperraw_TKEbudgetPercentilesCrestByCanopy.png'
fn_trough1 = './figure/paperraw/paperraw_TKEbudgetPercentilesTroughByCanopy.png'
fn_crest2 = './figure/paperraw/paperraw_TKEbudgetPercentilesCrestByDissip.png'
fn_trough2 = './figure/paperraw/paperraw_TKEbudgetPercentilesTroughByDissip.png'

# Flat case
lz_flat = 520

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
tts = 144100
tte = 216000
dz = 2
wid_crest = 4
items_bdg = ('prod_h', 'prod_v', 'dissip', 'canopy', 'adv_h', 'adv_v',
  'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'sum')
labs_bdg = (r'$P^h$', r'$P^v$', r'$\epsilon$', r'$\epsilon_c$', r'$A^h_e$',
  r'$A^v_e$', r'$T^h_e$', r'$T^v_e$', r'$\Pi^h_e$', r'$\Pi^v_e$', 'sum')
perc = (10, 50, 90)

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.05, top=0.95,
  wspace=0.1, hspace=0.3)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^', 'X', 'h', '+', '8', 'p', '*', 'P', 'd', 'H')
xspacing = 2
yspacing = 0.5
xlim = (-6, 7)
xlim2 = (-1.2, 2.2)
ylim = (0, 3)



## Functions
def get_dphidz(phi, dz, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1] = (phi[1:]-phi[:-1]) / (dz/dm.zi)
  dphidz[-1] = dphidz[-2]
  return dphidz



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

fn_real = fmt_real.format(case=case, tts=tts, tte=tte)
dset = np.load(fn_real, allow_pickle=True)
Z_prof = dset['Z_prof']
terms_crest = dset['bdg_crest']
terms_trough = dset['bdg_trough']

n_crest = len(terms_crest)
n_trough = len(terms_trough)



## Normalization by canopy height
for ic in range(n_crest):
  for item in items_bdg:
    terms_crest[ic][item] *= hc/dm.lz

for ic in range(n_trough):
  for item in items_bdg:
    terms_trough[ic][item] *= hc/dm.lz



## Reverse the dissipation
for ic in range(n_crest):
  terms_crest[ic]['dissip'] *= -1
  terms_crest[ic]['canopy'] *= -1

for ic in range(n_trough):
  terms_trough[ic]['dissip'] *= -1
  terms_trough[ic]['canopy'] *= -1



## Calculate the percentile
nz = len(Z_prof)
perc_crest = dict.fromkeys(items_bdg)
perc_trough = dict.fromkeys(items_bdg)
arr_crest = np.zeros(n_crest)
arr_trough = np.zeros(n_trough)
for item in items_bdg:
  perc_crest[item] = np.zeros((len(perc), len(Z_prof)))
  perc_trough[item] = np.zeros((len(perc), len(Z_prof)))
  for iz in range(nz):
    for ic in range(n_crest):
      arr_crest[ic] = terms_crest[ic][item][iz]
    perc_crest[item][:, iz] =\
      np.percentile(arr_crest, perc, overwrite_input=True)

    for ic in range(n_trough):
      arr_trough[ic] = terms_trough[ic][item][iz]
    perc_trough[item][:, iz] =\
      np.percentile(arr_trough, perc, overwrite_input=True)



## Transform percentiles to error
for item in items_bdg:
  perc_crest[item][0, :] = perc_crest[item][1, :] - perc_crest[item][0, :]
  perc_crest[item][2, :] = perc_crest[item][2, :] - perc_crest[item][1, :]
  perc_trough[item][0, :] = perc_trough[item][1, :] - perc_trough[item][0, :]
  perc_trough[item][2, :] = perc_trough[item][2, :] - perc_trough[item][1, :]



## Read the flat case
dset_flat = np.load(fn_flat, allow_pickle=True)
mp_flat = dset_flat['terms'].item()
Z_mp = dset_flat['Z_prof'] / hc
for item in mp_flat.keys():
  mp_flat[item] *= hc/lz_flat

mp_flat['dissip'] *= -1
mp_flat['canopy'] *= -1



## Plot the results
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

# Plot the percentiles over crests
ifig = 0
fig = plt.figure(ifig, figsize=(8, 10))
nbdg = len(items_bdg)
ncol = 4
gs = gridspec.GridSpec(int(np.ceil(nbdg/ncol)), ncol)
gs.update(**sz)

# Plot the budget terms
isub=-1
for it, item in enumerate(items_bdg):
  isub += 1
  ax = fig.add_subplot(gs[isub])
  ax.errorbar(perc_crest[item][1, :], Z_prof/hc,
    xerr=perc_crest[item][[0, 2], :], lw=lw,
    ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
    color='C{:02d}'.format(np.mod(it, 10)),
    capsize=2*lw, elinewidth=lw,
    markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
  ax.plot(mp_flat[item], Z_mp, color='k', lw=lw, ls='-',
    label='flat', zorder=10)

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls='-.', color='k')
  ax.axvline(x=0, lw=lw, ls='--', color='gray')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  
  # Axis label
  if it>=nbdg-ncol:
    ax.set_xlabel(
      r'$\frac{\partial \overline{e}}{\partial t}\frac{h_c}{u_*^3}$')
  else:
    ax.set_xticklabels([])
  if it%ncol==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])
  ax.set_title(labs_bdg[it], fontsize='large')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

fig.savefig(fn_crest1, dpi=400)

# Plot the percentiles over trough
ifig += 1
fig = plt.figure(ifig, figsize=(8, 10))
nbdg = len(items_bdg)
ncol = 4
gs = gridspec.GridSpec(int(np.ceil(nbdg/ncol)), ncol)
gs.update(**sz)

# Plot the budget terms
isub=-1
for it, item in enumerate(items_bdg):
  isub += 1
  ax = fig.add_subplot(gs[isub])
  ax.errorbar(perc_trough[item][1, :], Z_prof/hc,
    xerr=perc_trough[item][[0, 2], :], lw=lw,
    ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
    color='C{:02d}'.format(np.mod(it, 10)),
    capsize=2*lw, elinewidth=lw,
    markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
  ax.plot(mp_flat[item], Z_mp, color='k', lw=lw, ls='-',
    label='flat', zorder=10)

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls='-.', color='k')
  ax.axvline(x=0, lw=lw, ls='--', color='gray')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  
  # Axis label
  if it>=nbdg-ncol:
    ax.set_xlabel(
      r'$\frac{\partial \overline{e}}{\partial t}\frac{h_c}{u_*^3}$')
  else:
    ax.set_xticklabels([])
  if it%ncol==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])
  ax.set_title(labs_bdg[it], fontsize='large')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

fig.savefig(fn_trough1, dpi=400)



## Normalization by total dissipation
for ic in range(n_crest):
  dissipTtl_crest = terms_crest[ic]['dissip']\
    + terms_crest[ic]['canopy']
  for item in items_bdg:
    terms_crest[ic][item] /= dissipTtl_crest

for ic in range(n_trough):
  dissipTtl_trough = terms_trough[ic]['dissip']\
    + terms_trough[ic]['canopy']
  for item in items_bdg:
    terms_trough[ic][item] /= dissipTtl_trough



## Calculate the percentile
for item in items_bdg:
  perc_crest[item] = np.zeros((len(perc), len(Z_prof)))
  perc_trough[item] = np.zeros((len(perc), len(Z_prof)))
  for iz in range(nz):
    for ic in range(n_crest):
      arr_crest[ic] = terms_crest[ic][item][iz]
    perc_crest[item][:, iz] =\
      np.percentile(arr_crest, perc, overwrite_input=True)

    for ic in range(n_trough):
      arr_trough[ic] = terms_trough[ic][item][iz]
    perc_trough[item][:, iz] =\
      np.percentile(arr_trough, perc, overwrite_input=True)



## Transform percentiles to error
for item in items_bdg:
  perc_crest[item][0, :] = perc_crest[item][1, :] - perc_crest[item][0, :]
  perc_crest[item][2, :] = perc_crest[item][2, :] - perc_crest[item][1, :]
  perc_trough[item][0, :] = perc_trough[item][1, :] - perc_trough[item][0, :]
  perc_trough[item][2, :] = perc_trough[item][2, :] - perc_trough[item][1, :]



## Read the flat case
dissipTtl_flat = mp_flat['dissip'] + mp_flat['canopy']
for item in mp_flat.keys():
  mp_flat[item] /= dissipTtl_flat



## Plot the results
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

# Plot the percentiles over crests
ifig += 1
fig = plt.figure(ifig, figsize=(8, 10))
nbdg = len(items_bdg)
ncol = 4
gs = gridspec.GridSpec(int(np.ceil(nbdg/ncol)), ncol)
gs.update(**sz)

# Plot the budget terms
isub=-1
for it, item in enumerate(items_bdg):
  isub += 1
  ax = fig.add_subplot(gs[isub])
  ax.errorbar(perc_crest[item][1, :], Z_prof/hc,
    xerr=perc_crest[item][[0, 2], :], lw=lw,
    ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
    color='C{:02d}'.format(np.mod(it, 10)),
    capsize=2*lw, elinewidth=lw,
    markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
  ax.plot(mp_flat[item], Z_mp, color='k', lw=lw, ls='-',
    label='flat', zorder=10)

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls='-.', color='k')
  ax.axvline(x=0, lw=lw, ls='--', color='gray')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim(xlim2)
  ax.set_ylim(ylim)
  
  # Axis label
  if it>=nbdg-ncol:
    ax.set_xlabel(
      r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$')
  else:
    ax.set_xticklabels([])
  if it%ncol==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])
  ax.set_title(labs_bdg[it], fontsize='large')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

fig.savefig(fn_crest2, dpi=400)

# Plot the percentiles over trough
ifig += 1
fig = plt.figure(ifig, figsize=(8, 10))
nbdg = len(items_bdg)
ncol = 4
gs = gridspec.GridSpec(int(np.ceil(nbdg/ncol)), ncol)
gs.update(**sz)

# Plot the budget terms
isub=-1
for it, item in enumerate(items_bdg):
  isub += 1
  ax = fig.add_subplot(gs[isub])
  ax.errorbar(perc_trough[item][1, :], Z_prof/hc,
    xerr=perc_trough[item][[0, 2], :], lw=lw,
    ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
    color='C{:02d}'.format(np.mod(it, 10)),
    capsize=2*lw, elinewidth=lw,
    markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
  ax.plot(mp_flat[item], Z_mp, color='k', lw=lw, ls='-',
    label='flat', zorder=10)

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls='-.', color='k')
  ax.axvline(x=0, lw=lw, ls='--', color='gray')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim(xlim2)
  ax.set_ylim(ylim)
  
  # Axis label
  if it>=nbdg-ncol:
    ax.set_xlabel(
      r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$')
  else:
    ax.set_xticklabels([])
  if it%ncol==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])
  ax.set_title(labs_bdg[it], fontsize='large')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

fig.savefig(fn_trough2, dpi=400)
