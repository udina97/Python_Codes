#!/usr/bin/env python3

"""
Program: paperraw_advhVsZ
Plot the advection and turbulence transport terms vs Z/hc over selected locations and all crests and troughs.
"""

### Histories:
### 8/26/2021 -- Bicheng Chen (chabby@ucla.edu) -- First created.



## Prerequisite Module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import pandas as pd



## User-specified Variable
# Case
case_all = ('real', 'hill', 'hill05h', 'hill2h')
loc_all = ('crest', 'trough')

# File format
fmt_in = './data/eqs10N11_{loc:s}_{case:s}.txt'
fn_advh = './figure/paperraw/paperraw_advhVsZ.png'

# Topography parameters
zpad = (5, 0, None)
amp = (0, 25, None)
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
dz_real = 2
zran = (0, 2)
dzran = 0.5
hc_cri = 2/3
#hc_cri = 1
items_hill = ('Z/hc', 'adv_h', 'pi_v', 'eq10', 'adv', '-R-T^v_e', 'eps+eps_c')
items_real = ('num', 'hgt', 'size', 'Z/hc', 'adv_h', 'pi_v', 'eq10',
  'adv', '-R-T^v_e', 'eps+eps_c')

# Figure
sz = dict(left=0.08, right=0.9, bottom=0.05, top=0.96,
  wspace=0.1, hspace=0.25)
lw = 2
ls = ('-', '--', '-.')
markers = ('x', 'd', '^')
facecolors = ('k', 'gray', 'w')
ms_real = 5
ms_hill = 20
xspacing = 1
xspacing_minor = 0.5
yspacing = 0.5
xlim = (-1.5, 1.5)
ylim = (-1.5, 1.5)
r_gray = 0.7
ncmap = 60
cmap_val = cm.get_cmap("viridis")
cmap_msk = cm.get_cmap("gray")



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



## Read data and calculate the temporal mean
dset = dict.fromkeys(case_all, None)
nca = len(case_all)
for case in case_all:
  dset[case] = dict.fromkeys(loc_all, None)

for case in case_all:
  for loc in loc_all:
    fn = fmt_in.format(loc=loc, case=case)
    print(fn)
    if case == 'real':
      dset[case][loc] = pd.read_csv(fn, header=0, names=items_real)
    else:
      dset[case][loc] = pd.read_csv(fn, header=0, names=items_hill)



## Filter the data
for case in case_all:
  for loc in loc_all:
    ind = dset[case][loc]['Z/hc']<=zran[-1]
    dset[case][loc] = dset[case][loc].loc[ind]



## Error calculation
corr = dict.fromkeys(case_all, None)
rmse = dict.fromkeys(case_all, None)
for case in case_all:
  corr[case] = dict.fromkeys(loc_all, None)
  rmse[case] = dict.fromkeys(loc_all, None)
  for loc in loc_all:
    data = dset[case][loc]
    ind = data['Z/hc']>=hc_cri
    ind_op = ~ind

    corr[case][loc] = dict.fromkeys(['adv_h', 'pi_v', 'adv'], None)
    rmse[case][loc] = dict.fromkeys(['adv_h', 'pi_v', 'adv'], None)

    lNr = data.loc[ind][['adv_h', 'eq10']]
    corr[case][loc]['adv_h'] = lNr.corr().values[0, 1]
    rmse[case][loc]['adv_h'] = ((lNr['adv_h']-lNr['eq10'])**2).mean() ** 0.5

    lNr = data.loc[ind_op][['pi_v', 'eq10']]
    corr[case][loc]['pi_v'] = lNr.corr().values[0, 1]
    rmse[case][loc]['pi_v'] = ((lNr['pi_v']-lNr['eq10'])**2).mean() ** 0.5

    lNr = data.loc[ind][['adv', '-R-T^v_e']]
    corr[case][loc]['adv'] = lNr.corr().values[0, 1]
    rmse[case][loc]['adv'] = ((lNr['adv']-lNr['-R-T^v_e'])**2).mean() ** 0.5

    print('#'*80)
    print(case, loc)
    print('>={:f}'.format(hc_cri))
    print('corr=', corr[case][loc]['adv_h'])
    print('RMSE=', rmse[case][loc]['adv_h'])

    print('<{:f}'.format(hc_cri))
    print('corr=', corr[case][loc]['pi_v'])
    print('RMSE=', rmse[case][loc]['pi_v'])

    print('>={:f}'.format(hc_cri))
    print('corr=', corr[case][loc]['adv'])
    print('RMSE=', rmse[case][loc]['adv'])



## Plot the data
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)
xmaxLocator_minor = MultipleLocator(xspacing_minor)

# Generate the cmap
r_msk = (hc_cri-zran[0])/(zran[1]-zran[0]) 
n_msk = int(ncmap * r_msk)
cmap_ahe = colors.ListedColormap(np.vstack((cmap_msk(r_gray*np.ones(n_msk)),
  cmap_val(np.linspace(r_msk, 1, ncmap-n_msk)))), 'maskVal')

r_msk = 1-r_msk
n_msk = ncmap-n_msk
cmap_phe = colors.ListedColormap(np.vstack((
  cmap_val(np.linspace(0, r_msk, ncmap-n_msk)),
  cmap_msk(r_gray*np.ones(n_msk)))), 'valMask')

ifig = 0
fig = plt.figure(ifig, figsize=(8, 12))
gs = gridspec.GridSpec(3, 2)
gs.update(**sz)

# Plot the Advection terms vs eq 10
isub = 0
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['crest']
cax1 = ax.scatter(data['adv_h'], data['eq10'],
  c=data['Z/hc'], cmap=cmap_ahe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['crest']
  data = data.loc[data['Z/hc']>=hc_cri]
  ax.scatter(data['adv_h'], data['eq10'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'crest'
item = 'adv_h'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$A_e^h$')
ax.set_ylabel(r'$-R-T_e^v-A_e^v$')
ax.set_title('Crest, Z/hc>={:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the Advection terms vs eq 10
isub += 1
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['trough']
ax.scatter(data['adv_h'], data['eq10'],
  c=data['Z/hc'], cmap=cmap_ahe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['trough']
  data = data.loc[data['Z/hc']>=hc_cri]
  ax.scatter(data['adv_h'], data['eq10'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'trough'
item = 'adv_h'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$A_e^h$')
ax.set_yticklabels([])
ax.set_title('Trough, Z/hc>={:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the Advection terms vs eq 10
isub += 1
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['crest']
cax2 = ax.scatter(data['pi_v'], data['eq10'],
  c=data['Z/hc'], cmap=cmap_phe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['crest']
  data = data.loc[data['Z/hc']<hc_cri]
  ax.scatter(data['pi_v'], data['eq10'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'crest'
item = 'pi_v'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$\Pi_e^v$')
ax.set_ylabel(r'$-R-T_e^v-A_e^v$')
ax.set_title('Crest, Z/hc<{:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the Advection terms vs eq 10
isub += 1
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['trough']
ax.scatter(data['pi_v'], data['eq10'],
  c=data['Z/hc'], cmap=cmap_phe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['trough']
  data = data.loc[data['Z/hc']<hc_cri]
  ax.scatter(data['pi_v'], data['eq10'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'trough'
item = 'pi_v'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$\Pi_e^v$')
ax.set_yticklabels([])
ax.set_title('Trough, Z/hc<{:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the Advection terms vs eq 10
isub += 1
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['crest']
ax.scatter(data['adv'], data['-R-T^v_e'],
  c=data['Z/hc'], cmap=cmap_ahe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['crest']
  data = data.loc[data['Z/hc']>=hc_cri]
  ax.scatter(data['adv'], data['-R-T^v_e'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'crest'
item = 'adv'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$A_e$')
ax.set_ylabel(r'$-R-T^v_e$')
ax.set_title('Crest, Z/hc>={:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Plot the Advection terms vs eq 10
isub += 1
ax = fig.add_subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')

data = dset['real']['trough']
ax.scatter(data['adv'], data['-R-T^v_e'],
  c=data['Z/hc'], cmap=cmap_ahe, vmin=zran[0], vmax=zran[1],
  s=ms_real, alpha=0.5)

for ic in range(1, nca):
  data = dset[case_all[ic]]['trough']
  data = data.loc[data['Z/hc']>=hc_cri]
  ax.scatter(data['adv'], data['-R-T^v_e'], s=ms_hill, color='k',
    marker=markers[ic-1], facecolor=facecolors[ic-1])
  ax.plot([xlim[0], xlim[1]], [ylim[0], xlim[1]], color='k', lw=lw)

# Correlation and RMSE
loc = 'trough'
item = 'adv'
txt = "Case           r        RMSE\n"\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Real',
  corr['real'][loc][item], rmse['real'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H/2',
  corr['hill05h'][loc][item], rmse['hill05h'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal H',
  corr['hill'][loc][item], rmse['hill'][loc][item])\
  + "{:<12s}{:^7.3f}{:^7.3f}\n".format('Ideal 2H',
  corr['hill2h'][loc][item], rmse['hill2h'][loc][item])

ax.text(0.01, 1.0, txt,
  transform=ax.transAxes, fontsize='small', ha='left', va='top')

# Axis property
ax.xaxis.set_major_locator(xmaxLocator_minor)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Axis label
ax.set_xlabel(r'$A_e$')
ax.set_yticklabels([])
ax.set_title('Trough, Z/hc>={:.3f}'.format(hc_cri), fontsize='small')

ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# colorbar
x0 = 0.02
dx = 0.03
y0 = 0.15
dy = 0.2
dy_buf = 0.2
cbar_ax = fig.add_axes([sz['right']+x0, sz['bottom']+y0, dx, dy])
cbar = plt.colorbar(cax2,
  ticks = np.arange(zran[0], zran[1]+dzran, dzran), cax=cbar_ax,
  orientation='vertical')
cbar.ax.set_xlabel(r'$Z/h_c$', labelpad=10)
cbar.ax.xaxis.set_label_position('top')

cbar_ax = fig.add_axes([sz['right']+x0, sz['bottom']+y0+dy+dy_buf, dx, dy])
cbar = plt.colorbar(cax1,
  ticks = np.arange(zran[0], zran[1]+dzran, dzran), cax=cbar_ax,
  orientation='vertical')
cbar.ax.set_xlabel(r'$Z/h_c$', labelpad=10)
cbar.ax.xaxis.set_label_position('top')

# Save figure
fig.savefig(fn_advh, dpi=600)
