#!/usr/bin/env python3

"""
Program: paperraw_topography
Plot the topography used in LES and mark the virtual towers.
"""

### Histories:
### 08/11/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created.



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
case = 'amazon_canopy_real'
lab_case = 'real'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fn_tpg = './figure/paperraw/paperraw_tpg.png'
fmt_CTstat = './data/statistics_budget_CrestTrough_alpha2_{case:s}_tt{tts:08d}-{tte:08d}.npz'

# Plot range
nmax = 200
nmin = 100
#ind_crest = [1, 34]
ind_crest = [28, 183]
ind_trough = [0,]
flag_speNm = True
name_crest = ['Real ridge', 'Real hill']
name_trough = ['Real trough',]
x_crs = 917.5
y_crs = 1500
tts = 144100
tte = 324000

# Figure
sz = dict(left=0.12, right=0.9, bottom=0.07, top=0.98,
  wspace=0.2, hspace=0.3)
lw = 2
ls = '--'
color = 'gray'
xspacing = 500
yspacing = 500
levels_tpg = np.arange(0, 50+0.1, 5)



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
ibm = lp.domain.dmClass.immersedbdy(param)
dm = param.domain
z_tpg = ibm.z_phi0 * dm.zi
y_tpg, x_tpg = lp.domain.dmFun.xycoord(dm)



## Find the first n highest crest
fn_ct = fmt_CTstat.format(case=case, tts=tts, tte=tte)
dset = np.load(fn_ct, allow_pickle=True)
iy_crest = dset['iy_crest']
ix_crest = dset['ix_crest']
hgt_crest = dset['hgt_crest']

iy_max = iy_crest[:nmax]
ix_max = ix_crest[:nmax]
y_max = y_tpg[iy_max, ix_max]
x_max = x_tpg[iy_max, ix_max]
hgt_max = hgt_crest[:nmax]

print(iy_max[ind_crest])
print(ix_max[ind_crest])
print(hgt_max)
print(80*'#')



## Find the first n lowest trough
iy_trough = dset['iy_trough']
ix_trough = dset['ix_trough']
hgt_trough = dset['hgt_trough']

iy_min = iy_trough[:nmin]
ix_min = ix_trough[:nmin]
y_min = y_tpg[iy_min, ix_min]
x_min = x_tpg[iy_min, ix_min]
hgt_min = hgt_trough[:nmin]

print(iy_min)
print(ix_min)
print(hgt_min)



## Plot the results
# Initlize the figure
cmap_tpg = cm.get_cmap("viridis")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(6, 6))
gs = gridspec.GridSpec(1, 1)
gs.update(**sz)

# Plot the topography
isub = 0
ax = fig.add_subplot(gs[isub])
caxc_tke = ax.contourf(x_tpg, y_tpg, z_tpg, levels_tpg, cmap=cmap_tpg,
  extend='both')

# Plot highest crests
dx=-200
dy=50
ax.scatter(x_max[ind_crest], y_max[ind_crest], marker='o', color='k')
for inum, ind in enumerate(ind_crest):
  if flag_speNm:
    txt = '{speNm:s},\n{hgt:4.1f} m'.format(speNm=name_crest[inum],
      hgt=hgt_max[ind])
  else:
    txt = 'Real crest #{n:d}, {hgt:4.1f} m'.format(n=inum+1, hgt=hgt_max[ind])
  ax.annotate(txt, (x_max[ind]+dx, y_max[ind]+dy), fontsize='x-large',
    weight='bold', zorder=10)
#dx=50
#dy=0
#ax.scatter(x_max, y_max, marker='o', color='k')
#istart=183
#for inum in range(istart, istart+1):
#  txt = '{:d}'.format(inum)
#  ax.annotate(txt, (x_max[inum]+dx, y_max[inum]+dy), fontsize='x-small',
#    zorder=10)

# Plot lowest troughs
ax.scatter(x_min[ind_trough], y_min[ind_trough], marker='^', color='k')
for inum, ind in enumerate(ind_trough):
  if flag_speNm:
    txt = '{speNm:s},\n{hgt:4.1f} m'.format(speNm=name_trough[inum],
      hgt=hgt_min[ind])
  else:
    txt = 'Real trough #{n:d}, {hgt:4.1f} m'.format(n=inum+1, hgt=hgt_min[ind])
  ax.annotate(txt, (x_min[ind]+dx, y_min[ind]+dy), fontsize='x-large',
    weight='bold')

# Mark the cross-sections
ax.axvline(x=x_crs, color=color, lw=lw, ls=ls)
ax.axhline(y=y_crs, color=color, lw=lw, ls=ls)

# Axis property
ax.xaxis.set_major_locator(xmaxLocator)
ax.yaxis.set_major_locator(ymaxLocator)
ax.set_xlim([0, dm.lx])
ax.set_ylim([0, dm.ly])
ax.set_xlabel(r'$x (m)$', fontsize='large')
ax.set_ylabel(r'$y (m)$', fontsize='large')

# Axis label
ax.set_ylabel('$x (m)$')
ax.set_ylabel('$y (m)$')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
right = grid_pos[3][0]
rpad = 0.03
cbar_ax = fig.add_axes([right+rpad, 0.3, 0.02, 0.35])
ticks = np.arange(levels_tpg[0], levels_tpg[-1]+0.1, 10)
cbar = plt.colorbar(caxc_tke, ticks = ticks, cax=cbar_ax)
cbar.ax.set_xlabel(r'$z (m)$', ha='center', va='bottom')
cbar.ax.xaxis.set_label_position('top')

# Save figure
#fig.savefig(fn_tpg, dpi=400)
