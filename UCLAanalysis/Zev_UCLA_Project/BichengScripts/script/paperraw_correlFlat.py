#!/usr/bin/env python3

"""
Program: paperraw_crossCorrelHill
The crossection of correlation for the hill case
"""

### Histories:
### 07/24/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created



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
case = 'amazon_canopy_nohill'
lab_case = 'flat'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_2D/{case:s}'
fmt_base = './data/correl_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmtfig_bdg = './figure/paperraw/paperraw_correl{case:s}.png'

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
tts = 145200
tte = 216000
#tte = 147600
Z_prof = np.arange(1, 3*hc+1, 2)
dz = 2
items = ('uu', 'vv', 'ww')

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.1, top=0.96,
  wspace=0.1, hspace=0.3)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 500
levels_cr = np.arange(0, 1.01, 0.1)



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
  dphidz[:-1] = (phi[1:]-phi[:-1]) / (dm.dz/dm.zi)
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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  new_cmap = colors.LinearSegmentedColormap.from_list(
      'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
      cmap(np.linspace(minval, maxval, n)))
  return new_cmap



## Initialization
n_item = len(items)



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

fn_base = fmt_base.format(case=case, tts=tts, tte=tte)

dset_base = np.load(fn_base, mmap_mode='r', allow_pickle=True)

correl = dset_base['correl'].item()

x_gd = dset_base['x']
y_gd = dset_base['y']
z_gd = dset_base['z']
nz = z_gd.shape[0]



## Plot the results
# Initlize the figure
cmap_bdg = cm.get_cmap("viridis")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

# Plot the correlation

# x-cross-section
ifig = 0
fig = plt.figure(ifig, figsize=(8, 7))
gs = gridspec.GridSpec(nz, n_item)
gs.update(**sz)

isub = -1
for it in range(n_item):
  for iz in range(nz):
    isub += 1
    ax = fig.add_subplot(gs[isub])
    ax.set_aspect('equal', adjustable='box')
    cax=ax.contourf(x_gd[iz, :, :], y_gd[iz, :, :], correl[items[it]][iz, :, :],
      levels_cr, cmap=cmap_bdg, extend='both')
    ax.contour(x_gd[iz, :, :], y_gd[iz, :, :], correl[items[it]][iz, :, :],
      levels=[levels_cr[1], ], colors='w', linestyles='--', extend='both')
  
    # Axis property
    ax.xaxis.set_major_locator(xmaxLocator)
    ax.yaxis.set_major_locator(ymaxLocator)
    ax.set_xlim([-dm.lx/2, dm.lx/2])
    ax.set_ylim([-dm.ly/2, dm.ly/2])

    # Axis label
    ttl = "{item:s}, z={z:3.0f}m".format(item=items[it], z=z_gd[iz, 0, 0])
    ax.set_title(ttl)
    if it==n_item-1:
      ax.set_xlabel(r'$x (m)$')
    else:
      ax.set_xticklabels([])

    if iz == 0:
      ax.set_ylabel(r'$y (m)$')
    else:
      ax.set_yticklabels([])

    ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
      transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.07
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_cr[0], levels_cr[-1]+0.01, 0.1)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$r$', rotation=0, ha='right', va='center')

fnfig_bdg = fmtfig_bdg.format(case=lab_case.capitalize())
fig.savefig(fnfig_bdg, dpi=400)
