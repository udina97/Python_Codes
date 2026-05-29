#!/usr/bin/env python3

"""
Program: paperraw_crossAeHill2h
The crossection of Ae terms for the hill case
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
case = ('amazon_canopy_hill2h', 'amazon_canopy_nohill')
lab_case = ('hill2h', 'flat')

# File format
fmt_lespath = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}'
fn_budget = './figure/paperraw/paperraw_crossAeHill2h.png'

# Topography parameters
zpad = (15, 5)
amp = (50, 0)
wl = 1000

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 18000
te = 25200
items = ('u', 'v', 'w','u2', 'v2', 'w2', 'dissip',
  'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('dissip', 'canopy', 'adv_h', 'adv_v', 'Ae')
labs_bdg = (r'$\epsilon$', r'$\epsilon_c$', r'$A^h_e$', r'$A^v_e$', r'$A_e$')
ind_plot = [-1, ]

# Ae cancelation
cri_divfree = 0.1

# Figure
sz = dict(left=0.08, right=0.98, bottom=0.17, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ars = 1
ms = 3
xspacing = 500
yspacing = 20
ylim_z = 400
levels_bdg = np.arange(-0.6, 0.61, 0.05)
levels_ccl = np.arange(0, 1.01, 0.1)



## Function
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



## Initialization
nca = len(case)
data = [dict.fromkeys(items, None) for ic in range(nca)]
terms_bdg = [dict.fromkeys(items_bdg, None) for ic in range(nca)]
tke = [None for ic in range(nca)]



## Read data and calculate the temporal mean
param = [None for ic in range(nca)]
dm = [None for ic in range(nca)]
for ic in range(nca):
  
  lespath = fmt_lespath.format(case=case[ic])
  param[ic] = lp.lesParam.lesClass.param(lespath)
  dm[ic] = param[ic].domain

for ic in range(nca):
  for item in items:
    print('loading '+ item)  
    data[ic][item], time = lp.io.io_averFile.loadLES_averFile(param[ic],
      qtype=item, tss=ts, tes=te)
    data[ic][item] = np.mean(data[ic][item], axis=0)

print('done loading vars')
## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]

mask = [None for ic in range(nca)]
for ic in range(nca):
  z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
  z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
  z_gd[ic] -= zpad[ic]
  Z_gd[ic] = z_gd[ic]-z_tpg[ic]
  mask[ic] = Z_gd[ic]<=0



## Calculate Ae terms
wn = [None for ic in range(nca)]
for ic in range(nca):
  wn[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)

#ix_Ae0 = [[] for ic in range(nca)]
#iz_Ae0 = [[] for ic in range(nca)]
for ic in range(nca):
  # Perturbation term
  data[ic]['uu_t'] = data[ic]['u2'] - data[ic]['u']**2
  data[ic]['vv_t'] = data[ic]['v2'] - data[ic]['v']**2
  data[ic]['ww_t'] = data[ic]['w2'] - data[ic]['w']**2
  
  # TKE
  tke[ic] = (data[ic]['uu_t'] + data[ic]['vv_t'] + data[ic]['ww_t']) / 2
  
  # Advection term
  ue = data[ic]['u']*tke[ic]
  duedx = get_dphidx(ue, wn[ic])
  tkez = np.zeros(tke[ic].shape)
  tkez[0, :] = 0
  tkez[1:, :] = 0.5*(tke[ic][:-1, :] + tke[ic][1:, :])
  we = data[ic]['w']*tkez
  dwedz = get_dphidz(we, dm[ic])
  terms_bdg[ic]['adv_h'] = -duedx
  terms_bdg[ic]['adv_v'] = -dwedz
  terms_bdg[ic]['Ae'] = -duedx-dwedz

  # Look for the positions where Ae are zero
  #if lab_case[ic] == 'hill':
  #  for iz in range(dm[ic].nz):
  #    for ix in range(dm[ic].nx):
  #      if ~mask[ic][iz, ix]:
  #        if terms_bdg[ic]['Ae'][iz, ix-1]*terms_bdg[ic]['Ae'][iz, ix] <=0:
  #          ix_Ae0[ic].append(ix)
  #          iz_Ae0[ic].append(iz)
  
  # Dissipation
  terms_bdg[ic]['canopy'] = data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz']
  terms_bdg[ic]['canopy'] = wnode2uvpnode(terms_bdg[ic]['canopy'])
  terms_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
    +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
  terms_bdg[ic]['dissip'] = data[ic]['dissip']



## Smooth terms
for ic in range(nca):
  for key in terms_bdg[ic].keys():
    print(key)
    terms_bdg[ic][key] = gaussian_filter(terms_bdg[ic][key], sigma=(1, 2), mode='wrap')
    print(terms_bdg[ic][key].min(), terms_bdg[ic][key].max())



## Mask the data[ic]
for ic in range(nca):
  for key in terms_bdg[ic].keys():
    terms_bdg[ic][key] = np.ma.array(terms_bdg[ic][key], mask=mask[ic])



## Renormalize the data[ic]
for ic in range(nca):
  total_dissip = -(terms_bdg[ic]['dissip'] + terms_bdg[ic]['canopy'])
  for item in items_bdg:
    terms_bdg[ic][item] /= total_dissip



## Calculate the cancelation level for small divergence area.
mask_divfree = [None for ic in range(nca)]
lvl_ccl = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'flat':
    mask_divfree[ic] = np.abs(terms_bdg[ic]['Ae']) > cri_divfree
    lvl_ccl[ic] =\
      np.ma.array(np.abs(terms_bdg[ic]['adv_v']), mask=mask_divfree[ic])



## Smooth the result over flat terrain
if ic==1:
  for ind in ind_plot:
    terms_bdg[ic][items_bdg[ind]][:, :] =\
      np.mean(terms_bdg[ic][items_bdg[ind]], axis=-1)[:, np.newaxis]

# %%

## Plot the results
# Initlize the figure
#cmap_bdg = cm.get_cmap("viridis")
cmap_bdg = cm.get_cmap("bwr")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)
ifig = 0
fig = plt.figure(ifig, figsize=(8, 6))
nplot = len(ind_plot) * nca
gs = gridspec.GridSpec(1, nplot, width_ratios=[9.5, 0.5])
gs.update(**sz)

for ic in range(nca):
  for inum, ind in enumerate(ind_plot):
    isub = 2*inum+ic
    ax = fig.add_subplot(gs[isub])
    if ic==1:
      box = ax.get_position()
      box.x0 -= sz['wspace']/8
      box.x1 -= sz['wspace']/8
      ax.set_position(box)
    cax=ax.contourf(x_gd[ic], z_gd[ic], terms_bdg[ic][items_bdg[ind]],
      levels_bdg, cmap=cmap_bdg, extend='both')
    #if lab_case[ic] == 'hill':
    #  x_Ae0 = x_gd[ic][iz_Ae0[ic], ix_Ae0[ic]]
    #  z_Ae0 = z_gd[ic][iz_Ae0[ic], ix_Ae0[ic]]
    #  ax.scatter(x_Ae0, z_Ae0, s=ms, color='k')
    if lab_case[ic] != 'flat':
      cs=ax.contour(x_gd[ic], z_gd[ic], lvl_ccl[ic], levels=levels_ccl,
        linewidths=lw-0.5, linestyles='--')
      ax.clabel(cs, inline=True, fontsize='small')
    ax.plot(x_gd[ic][0, :], z_tpg[ic][0, :], ls='-', lw=lw*2, c='k')
    ax.plot(x_gd[ic][0, :], z_tpg[ic][0, :]+hc, ls='--', lw=lw, c='k')
    ax.plot(x_gd[ic][0, :], z_tpg[ic][0, :]+2*hc, ls='--', lw=lw, c='k')
    ax.imshow(~mask[ic], extent=(0, dm[ic].lx, 0, dm[ic].lz), alpha=0.5,
      interpolation='bilinear', cmap='gray', aspect='auto',
      origin='lower')
    
    # Axis property
    ax.set_xlim([0, dm[ic].lx])
    ax.set_ylim([0, ylim_z])
    if ic==0:
      ax.xaxis.set_major_locator(xmaxLocator)
      ax.yaxis.set_major_locator(ymaxLocator)
  
    # Axis label
    if ic==0:
      ax.set_title(labs_bdg[ind])
      if isub >= nplot//2:
        ax.set_xlabel(r'$x (m)$')
      else:
        ax.set_xticklabels([])
      if isub%(nplot//2) == 0:
        ax.set_ylabel('$z (m)$')
      else:
        ax.set_yticklabels([])
    else:
      ax.set_xticks([])
      ax.set_yticks([])
  
    if ic==0:
      ax.text(0, 1.01, '({seq})'.format(seq=chr(97+inum)),
        transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.12
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, 0.2)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

# Save figure
#fig.savefig(fn_budget, dpi=400)
