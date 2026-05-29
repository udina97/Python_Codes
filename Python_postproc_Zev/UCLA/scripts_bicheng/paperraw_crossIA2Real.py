#!/usr/bin/env python3

"""
Program: paperraw_crossIAReal
The crossection of IA terms for the real case
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
case = 'amazon_canopy_real'
lab_case = 'real'
case_flat = 'amazon_canopy_nohill'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fmt_lespathFlat = '/data/2/bzc/LES/amazon_2D/{case:s}'
fmt_base = './data/statistics3D_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_ptb = './data/statistics_pertubation_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_drv = './data/statistics_derivative_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_bdg = './data/statistics_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmtfig_bdg = './figure/paperraw/paperraw_crossIA2{case:s}.png'

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
tts = 144100
tte = 324000
ts = 14400
te = 32400
ix_d = 115
#iy_d = 188
iy_d = 269
Z_prof = np.arange(1, 3*hc+1, 2)
dz = 2
zpad_flat = 5
items = ('u', 'v', 'w','u2', 'v2', 'w2',
  'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('adv_h', 'adv_v', 'canopy', 'dissip')
labs_bdg = (r'$A^h_e$', r'$A^v_e$', r'$\epsilon_c$', r'$\epsilon$')
items_plt = ('IA',)
labs_plt = (r'$|A_e|$',)

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.1, top=0.96,
  wspace=0.1, hspace=0.3)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
levels_bdg = np.arange(0, 0.61, 0.05)
ylim_z = 200



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
terms_plt = dict.fromkeys(items_plt, None)
nplt = len(items_plt)



## Read data
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

fn_base = fmt_base.format(case=case, tts=tts, tte=tte)
fn_bdg = fmt_bdg.format(case=case, tts=tts, tte=tte)

dset_base = np.load(fn_base, mmap_mode='r', allow_pickle=True)
dset_bdg = np.load(fn_bdg, mmap_mode='c', allow_pickle=True)

z_tpg = dset_base['z_tpg']
x_gd = dset_base['x']
y_gd = dset_base['y']
z_gd = dset_base['z']
mask = z_gd<=z_tpg[np.newaxis, :, :]
terms_bdg = dset_bdg['terms'].item()



## Smooth the data
for key in terms_bdg.keys():
  if key != 'sum':
    terms_bdg[key] = gaussian_filter(terms_bdg[key], sigma=(0, 3, 3),
      mode='wrap')



## Calculate terms
terms_plt['IA'] = np.abs(terms_bdg['adv_h'] + terms_bdg['adv_v'])

# Total dissipation rate
terms_bdg['dissip_ttl'] = terms_bdg['dissip'] + terms_bdg['canopy']



## Normalization by total dissipation
for item in items_plt:
  terms_plt[item] /= terms_bdg['dissip_ttl']



## Read data of flat case
lespath_flat = fmt_lespathFlat.format(case=case_flat)
param_flat = lp.lesParam.lesClass.param(lespath_flat)
dm_flat = param_flat.domain

data_flat = dict.fromkeys(items, None)
for item in items:
  data_flat[item], time = lp.io.io_averFile.loadLES_averFile(param_flat,
    qtype=item, tss=ts, tes=te)
  data_flat[item] = np.mean(data_flat[item], axis=0)



## Get the coordinates
z_flat, x_flat = lp.domain.dmFun.xzcoord(dm_flat, ztype='staggered')
z_flat -= zpad_flat
Z_flat = z_flat



## Calculate TKE budget terms
terms_bdg_flat = dict.fromkeys(items_bdg, None)

wn_flat = 2*np.pi*np.fft.rfftfreq(dm_flat.nx, dm_flat.dx/dm_flat.zi)

# Production term
data_flat['uu_t'] = data_flat['u2'] - data_flat['u']**2
data_flat['uu_t'] = gaussian_filter(data_flat['uu_t'], sigma=(1, 1), mode='wrap')
data_flat['vv_t'] = data_flat['v2'] - data_flat['v']**2
data_flat['vv_t'] = gaussian_filter(data_flat['vv_t'], sigma=(1, 1), mode='wrap')
data_flat['ww_t'] = data_flat['w2'] - data_flat['w']**2
data_flat['ww_t'] = gaussian_filter(data_flat['ww_t'], sigma=(1, 1), mode='wrap')

# TKE
tke_flat = (data_flat['uu_t'] + data_flat['vv_t'] + data_flat['ww_t']) / 2

# Advection term
ue = data_flat['u']*tke_flat
duedx = get_dphidx(ue, wn_flat)
tkez = np.zeros(tke_flat.shape)
tkez[0, :] = 0
tkez[1:, :] = 0.5*(tke_flat[:-1, :] + tke_flat[1:, :])
we = data_flat['w']*tkez
dwedz = get_dphidz(we, dm_flat)
terms_bdg_flat['IA'] = np.abs(-duedx-dwedz)

# Dissipation
terms_bdg_flat['canopy'] = data_flat['wFcz']-data_flat['w']*data_flat['Fcz']
terms_bdg_flat['canopy'] = wnode2uvpnode(terms_bdg_flat['canopy'])
terms_bdg_flat['canopy'] += (data_flat['uFcx']-data_flat['u']*data_flat['Fcx'])\
  +(data_flat['vFcy']-data_flat['v']*data_flat['Fcy'])
terms_bdg_flat['dissip'] = data_flat['dissip']



## Mask the data_flat
mask_flat = Z_flat<=0
for item in items_plt:
  terms_bdg_flat[item] = np.ma.array(terms_bdg_flat[item], mask=mask_flat)



## Renormalize the data_flat
for item in items_plt:
  terms_bdg_flat[item] /= -(terms_bdg_flat['dissip'] + terms_bdg_flat['canopy'])



## Smooth the result over flat terrain
for item in items_plt:
  terms_bdg_flat[item][:, :] =\
    np.mean(terms_bdg_flat[item], axis=-1)[:, np.newaxis]



## Plot the results
# Initlize the figure
cmap_bdg = cm.get_cmap("viridis")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

# Plot the budget terms

# x-cross-section
ifig = 0
fig = plt.figure(ifig, figsize=(8, 8))
gs = gridspec.GridSpec(nplt*2, 2, width_ratios=[9.5, 0.5])
gs.update(**sz)

for it, item in enumerate(items_plt):
  ax = fig.add_subplot(gs[it, 0])
  cax=ax.contourf(x_gd[:, iy_d, :], z_gd[:, iy_d, :],
    np.ma.array(-terms_plt[item][:, iy_d, :], mask=mask[:, iy_d, :]),
    levels_bdg, cmap=cmap_bdg, extend='both')
  #ax.streamplot(x_gd[:, iy_d, :], z_gd[:, iy_d, :],
  #  u_gd[:, iy_d, :], w_gd[:, iy_d, :], color='gray', density=2,
  #  linewidth=lw-0.5, arrowsize=ars)
  ax.plot(x_gd[0, iy_d, :], z_tpg[iy_d, :], ls='-', lw=lw*2, c='k')
  ax.plot(x_gd[0, iy_d, :], z_tpg[iy_d, :]+hc, ls='--', lw=lw, c='k')
  ax.plot(x_gd[0, iy_d, :], z_tpg[iy_d, :]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow(~mask[:, iy_d, :], extent=(0, dm.lx, 0, dm.lz), alpha=0.5,
    interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim([0, dm.lx])
  ax.set_ylim([0, ylim_z])

  # Axis label
  ax.set_title(labs_plt[it])
  ax.set_xlabel(r'$x (m)$')
  ax.set_ylabel(r'$z (m)$')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+2*it)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

  # Flat case
  ax = fig.add_subplot(gs[it, 1])
  Z_shift = Z_flat + z_tpg[iy_d, -1]
  mask_flat = Z_shift<=z_tpg[iy_d, -1]
  terms_bdg_flat[item] = np.ma.array(terms_bdg_flat[item], mask=mask_flat)
  if item=='res':
    cax=ax.contourf(x_flat, Z_shift, terms_bdg_flat[item],
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    cax=ax.contourf(x_flat, Z_shift, -terms_bdg_flat[item],
      levels_bdg, cmap=cmap_bdg, extend='both')
  ax.axhline(y=z_tpg[iy_d, -1], ls='-', lw=lw*2, c='k')
  ax.axhline(y=z_tpg[iy_d, -1]+hc, ls='--', lw=lw, c='k')
  ax.axhline(y=z_tpg[iy_d, -1]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow([[False, False], [False, False]],
    extent=(0, dm_flat.lx, 0, z_tpg[iy_d, -1]),
    alpha=0.5, interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_ylim([0, ylim_z])

# y-cross-section
for it, item in enumerate(items_plt):
  ax = fig.add_subplot(gs[it+1, 0])
  if item=='res':
    cax=ax.contourf(y_gd[:, :, ix_d], z_gd[:, :, ix_d],
      np.ma.array(terms_plt[item][:, :, ix_d], mask=mask[:, :, ix_d]),
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    cax=ax.contourf(y_gd[:, :, ix_d], z_gd[:, :, ix_d],
      np.ma.array(-terms_plt[item][:, :, ix_d], mask=mask[:, :, ix_d]),
      levels_bdg, cmap=cmap_bdg, extend='both')
  #ax.streamplot(y_gd[:, :, ix_d], z_gd[:, :, ix_d],
  #  v_gd[:, :, ix_d], w_gd[:, :, ix_d], color='gray', density=2,
  #  linewidth=lw-0.5, arrowsize=ars)
  ax.plot(y_gd[0, :, ix_d], z_tpg[:, ix_d], ls='-', lw=lw*2, c='k')
  ax.plot(y_gd[0, :, ix_d], z_tpg[:, ix_d]+hc, ls='--', lw=lw, c='k')
  ax.plot(y_gd[0, :, ix_d], z_tpg[:, ix_d]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow(~mask[:, :, ix_d], extent=(0, dm.ly, 0, dm.lz), alpha=0.5,
    interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim([0, dm.ly])
  ax.set_ylim([0, ylim_z])

  # Axis label
  ax.set_title(labs_plt[it])
  ax.set_xlabel(r'$y (m)$')
  ax.set_ylabel(r'$z (m)$')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+2*it+1)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

  # Flat case
  ax = fig.add_subplot(gs[it+1, 1])
  Z_shift = Z_flat + z_tpg[-1, ix_d]
  mask_flat = Z_shift<=z_tpg[-1, ix_d]
  terms_bdg_flat[item] = np.ma.array(terms_bdg_flat[item], mask=mask_flat)
  cax=ax.contourf(x_flat, Z_shift, terms_bdg_flat[item],
    levels_bdg, cmap=cmap_bdg, extend='both')
  ax.axhline(y=z_tpg[-1, ix_d], ls='-', lw=lw*2, c='k')
  ax.axhline(y=z_tpg[-1, ix_d]+hc, ls='--', lw=lw, c='k')
  ax.axhline(y=z_tpg[-1, ix_d]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow([[False, False], [False, False]],
    extent=(0, dm_flat.lx, 0, z_tpg[-1, ix_d]),
    alpha=0.5, interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_ylim([0, ylim_z])

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.07
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, 0.3)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

fnfig_bdg = fmtfig_bdg.format(case=lab_case.capitalize())
fig.savefig(fnfig_bdg, dpi=400)
