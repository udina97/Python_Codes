#!/usr/bin/env python3

"""
Program: paperraw_crossTKEbudgetReal
The crossection of TKE budget terms for the real case
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
nm_case = 'Real'

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
fmt_lespathFlat = '/data/2/bzc/LES/amazon_2D/{case:s}'
fmt_base = './data/statistics3D_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_ptb = './data/statistics_pertubation_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_drv = './data/statistics_derivative_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_bdg = './data/statistics_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_bdgProf = './data/budget_crest_wid{wid:02d}_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmtfig_bdg = './figure/paperraw/paperraw_crossTKEbudget{case:s}.png'

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
tts = 144100
tte = 324000
#tte = 216000
ts = 14400
te = 32400
#te = 21600
ix_d = 115
iy_d = 188
Z_prof = np.arange(1, 3*hc+1, 2)
dz = 2
zpad_flat = 5
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'canopy', 'dissip', 'sum', 'prod_dudz')
labs_bdg = (r'$A^h_e$', r'$A^v_e$', r'$P^h$', r'$P^v$', r'$T^h_e$', r'$T^v_e$',
  r'$\Pi^h_e$', r'$\Pi^v_e$', r'$\epsilon_c$', r'$\epsilon$', 'sum', 
  r'$P_{homo}$')
items_plt = ('res', 'adv', 'uturb', 'pturb')
labs_plt = (r'$R$',r'$-A_e$', r'$-T_e$', r'$-\Pi_e$')

# Figure
sz = dict(left=0.07, right=0.98, bottom=0.1, top=0.96,
  wspace=0.1, hspace=0.3)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
levels_bdg = np.arange(-1.5, 1.5+0.1, 0.1)
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
#data = dset_base['data'].item()
#u_gd = data['u']
#v_gd = data['v']
#w_gd = data['w']
mask = z_gd<=z_tpg[np.newaxis, :, :]
terms_bdg = dset_bdg['terms'].item()



## Smooth the data
for key in terms_bdg.keys():
  if key != 'sum':
    terms_bdg[key] = gaussian_filter(terms_bdg[key], sigma=(0, 3, 3),
      mode='wrap')



## Calculate terms
terms_plt['res'] = terms_bdg['prod_h'] + terms_bdg['prod_v']\
  + terms_bdg['dissip'] + terms_bdg['canopy']
terms_plt['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v']
terms_plt['uturb'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']
terms_plt['pturb'] = terms_bdg['pturb_h'] + terms_bdg['pturb_v']

# Total dissipation rate
terms_bdg['dissip_ttl'] = terms_bdg['dissip'] + terms_bdg['canopy']



## Normalization by total dissipation
for item in items_plt:
  terms_plt[item] /= -terms_bdg['dissip_ttl']



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
data_flat['dwdx'] = get_dphidx(data_flat['w'], wn_flat)
data_flat['dwdx'] = gaussian_filter(data_flat['dwdx'], sigma=(1, 1), mode='wrap')

data_flat['dudx'] = get_dphidx(data_flat['u'], wn_flat)
data_flat['dudx'][1:, :] = 0.5 * (data_flat['dudx'][:-1, :]
  +data_flat['dudx'][1:, :])
data_flat['dudx'][0, :] = 0
data_flat['dudx'] = gaussian_filter(data_flat['dudx'], sigma=(1, 1),
  mode='wrap')

data_flat['dudz'] = np.zeros(data_flat['u'].shape) 
data_flat['dudz'][1:, :] =\
  (data_flat['u'][1:, :]-data_flat['u'][:-1, :]) / (dm_flat.dz/dm_flat.zi)
data_flat['dudz'] = gaussian_filter(data_flat['dudz'], sigma=(1, 1), mode='wrap')

data_flat['dwdz'] = np.zeros(data_flat['w'].shape) 
data_flat['dwdz'][1:-1, :] =\
  (data_flat['w'][2:, :]-data_flat['w'][:-2, :]) / (2*dm_flat.dz/dm_flat.zi)
data_flat['dwdz'] = gaussian_filter(data_flat['dwdz'], sigma=(1, 1), mode='wrap')

data_flat['uw_t'] = data_flat['uw'] - data_flat['u']*data_flat['w']
data_flat['uw_t'] = gaussian_filter(data_flat['uw_t'], sigma=(1, 1), mode='wrap')
data_flat['uu_t'] = data_flat['u2'] - data_flat['u']**2
data_flat['uu_t'] = gaussian_filter(data_flat['uu_t'], sigma=(1, 1), mode='wrap')
data_flat['vv_t'] = data_flat['v2'] - data_flat['v']**2
data_flat['vv_t'] = gaussian_filter(data_flat['vv_t'], sigma=(1, 1), mode='wrap')
data_flat['ww_t'] = data_flat['w2'] - data_flat['w']**2
data_flat['ww_t'] = gaussian_filter(data_flat['ww_t'], sigma=(1, 1), mode='wrap')

# TKE
tke_flat = (data_flat['uu_t'] + data_flat['vv_t'] + data_flat['ww_t']) / 2

# Production
terms_bdg_flat['prod'] = -data_flat['uu_t']*data_flat['dudx']\
  - data_flat['ww_t']*data_flat['dwdz']\
  - data_flat['uw_t']*(data_flat['dudz']+data_flat['dwdx'])

# Advection term
ue = data_flat['u']*tke_flat
duedx = get_dphidx(ue, wn_flat)
tkez = np.zeros(tke_flat.shape)
tkez[0, :] = 0
tkez[1:, :] = 0.5*(tke_flat[:-1, :] + tke_flat[1:, :])
we = data_flat['w']*tkez
dwedz = get_dphidz(we, dm_flat)
terms_bdg_flat['adv'] = -duedx-dwedz

# Turbulent transport term
turb_u3 = data_flat['u3'] - 3*data_flat['u']*data_flat['u2']+2*data_flat['u']**3
turb_uv2 = data_flat['uv2'] - 2*data_flat['v']*data_flat['uv']\
  + 2*data_flat['u']*data_flat['v']**2-data_flat['u']*data_flat['v2']
uw = wnode2uvpnode(data_flat['uw'])
w_c = wnode2uvpnode(data_flat['w'])
w2_c = wnode2uvpnode(data_flat['w2'])
turb_uw2 = data_flat['uw2'] - 2*w_c*uw\
  + 2*data_flat['u']*w_c**2-data_flat['u']*w2_c
ue = 0.5*(turb_u3 + turb_uv2 + turb_uw2)
duedx = get_dphidx(ue, wn_flat)
u_w = uvpnode2wnode(data_flat['u'])
u2_w = uvpnode2wnode(data_flat['u2'])
turb_wu2 = data_flat['u2w'] - 2*u_w*data_flat['uw']\
  + 2*data_flat['w']*u_w**2 - data_flat['w']*u2_w
v_w = uvpnode2wnode(data_flat['v'])
v2_w = uvpnode2wnode(data_flat['v2'])
turb_wv2 = data_flat['v2w'] - 2*v_w*data_flat['vw']\
  + 2*data_flat['w']*v_w**2 - data_flat['w']*v2_w
turb_w3 = data_flat['w3'] - 3*data_flat['w']*data_flat['w2'] + 2*data_flat['w']**3
we = 0.5*(turb_wu2 + turb_wv2 + turb_w3)
dwedz = get_dphidz(we, dm_flat)
terms_bdg_flat['uturb'] = -duedx-dwedz

utxx = data_flat['utxx'] - data_flat['u']*data_flat['txx']
vtxy = data_flat['vtxy'] - data_flat['v']*data_flat['txy']
wtxz = data_flat['wtxz'] - data_flat['w']*data_flat['txz']
wtxz_uv = wnode2uvpnode(wtxz)
dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn_flat)
utxz = data_flat['utxz'] - u_w*data_flat['txz']
vtyz = data_flat['vtyz'] - v_w*data_flat['tyz']
wtzz = data_flat['wtzz'] - data_flat['w']*data_flat['tzz']
dutaudz = get_dphidz(utxz+vtyz+wtzz, dm_flat)
terms_bdg_flat['uturb'] -= dutaudx+dutaudz

# Pressure transport term
pu = data_flat['pu'] - data_flat['p']*data_flat['u']
dpudx = get_dphidx(pu, wn_flat)
p_w = uvpnode2wnode(data_flat['p'])
pw = data_flat['pw'] - p_w*data_flat['w']
dpwdz = get_dphidz(pw, dm_flat)
terms_bdg_flat['pturb'] = -dpudx-dpwdz

# Dissipation
terms_bdg_flat['canopy'] = data_flat['wFcz']-data_flat['w']*data_flat['Fcz']
terms_bdg_flat['canopy'] = wnode2uvpnode(terms_bdg_flat['canopy'])
terms_bdg_flat['canopy'] += (data_flat['uFcx']-data_flat['u']*data_flat['Fcx'])\
  +(data_flat['vFcy']-data_flat['v']*data_flat['Fcy'])
terms_bdg_flat['dissip'] = data_flat['dissip']

# Residual term
terms_bdg_flat['res'] = terms_bdg_flat['prod'] + terms_bdg_flat['dissip'] + terms_bdg_flat['canopy']

# Sum of above terms
terms_bdg_flat['sum'] = terms_bdg_flat['prod'] + terms_bdg_flat['dissip']\
  + terms_bdg_flat['canopy'] + terms_bdg_flat['adv'] + terms_bdg_flat['uturb']\
  + terms_bdg_flat['pturb'] 



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
cmap_bdg = cm.get_cmap("seismic")
cmap_bdg = cmap_bdg(np.linspace(0, 1, levels_bdg.size))
half = levels_bdg.size // 2
cmap_bdg[half-1:half+2, :] = 1
print(cmap_bdg)
cmap_bdg = colors.ListedColormap(cmap_bdg)
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

# Plot the budget terms

# x-cross-section
ifig = 0
fig = plt.figure(ifig, figsize=(8, 10))
gs = gridspec.GridSpec(nplt, 6, width_ratios=[9.5, 0.5, 0.2, 9.5, 0.5, 0.2])
gs.update(**sz)

for it, item in enumerate(items_plt):
  ax = fig.add_subplot(gs[it, 0])
  if item=='res':
    cax=ax.contourf(x_gd[:, iy_d, :], z_gd[:, iy_d, :],
      np.ma.array(terms_plt[item][:, iy_d, :], mask=mask[:, iy_d, :]),
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
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
  if it == nplt-1:
    ax.set_xlabel(r'$x (m)$')
  else:
    ax.set_xticklabels([])
  ax.set_ylabel('$z (m)$')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+2*it)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

  fn_npz = "./data/forpaper/fig5_crossTKEbudget_{case:s}_iy{iy:03d}".format(case=nm_case, iy=iy_d)
  terms = dict.fromkeys(items_plt, None)
  for item in items_plt:
    if item=='res':
      terms[item] = -terms_plt[item][:, iy_d, :]
    else:
      terms[item] = terms_plt[item][:, iy_d, :]
  np.savez(fn_npz, x=x_gd[:, iy_d, :], z=z_gd[:, iy_d, :], terms=terms)

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
  ax = fig.add_subplot(gs[it, 3])
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
  if it == nplt-1:
    ax.set_xlabel(r'$y (m)$')
  else:
    ax.set_xticklabels([])
  ax.set_yticklabels([])

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+2*it+1)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

  fn_npz = "./data/forpaper/fig5_crossTKEbudget_{case:s}_ix{ix:03d}".format(case=nm_case, ix=ix_d)
  terms = dict.fromkeys(items_plt, None)
  for item in items_plt:
    if item=='res':
      terms[item] = -terms_plt[item][:, :, ix_d]
    else:
      terms[item] = terms_plt[item][:, :, ix_d]
  np.savez(fn_npz, y=y_gd[:, :, ix_d], z=z_gd[:, :, ix_d], terms=terms)

  # Flat case
  ax = fig.add_subplot(gs[it, 4])
  Z_shift = Z_flat + z_tpg[-1, ix_d]
  mask_flat = Z_shift<=z_tpg[-1, ix_d]
  terms_bdg_flat[item] = np.ma.array(terms_bdg_flat[item], mask=mask_flat)
  if item=='res':
    cax=ax.contourf(x_flat, Z_shift, terms_bdg_flat[item],
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    cax=ax.contourf(x_flat, Z_shift, -terms_bdg_flat[item],
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
