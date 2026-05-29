#!/usr/bin/env python3

"""
Program: paperraw_crossTKEbudgetFlat
The crossection of TKE budget terms for the flat case
"""

### Histories:
### 07/21/2020 -- Bicheng Chen (chabby@ucla.edu) -- First created



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
fmt_budget = './figure/paperraw/supplementraw_crossTKEbudget{case:s}.png'

# Topography parameters
zpad = 5
amp = 0
wl = 1000
lh = wl / 4

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 14400
te = 21600
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('prod', 'dissip', 'canopy', 'res', 'adv', 'uturb', 'pturb', 'sum')
labs_bdg = (r'$P$', r'$\epsilon$', r'$\epsilon_c$', r'$R$',r'$-A_e$', r'$-T_e$',
  r'$-\Pi_e$', 'sum')
ind_plot = [3, 4, 5 ,6]

# Figure
sz = dict(left=0.08, right=0.98, bottom=0.17, top=0.95,
  wspace=0.2, hspace=0.3)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
ylim_z = 160
levels_bdg = np.arange(-1.5, 1.5+0.1, 0.1)



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
data = dict.fromkeys(items, None)
terms_bdg = dict.fromkeys(items_bdg, None)



## Read data and calculate the temporal mean
lespath = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(lespath)
dm = param.domain

for item in items:
  data[item], time = lp.io.io_averFile.loadLES_averFile(param,
    qtype=item, tss=ts, tes=te)
  data[item] = np.mean(data[item], axis=0)



## Get the coordinates
z_gd, x_gd = lp.domain.dmFun.xzcoord(dm, ztype='staggered')
z_tpg = get_topo(x_gd, wl, amp)
z_gd -= zpad
Z_gd = z_gd-z_tpg



## Calculate TKE budget terms
wn = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)

# Production term
data['dwdx'] = get_dphidx(data['w'], wn)
data['dwdx'] = gaussian_filter(data['dwdx'], sigma=(1, 1), mode='wrap')

data['dudx'] = get_dphidx(data['u'], wn)
data['dudx'][1:, :] = 0.5 * (data['dudx'][:-1, :]
  +data['dudx'][1:, :])
data['dudx'][0, :] = 0
data['dudx'] = gaussian_filter(data['dudx'], sigma=(1, 1), mode='wrap')

data['dudz'] = np.zeros(data['u'].shape) 
data['dudz'][1:, :] =\
  (data['u'][1:, :]-data['u'][:-1, :]) / (dm.dz/dm.zi)
data['dudz'] = gaussian_filter(data['dudz'], sigma=(1, 1), mode='wrap')

data['dwdz'] = np.zeros(data['w'].shape) 
data['dwdz'][1:-1, :] =\
  (data['w'][2:, :]-data['w'][:-2, :]) / (2*dm.dz/dm.zi)
data['dwdz'] = gaussian_filter(data['dwdz'], sigma=(1, 1), mode='wrap')

data['uw_t'] = data['uw'] - data['u']*data['w']
data['uw_t'] = gaussian_filter(data['uw_t'], sigma=(1, 1), mode='wrap')
data['uu_t'] = data['u2'] - data['u']**2
data['uu_t'] = gaussian_filter(data['uu_t'], sigma=(1, 1), mode='wrap')
data['vv_t'] = data['v2'] - data['v']**2
data['vv_t'] = gaussian_filter(data['vv_t'], sigma=(1, 1), mode='wrap')
data['ww_t'] = data['w2'] - data['w']**2
data['ww_t'] = gaussian_filter(data['ww_t'], sigma=(1, 1), mode='wrap')

# TKE
tke = (data['uu_t'] + data['vv_t'] + data['ww_t']) / 2

# Production
terms_bdg['prod'] = -data['uu_t']*data['dudx']\
  - data['ww_t']*data['dwdz']\
  - data['uw_t']*(data['dudz']+data['dwdx'])

# Advection term
ue = data['u']*tke
duedx = get_dphidx(ue, wn)
tkez = np.zeros(tke.shape)
tkez[0, :] = 0
tkez[1:, :] = 0.5*(tke[:-1, :] + tke[1:, :])
we = data['w']*tkez
dwedz = get_dphidz(we, dm)
terms_bdg['adv'] = -duedx-dwedz

# Turbulent transport term
turb_u3 = data['u3'] - 3*data['u']*data['u2']+2*data['u']**3
turb_uv2 = data['uv2'] - 2*data['v']*data['uv']\
  + 2*data['u']*data['v']**2-data['u']*data['v2']
uw = wnode2uvpnode(data['uw'])
w_c = wnode2uvpnode(data['w'])
w2_c = wnode2uvpnode(data['w2'])
turb_uw2 = data['uw2'] - 2*w_c*uw\
  + 2*data['u']*w_c**2-data['u']*w2_c
ue = 0.5*(turb_u3 + turb_uv2 + turb_uw2)
duedx = get_dphidx(ue, wn)
u_w = uvpnode2wnode(data['u'])
u2_w = uvpnode2wnode(data['u2'])
turb_wu2 = data['u2w'] - 2*u_w*data['uw']\
  + 2*data['w']*u_w**2 - data['w']*u2_w
v_w = uvpnode2wnode(data['v'])
v2_w = uvpnode2wnode(data['v2'])
turb_wv2 = data['v2w'] - 2*v_w*data['vw']\
  + 2*data['w']*v_w**2 - data['w']*v2_w
turb_w3 = data['w3'] - 3*data['w']*data['w2'] + 2*data['w']**3
we = 0.5*(turb_wu2 + turb_wv2 + turb_w3)
dwedz = get_dphidz(we, dm)
terms_bdg['uturb'] = -duedx-dwedz

utxx = data['utxx'] - data['u']*data['txx']
vtxy = data['vtxy'] - data['v']*data['txy']
wtxz = data['wtxz'] - data['w']*data['txz']
wtxz_uv = wnode2uvpnode(wtxz)
dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn)
utxz = data['utxz'] - u_w*data['txz']
vtyz = data['vtyz'] - v_w*data['tyz']
wtzz = data['wtzz'] - data['w']*data['tzz']
dutaudz = get_dphidz(utxz+vtyz+wtzz, dm)
terms_bdg['uturb'] -= dutaudx+dutaudz

# Pressure transport term
pu = data['pu'] - data['p']*data['u']
dpudx = get_dphidx(pu, wn)
p_w = uvpnode2wnode(data['p'])
pw = data['pw'] - p_w*data['w']
dpwdz = get_dphidz(pw, dm)
terms_bdg['pturb'] = -dpudx-dpwdz

# Dissipation
data['dudz'] = get_dphidz(data['u'], dm)
data['dudz'] = wnode2uvpnode(data['dudz'])

data['dvdx'] = get_dphidx(data['v'], wn)
data['dvdz'] = get_dphidz(data['v'], dm)
data['dvdz'] = wnode2uvpnode(data['dvdz'])

data['dwdz'] = get_dphidz(data['w'], dm)

s11 = data['dudx']
s12 = 0.5*data['dvdx']
s13 = 0.5*(data['dudz'] + data['dwdx'])
s22 = 0
s23 = 0.5*data['dvdz']
s33 = data['dwdz']

terms_bdg['canopy'] = (data['wFcz']-data['w']*data['Fcz'])
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data['uFcx']-data['u']*data['Fcx'])\
  +(data['vFcy']-data['v']*data['Fcy'])
terms_bdg['dissip'] = data['dissip']\
  - data['txx']*s11 - data['tyy']*s22 - data['tzz']*s33\
  - 2*data['txy']*s12 - 2*data['txz']*s13 - 2*data['tyz']*s23

# Residual term
terms_bdg['res'] = terms_bdg['prod'] + terms_bdg['dissip'] + terms_bdg['canopy']

# Sum of above terms
terms_bdg['sum'] = terms_bdg['prod'] + terms_bdg['dissip']\
  + terms_bdg['canopy'] + terms_bdg['adv'] + terms_bdg['uturb']\
  + terms_bdg['pturb'] 



## Smooth terms
for key in terms_bdg.keys():
  print(key)
  terms_bdg[key] = gaussian_filter(terms_bdg[key], sigma=(0, 2), mode='wrap')



## Mask the data
mask = Z_gd<=0
for key in terms_bdg.keys():
  terms_bdg[key] = np.ma.array(terms_bdg[key], mask=mask)



## Renomalize the data
for ind in ind_plot:
  terms_bdg[items_bdg[ind]] /= -(terms_bdg['dissip']+terms_bdg['canopy'])



## Plot the results
# Initlize the figure
cmap_bdg = cm.get_cmap("seismic")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)
ifig = 0
fig = plt.figure(ifig, figsize=(8, 6))
nplot = len(ind_plot)
if nplot == 1:
  gs = gridspec.GridSpec(1, 1)
else:
  gs = gridspec.GridSpec((nplot+1)//2, 2)
gs.update(**sz)

for isub, ind in enumerate(ind_plot):
  ax = fig.add_subplot(gs[isub])
  if ind == 3:
    cax=ax.contourf(x_gd, z_gd, terms_bdg[items_bdg[ind]],
      levels_bdg, cmap=cmap_bdg, extend='both')
  else:
    cax=ax.contourf(x_gd, z_gd, -terms_bdg[items_bdg[ind]],
      levels_bdg, cmap=cmap_bdg, extend='both')
  #ax.streamplot(x_gd, z_gd,
  #  data['u'], data['w'], color='gray', density=2, linewidth=1, arrowsize=ars)
  ax.plot(x_gd[0, :], z_tpg[0, :], ls='-', lw=lw*2, c='k')
  ax.plot(x_gd[0, :], z_tpg[0, :]+hc, ls='--', lw=lw, c='k')
  ax.plot(x_gd[0, :], z_tpg[0, :]+2*hc, ls='--', lw=lw, c='k')
  ax.imshow(~mask, extent=(0, dm.lx, 0, dm.lz), alpha=0.5,
    interpolation='bilinear', cmap='gray', aspect='auto',
    origin='lower')
  
  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim([0, dm.lx])
  ax.set_ylim([0, ylim_z])

  # Axis label
  ax.set_title(labs_bdg[ind])
  if isub >= nplot-2:
    ax.set_xlabel(r'$x (m)$')
  else:
    ax.set_xticklabels([])
  if isub%2 == 0:
    ax.set_ylabel('$z (m)$')
  else:
    ax.set_yticklabels([])

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+isub)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

# Colorbar
grid_pos=gs.get_grid_positions(fig=fig)
left = grid_pos[2][0]
bottom = grid_pos[0][-1]
lpad = 0.2
bpad = -0.12
cbar_ax = fig.add_axes([left+lpad, bottom+bpad, 0.45, 0.02])
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, 0.3)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

# Save figure
fn_fig = fmt_budget.format(case=lab_case.capitalize())
fig.savefig(fn_fig, dpi=400)
