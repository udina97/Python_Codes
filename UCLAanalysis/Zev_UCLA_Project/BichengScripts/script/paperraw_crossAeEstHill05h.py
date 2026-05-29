#!/usr/bin/env python3

"""
Program: paperraw_crossAeEstHill05h
The crossection of TKE budget terms for the hill05h case
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
case = ('amazon_canopy_hill05h', 'amazon_canopy_nohill')
lab_case = ('hill05h', 'flat')

# File format
fmt_lespath = '/data/2/bzc/LES/amazon_2D/{case:s}'
fn_budget = './figure/paperraw/paperraw_crossAeEstHill05h.png'

# Topography parameters
zpad = (0.5, 5)
amp = (12.5, 0)
wl = 1000

# Canopy parameters
hc = 39
ustar = 0.4

# Data range
ts = 18000
te = 25200
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
items_bdg = ('prod', 'dissip', 'canopy', 'res', 'adv_h', 'adv_v', 'adv', 'uturb', 'pturb', 'res+uturb', 'sum')
labs_bdg = (r'$P$', r'$\epsilon$', r'$\epsilon_c$', r'$R$', r'$-A^h_e$',
  r'$-A^v_e$', r'$A_e$', r'$-T_e$', r'$-\Pi_e$', r'$-R-T_e$', 'sum')
ind_plot = [6, 9]

# Ae cancelation
cri_divfree = 0.1

# Figure
sz = dict(left=0.08, right=0.98, bottom=0.17, top=0.95,
  wspace=0.1, hspace=0.2)
lw = 1.5
ars = 1
xspacing = 500
yspacing = 20
ylim_z = 200
levels_bdg = np.arange(-1.5, 1.5+0.1, 0.1)
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
    data[ic][item], time = lp.io.io_averFile.loadLES_averFile(param[ic],
      qtype=item, tss=ts, tes=te)
    data[ic][item] = np.mean(data[ic][item], axis=0)


## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]

for ic in range(nca):
  z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
  z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
  z_gd[ic] -= zpad[ic]
  Z_gd[ic] = z_gd[ic]-z_tpg[ic]



## Calculate TKE budget terms
wn = [None for ic in range(nca)]
for ic in range(nca):
  wn[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)

for ic in range(nca):
  # Production term
  data[ic]['dwdx'] = get_dphidx(data[ic]['w'], wn[ic])
  data[ic]['dwdx'] = gaussian_filter(data[ic]['dwdx'], sigma=(1, 1), mode='wrap')
  
  data[ic]['dudx'] = get_dphidx(data[ic]['u'], wn[ic])
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
  
  # TKE
  tke[ic] = (data[ic]['uu_t'] + data[ic]['vv_t'] + data[ic]['ww_t']) / 2
  
  # Production
  terms_bdg[ic]['prod'] = -data[ic]['uu_t']*data[ic]['dudx']\
    - data[ic]['ww_t']*data[ic]['dwdz']\
    - data[ic]['uw_t']*(data[ic]['dudz']+data[ic]['dwdx'])
  
  # Advection term
  ue = data[ic]['u']*tke[ic]
  duedx = get_dphidx(ue, wn[ic])
  tkez = np.zeros(tke[ic].shape)
  tkez[0, :] = 0
  tkez[1:, :] = 0.5*(tke[ic][:-1, :] + tke[ic][1:, :])
  we = data[ic]['w']*tkez
  dwedz = get_dphidz(we, dm[ic])
  terms_bdg[ic]['adv_h'] = -duedx
  terms_bdg[ic]['adv_v'] = dwedz
  terms_bdg[ic]['adv'] = -duedx-dwedz
  
  # Turbulent transport term
  turb_u3 = data[ic]['u3'] - 3*data[ic]['u']*data[ic]['u2']+2*data[ic]['u']**3
  turb_uv2 = data[ic]['uv2'] - 2*data[ic]['v']*data[ic]['uv']\
    + 2*data[ic]['u']*data[ic]['v']**2-data[ic]['u']*data[ic]['v2']
  uw = wnode2uvpnode(data[ic]['uw'])
  w_c = wnode2uvpnode(data[ic]['w'])
  w2_c = wnode2uvpnode(data[ic]['w2'])
  turb_uw2 = data[ic]['uw2'] - 2*w_c*uw\
    + 2*data[ic]['u']*w_c**2-data[ic]['u']*w2_c
  ue = 0.5*(turb_u3 + turb_uv2 + turb_uw2)
  duedx = get_dphidx(ue, wn[ic])
  u_w = uvpnode2wnode(data[ic]['u'])
  u2_w = uvpnode2wnode(data[ic]['u2'])
  turb_wu2 = data[ic]['u2w'] - 2*u_w*data[ic]['uw']\
    + 2*data[ic]['w']*u_w**2 - data[ic]['w']*u2_w
  v_w = uvpnode2wnode(data[ic]['v'])
  v2_w = uvpnode2wnode(data[ic]['v2'])
  turb_wv2 = data[ic]['v2w'] - 2*v_w*data[ic]['vw']\
    + 2*data[ic]['w']*v_w**2 - data[ic]['w']*v2_w
  turb_w3 = data[ic]['w3'] - 3*data[ic]['w']*data[ic]['w2'] + 2*data[ic]['w']**3
  we = 0.5*(turb_wu2 + turb_wv2 + turb_w3)
  dwedz = get_dphidz(we, dm[ic])
  terms_bdg[ic]['uturb'] = -duedx-dwedz
  
  utxx = data[ic]['utxx'] - data[ic]['u']*data[ic]['txx']
  vtxy = data[ic]['vtxy'] - data[ic]['v']*data[ic]['txy']
  wtxz = data[ic]['wtxz'] - data[ic]['w']*data[ic]['txz']
  wtxz_uv = wnode2uvpnode(wtxz)
  dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn[ic])
  utxz = data[ic]['utxz'] - u_w*data[ic]['txz']
  vtyz = data[ic]['vtyz'] - v_w*data[ic]['tyz']
  wtzz = data[ic]['wtzz'] - data[ic]['w']*data[ic]['tzz']
  dutaudz = get_dphidz(utxz+vtyz+wtzz, dm[ic])
  terms_bdg[ic]['uturb'] -= dutaudx+dutaudz
  
  # Pressure transport term
  pu = data[ic]['pu'] - data[ic]['p']*data[ic]['u']
  dpudx = get_dphidx(pu, wn[ic])
  p_w = uvpnode2wnode(data[ic]['p'])
  pw = data[ic]['pw'] - p_w*data[ic]['w']
  dpwdz = get_dphidz(pw, dm[ic])
  terms_bdg[ic]['pturb'] = -dpudx-dpwdz

  # Dissipation
  terms_bdg[ic]['canopy'] = data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz']
  terms_bdg[ic]['canopy'] = wnode2uvpnode(terms_bdg[ic]['canopy'])
  terms_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
    +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
  terms_bdg[ic]['dissip'] = data[ic]['dissip']
  
  # Residual term
  terms_bdg[ic]['res'] = terms_bdg[ic]['prod'] + terms_bdg[ic]['dissip'] + terms_bdg[ic]['canopy']
  
  # R-T_e term
  terms_bdg[ic]['res+uturb'] = terms_bdg[ic]['res'] + terms_bdg[ic]['uturb']
  
  # Sum of above terms
  terms_bdg[ic]['sum'] = terms_bdg[ic]['prod'] + terms_bdg[ic]['dissip']\
    + terms_bdg[ic]['canopy'] + terms_bdg[ic]['adv'] + terms_bdg[ic]['uturb']\
    + terms_bdg[ic]['pturb'] 



## Smooth terms
for ic in range(nca):
  for key in terms_bdg[ic].keys():
    print(key)
    terms_bdg[ic][key] = gaussian_filter(terms_bdg[ic][key], sigma=(0, 2), mode='wrap')
    print(terms_bdg[ic][key].min(), terms_bdg[ic][key].max())



## Mask the data[ic]
mask = [None for ic in range(nca)]
for ic in range(nca):
  mask[ic] = Z_gd[ic]<=0
  for key in terms_bdg[ic].keys():
    terms_bdg[ic][key] = np.ma.array(terms_bdg[ic][key], mask=mask[ic])



## Renormalize the data[ic]
for ic in range(nca):
  total_dissip = -(terms_bdg[ic]['dissip'] + terms_bdg[ic]['canopy'])
  for item in items_bdg:
    terms_bdg[ic][item] /= total_dissip



### Calculate the cancelation level for small divergence area.
#mask_divfree = [None for ic in range(nca)]
#lvl_ccl = [None for ic in range(nca)]
#for ic in range(nca):
#  if lab_case[ic] == 'hill':
#    mask_divfree[ic] = np.abs(terms_bdg[ic]['adv']) > cri_divfree
#    lvl_ccl[ic] =\
#      np.ma.array(np.abs(terms_bdg[ic]['adv_v']), mask=mask_divfree[ic])



## Smooth the result over flat terrain
if ic==1:
  for ind in ind_plot:
    terms_bdg[ic][items_bdg[ind]][:, :] =\
      np.mean(terms_bdg[ic][items_bdg[ind]], axis=-1)[:, np.newaxis]



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
ifig = 0
fig = plt.figure(ifig, figsize=(8, 4))
nplot = len(ind_plot) * nca
gs = gridspec.GridSpec(1, nplot, width_ratios=[9.5, 0.5, 9.5, 0.5])
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
    if ind == 9:
      cax=ax.contourf(x_gd[ic], z_gd[ic], -terms_bdg[ic][items_bdg[ind]],
        levels_bdg, cmap=cmap_bdg, extend='both')
    else:
      cax=ax.contourf(x_gd[ic], z_gd[ic], terms_bdg[ic][items_bdg[ind]],
        levels_bdg, cmap=cmap_bdg, extend='both')
    #if lab_case[ic] == 'hill':
    #  cs=ax.contour(x_gd[ic], z_gd[ic], lvl_ccl[ic], levels=levels_ccl,
    #    linewidths=lw-0.5, linestyles='--')
    #  ax.clabel(cs, inline=True, fontsize='small')
    #ax.streamplot(x_gd, z_gd,
    #  data[ic]['u'], data[ic]['w'], color='gray', density=2, linewidth=lw-0.5,
    #  arrowsize=ars)
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
      ax.set_xlabel(r'$x (m)$')

      if isub == 0:
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
ticks = np.arange(levels_bdg[0], levels_bdg[-1]+0.1, 0.5)
cbar = plt.colorbar(cax, ticks = ticks, cax=cbar_ax,
  orientation='horizontal')
cbar.ax.set_ylabel(r'$\frac{\partial \overline{e}}{\partial t}/(\epsilon+\epsilon_c)$',
  rotation=0, ha='right', va='center')

# Save figure
fig.savefig(fn_budget, dpi=400)
