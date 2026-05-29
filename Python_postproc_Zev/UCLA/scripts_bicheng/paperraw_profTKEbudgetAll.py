#!/usr/bin/env python3

"""
Program: paperraw_profTKEbudgetAll
Plot the plane-averaged TKE budget (normalized by dissipation)
"""

### Histories:
### 07/21/2020 -- Bicheng Chen (chabby@ucla.edu) -- First released



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
#lab_case = ('Flat', 'Idealized', 'Real')
lab_case = ('Flat', 'Idealized', 'Real')

# File format
fmt_lespath = ('/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}',
  '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}', 
  '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/{case:s}')
fmt_real = './data/prof_budget_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fn_fig = './figure/paperraw/paperraw_profTKEbudgetAll.png'
flag_out = False
fmt_out = './data/meanProf_{case:s}_tt{tts:08d}-{tte:08d}.npz'

# Topography parameters
zpad = (5, 0)
amp = (0, 25)
wl = 1000
lh = wl/4

# Canopy parameters
hc = 39
ustar = 0.4

# Simulation parameters
dxy = (6.25, 6.25, 3000/376) 
dz = (2, 2, 2)

# Data range
ts = 14400
te = 32400
#te = 18000
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uw', 'uv', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw', 'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')
#items_bdg = ('prod', 'canopy', 'dissip', 'adv', 'uturb', 'pturb', 'sum')
#labs_bdg = (r'$P$', r'$\epsilon_c$', r'$\epsilon$', r'$A_e$', r'$T_e$',
#  r'$\Pi_e$', 'sum')
items_bdg = ('prod', 'canopy', 'dissip', 'uturb', 'pturb', 'adv')
labs_bdg = (r'$P$', r'$\epsilon_c$', r'$\epsilon$', r'$T_e$',
  r'$\Pi_e$', r'$A_e$')

# Figure
sz = dict(left=0.07, right=0.88, bottom=0.13, top=0.94,
  wspace=0.1, hspace=0.2)
lw = 1.5
ls = ('-', '--', '-.')
ms = ('D', 'o', 'v', 's', '^')
xspacing = 1
yspacing = 0.5
xlim = (-1.1, 2.1)
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

def get_vprof(term, Z_gd, amp):
  if amp==0:
    prof = term.mean(axis=-1)
    return prof
  else:
    Z_prof = np.squeeze(Z_gd[:, 0])
    prof = term[:, 0] / term.shape[-1]
    for ix in range(1, term.shape[-1]):
      prof += np.interp(Z_prof, Z_gd[:, ix], term[:, ix]) / term.shape[-1]
    return prof



## Initialization
nca = len(case)
data = [{item: None for item in items} for ic in range(nca)]
tke = [None for ic in range(nca)]
res = [None for ic in range(nca)]
ratio = [None for ic in range(nca)]
term_bdg = [dict.fromkeys(items_bdg, None) for ic in range(nca)]
dissip_ttl = [None for ic in range(nca)]



## Read data and calculate the temporal mean
param = [None for ic in range(nca)]
dm = [None for ic in range(nca)]
for ic in range(nca):
  lespath = fmt_lespath[ic].format(case=case[ic])
  param[ic] = lp.lesParam.lesClass.param(lespath)
  dm[ic] = param[ic].domain

some = [dict() for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    for item in items:
      data[ic][item], time = lp.io.io_averFile.loadLES_averFile(param[ic],
        qtype=item, tss=ts, tes=te)
      if item in ('u2', 'v2', 'w2'):
        some[ic][item] = data[ic][item].copy()
      data[ic][item] = np.mean(data[ic][item], axis=0)
  else:
    fn_real = fmt_real.format(case=case[ic], tts=ts*10+100, tte=te*10)
    dset = np.load(fn_real, allow_pickle=True)
    Z_prof = dset['Z_prof']
    term_bdg[ic] = dset['terms'].item()



## Calculate the de/dt
n_half = 12
dedt = [None for ic in range(nca)]
dkedt = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    u2s = np.mean(some[ic]['u2'][:n_half, :, :], axis=0)
    v2s = np.mean(some[ic]['v2'][:n_half, :, :], axis=0)
    w2s = np.mean(some[ic]['w2'][:n_half, :, :], axis=0)
    u2e = np.mean(some[ic]['u2'][n_half:, :, :], axis=0)
    v2e = np.mean(some[ic]['v2'][n_half:, :, :], axis=0)
    w2e = np.mean(some[ic]['w2'][n_half:, :, :], axis=0)
    kes = 0.5 * (u2s + v2s + w2s)
    kee = 0.5 * (u2e + v2e + w2e)
    dedt[ic] = np.sum((kee-kes) / (3600 / (hc/ustar))) * dz[ic] / dm[ic].lz
    print("dedt =", dedt[ic] / dm[ic].nx)
    print("ke=", np.sum(kee) * dz[ic] / dm[ic].lz)



## Get the coordinates
z_gd = [None for ic in range(nca)]
x_gd = [None for ic in range(nca)]
Z_gd = [None for ic in range(nca)]
z_tpg = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    z_gd[ic], x_gd[ic] = lp.domain.dmFun.xzcoord(dm[ic], ztype='staggered')
    z_tpg[ic] = get_topo(x_gd[ic], wl, amp[ic])
    z_gd[ic] -= zpad[ic]
    Z_gd[ic] = z_gd[ic]-z_tpg[ic]



wn_x = [None for ic in range(nca)]
wn_y = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    wn_x[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].nx, dm[ic].dx/dm[ic].zi)
    wn_y[ic] = 2*np.pi*np.fft.rfftfreq(dm[ic].ny, dm[ic].dy/dm[ic].zi)

for ic in range(nca):
  if lab_case[ic] != 'Real':
    # Calculate the production
    data[ic]['dwdx'] = get_dphidx(data[ic]['w'], wn_x[ic])
    #data[ic]['dwdx'] = gaussian_filter(data[ic]['dwdx'], sigma=(1, 1), mode='wrap')
    
    data[ic]['dudx'] = get_dphidx(data[ic]['u'], wn_x[ic])
    #data[ic]['dudx'][1:, :] = 0.5 * (data[ic]['dudx'][:-1, :]
    #  +data[ic]['dudx'][1:, :])
    #data[ic]['dudx'][0, :] = 0
    #data[ic]['dudx'] = gaussian_filter(data[ic]['dudx'], sigma=(1, 1), mode='wrap')
    
    data[ic]['dudz'] = np.zeros(data[ic]['u'].shape) 
    data[ic]['dudz'][1:, :] =\
      (data[ic]['u'][1:, :]-data[ic]['u'][:-1, :]) / (dm[ic].dz/dm[ic].zi)
    #data[ic]['dudz'] = gaussian_filter(data[ic]['dudz'], sigma=(1, 1), mode='wrap')

    data[ic]['dvdx'] = get_dphidx(data[ic]['v'], wn_x[ic])
    #data[ic]['dvdx'][1:, :] = 0.5 * (data[ic]['dvdx'][:-1, :]
    #  +data[ic]['dvdx'][1:, :])
    #data[ic]['dvdx'][0, :] = 0
    #data[ic]['dvdx'] = gaussian_filter(data[ic]['dvdx'], sigma=(1, 1), mode='wrap')
    

    data[ic]['dvdz'] = np.zeros(data[ic]['v'].shape) 
    data[ic]['dvdz'][1:, :] =\
      (data[ic]['v'][1:, :]-data[ic]['v'][:-1, :]) / (dm[ic].dz/dm[ic].zi)
    #data[ic]['dvdz'] = gaussian_filter(data[ic]['dvdz'], sigma=(1, 1), mode='wrap')

    data[ic]['dwdz'] = np.zeros(data[ic]['w'].shape) 
    data[ic]['dwdz'][1:-1, :] =\
      (data[ic]['w'][2:, :]-data[ic]['w'][:-2, :]) / (2*dm[ic].dz/dm[ic].zi)
    #data[ic]['dwdz'] = gaussian_filter(data[ic]['dwdz'], sigma=(1, 1), mode='wrap')
    
    data[ic]['uw_t'] = data[ic]['uw'] - data[ic]['u']*data[ic]['w']
    #data[ic]['uw_t'] = gaussian_filter(data[ic]['uw_t'], sigma=(1, 1), mode='wrap')
    data[ic]['vw_t'] = data[ic]['vw'] - data[ic]['v']*data[ic]['w']
    #data[ic]['vw_t'] = gaussian_filter(data[ic]['vw_t'], sigma=(1, 1), mode='wrap')
    data[ic]['uv_t'] = data[ic]['uv'] - data[ic]['u']*data[ic]['v']
    #data[ic]['uv_t'] = gaussian_filter(data[ic]['uv_t'], sigma=(1, 1), mode='wrap')
    data[ic]['uu_t'] = data[ic]['u2'] - data[ic]['u']**2
    #data[ic]['uu_t'] = gaussian_filter(data[ic]['uu_t'], sigma=(1, 1), mode='wrap')
    data[ic]['vv_t'] = data[ic]['v2'] - data[ic]['v']**2
    #data[ic]['vv_t'] = gaussian_filter(data[ic]['vv_t'], sigma=(1, 1), mode='wrap')
    data[ic]['ww_t'] = data[ic]['w2'] - data[ic]['w']**2
    #data[ic]['ww_t'] = gaussian_filter(data[ic]['ww_t'], sigma=(1, 1), mode='wrap')
    
    # TKE
    tke[ic] = (data[ic]['uu_t'] + data[ic]['vv_t'] + data[ic]['ww_t']) / 2

    # Production
    term_bdg[ic]['prod'] = -data[ic]['uu_t']*data[ic]['dudx']\
      - data[ic]['ww_t']*data[ic]['dwdz']\
      - data[ic]['uw_t']*(data[ic]['dudz']+data[ic]['dwdx'])\
      - data[ic]['vw_t']*data[ic]['dvdz']\
      - data[ic]['uv_t']*data[ic]['dvdx']
    #term_bdg[ic]['prod'] = gaussian_filter(term_bdg[ic]['prod'], sigma=(0, 2), mode='wrap')

    # Advection term
    ue = data[ic]['u']*tke[ic]
    duedx = get_dphidx(ue, wn_x[ic])
    tkez = np.zeros(tke[ic].shape)
    tkez[0, :] = 0
    tkez[1:, :] = 0.5*(tke[ic][:-1, :] + tke[ic][1:, :])
    we = data[ic]['w']*tkez
    dwedz = get_dphidz(we, dm[ic])
    term_bdg[ic]['adv'] = -duedx-dwedz

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
    duedx = get_dphidx(ue, wn_x[ic])
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
    term_bdg[ic]['uturb'] = -duedx-dwedz

    utxx = data[ic]['utxx'] - data[ic]['u']*data[ic]['txx']
    vtxy = data[ic]['vtxy'] - data[ic]['v']*data[ic]['txy']
    wtxz = data[ic]['wtxz'] - data[ic]['w']*data[ic]['txz']
    wtxz_uv = wnode2uvpnode(wtxz)
    dutaudx = get_dphidx(utxx+vtxy+wtxz_uv, wn_x[ic])
    utxz = data[ic]['utxz'] - u_w*data[ic]['txz']
    vtyz = data[ic]['vtyz'] - v_w*data[ic]['tyz']
    wtzz = data[ic]['wtzz'] - data[ic]['w']*data[ic]['tzz']
    dutaudz = get_dphidz(utxz+vtyz+wtzz, dm[ic])
    term_bdg[ic]['uturb'] -= dutaudx+dutaudz

    # Pressure transport term
    pu = data[ic]['pu'] - data[ic]['p']*data[ic]['u']
    dpudx = get_dphidx(pu, wn_x[ic])
    p_w = uvpnode2wnode(data[ic]['p'])
    pw = data[ic]['pw'] - p_w*data[ic]['w']
    dpwdz = get_dphidz(pw, dm[ic])
    term_bdg[ic]['pturb'] = -dpudx-dpwdz

    # Dissipation
    data[ic]['dudz'] = get_dphidz(data[ic]['u'], dm[ic])
    data[ic]['dudz'] = wnode2uvpnode(data[ic]['dudz'])

    data[ic]['dvdx'] = get_dphidx(data[ic]['v'], wn_x[ic])
    data[ic]['dvdz'] = get_dphidz(data[ic]['v'], dm[ic])
    data[ic]['dvdz'] = wnode2uvpnode(data[ic]['dvdz'])
  
    data[ic]['dwdz'] = get_dphidz(data[ic]['w'], dm[ic])

    s11 = data[ic]['dudx']
    s12 = 0.5*data[ic]['dvdx']
    s13 = 0.5*(data[ic]['dudz'] + data[ic]['dwdx'])
    s22 = 0
    s23 = 0.5*data[ic]['dvdz']
    s33 = data[ic]['dwdz']

    term_bdg[ic]['canopy'] = (data[ic]['wFcz']-data[ic]['w']*data[ic]['Fcz'])
    term_bdg[ic]['canopy'] = wnode2uvpnode(term_bdg[ic]['canopy'])
    term_bdg[ic]['canopy'] += (data[ic]['uFcx']-data[ic]['u']*data[ic]['Fcx'])\
      +(data[ic]['vFcy']-data[ic]['v']*data[ic]['Fcy'])
    term_bdg[ic]['dissip'] = data[ic]['dissip']\
      - data[ic]['txx']*s11 - data[ic]['tyy']*s22 - data[ic]['tzz']*s33\
      - 2*data[ic]['txy']*s12 - 2*data[ic]['txz']*s13 - 2*data[ic]['tyz']*s23
    dissip_ttl[ic] = data[ic]['dissip'] + term_bdg[ic]['canopy']

    # Sum of above terms
    term_bdg[ic]['sum'] = term_bdg[ic]['prod'] + term_bdg[ic]['adv']\
      + term_bdg[ic]['uturb'] + term_bdg[ic]['pturb']\
      + term_bdg[ic]['canopy'] + term_bdg[ic]['dissip'] 



## Smooth the data
for ic in range(nca):
  if lab_case[ic] != 'Real':
    dissip_ttl[ic] = gaussian_filter(dissip_ttl[ic], sigma=(1, 2), mode='wrap')
    for item in items_bdg:
      term_bdg[ic][item] = gaussian_filter(term_bdg[ic][item],
        sigma=(1, 2), mode='wrap')



## Mask the data
mask = [None for ic in range(nca)]
for ic in range(nca):
  if lab_case[ic] != 'Real':
    mask[ic] = Z_gd[ic]<=0
    for item in items_bdg:
      term_bdg[ic][item] = np.ma.array(term_bdg[ic][item], mask=mask[ic])



## Compare the production and dissipation
for ic in range(nca):
  print(lab_case[ic], '*'*80)
  #if lab_case[ic] != 'Real':
  #  print(term_bdg[ic]['prod'][mask[ic]])
  #  term_bdg[ic]['prod'][mask[ic]] = 0
  #  term_bdg[ic]['dissip'][mask[ic]] = 0
  #  term_bdg[ic]['canopy'][mask[ic]] = 0
  sum_prod = np.sum(term_bdg[ic]['prod']) *hc/dm[ic].lz * dz[ic]/hc
  sum_dissip = -np.sum(term_bdg[ic]['dissip'] + term_bdg[ic]['canopy']) *hc/dm[ic].lz * dz[ic]/hc
  if lab_case[ic] != 'Real':
    print('total prod', sum_prod/dm[ic].nx)
    print('total dissip', sum_dissip/dm[ic].nx)
    print('dissip/prod', sum_dissip/sum_prod)
  else:
    print('total prod', sum_prod)
    print('total dissip', sum_dissip)
    print('dissip/prod', sum_dissip/sum_prod)



## Get vertical profiles
for ic in range(nca):
  if lab_case[ic] != 'Real':
    dissip_ttl[ic] = get_vprof(dissip_ttl[ic], Z_gd[ic], amp[ic])
    for item in items_bdg:
      print(item)
      term_bdg[ic][item] = get_vprof(term_bdg[ic][item], Z_gd[ic], amp[ic])



## Get shear length scale Ls=U(hc)/(dudz)_hc
for ic in range(nca):
  print("Shear lenght scale:")
  if lab_case[ic] == 'flat':
    data[ic]['u'] = np.mean(data[ic]['u'], axis=1)
    data[ic]['dudz'] = np.mean(data[ic]['dudz'], axis=1)
    Ls = data[ic]['u'] / data[ic]['dudz'] * dm[ic].lz
    for ele, z in zip(Ls, Z_gd[ic][:, 0]):
      print(ele, z)



## Normalization by total dissipation
for ic in range(nca):
  if lab_case[ic] == 'Real':
    dissip_ttl[ic] = term_bdg[ic]['dissip'] + term_bdg[ic]['canopy']

  print(lab_case[ic], '#'*80)
  sum_prod = np.sum(term_bdg[ic]['prod']) *hc/dm[ic].lz * dz[ic]/hc
  sum_dissip = -np.sum(dissip_ttl[ic]) *hc/dm[ic].lz * dz[ic]/hc
  print('total prod', sum_prod)
  print('total dissip', sum_dissip)
  print('dissip/prod', sum_dissip/sum_prod)

  if flag_out:
    fn_out = fmt_out.format(case=case[ic], tts=ts*10+100, tte=te*10)
    if lab_case[ic] != 'Real':
      np.savez(fn_out, terms=term_bdg[ic], Z_prof=z_gd[ic][:, 0])
    else:
      np.savez(fn_out, terms=term_bdg[ic], Z_prof=Z_prof)

  for item in items_bdg:
    term_bdg[ic][item] /= -dissip_ttl[ic]



## Plot the results
# Initlize the figure
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

ifig = 0
fig = plt.figure(ifig, figsize=(8, 4))
gs = gridspec.GridSpec(1, 3)
gs.update(**sz)

# Plot the budget terms
for ic in range(nca):

  # Plot the prod
  ax = fig.add_subplot(gs[ic])
  if lab_case[ic] != 'Real':
    for it, item in enumerate(items_bdg):
      ax.plot(term_bdg[ic][item], Z_gd[ic][:, 0]/hc, lw=lw,
        ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
        markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
      #if item=='canopy':
      #  print('canopy')
      #  print(np.argwhere(term_bdg[ic][item]>-0.1))
      #  print(Z_gd[ic][np.argwhere(term_bdg[ic][item]>-0.1)[0][0]-1, 0]/hc)
    fn_npz = "./data/forpaper/fig6_profTKEbudget_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_gd[ic][:, 0]/hc, budget=term_bdg[ic])
  else:
    for it, item in enumerate(items_bdg):
      ax.plot(term_bdg[ic][item], Z_prof/hc, lw=lw,
        ls=ls[np.mod(it, len(ls))], marker=ms[np.mod(it, len(ms))],
        markevery=(np.mod(it, 5), 5), label=labs_bdg[it])
      #if item=='canopy':
      #  print('canopy')
      #  print(np.argwhere(term_bdg[ic][item]>-0.1))
      #  print(Z_prof[np.argwhere(term_bdg[ic][item]>-0.1)[0][0]-1]/hc)
    fn_npz = "./data/forpaper/fig6_profTKEbudget_{case:s}".format(case=lab_case[ic])
    np.savez(fn_npz, z=Z_prof/hc, budget=term_bdg[ic])

  ax.axhline(y=1, lw=lw, ls='--', color='k')
  ax.axhline(y=2, lw=lw, ls=':', color='k')
  ax.axvline(x=1, lw=lw, ls='-.', color='gray')

  # Axis property
  ax.xaxis.set_major_locator(xmaxLocator)
  ax.yaxis.set_major_locator(ymaxLocator)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  # Axis label
  ax.set_xlabel(r'$\langle\frac{\partial \overline{e}}{\partial t}\rangle/\langle\epsilon_t\rangle$')
  if ic==0:
    ax.set_ylabel('$z/h_c$')
  else:
    ax.set_yticklabels([])
  if ic==nca-1:
    #ax.legend(loc='upper right', fontsize='medium')
    ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))
  ax.set_title(lab_case[ic], fontsize='large')

  ax.text(0, 1.01, '({seq})'.format(seq=chr(97+ic)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')
  
# Save figure
#fig.savefig(fn_fig, dpi=600)
