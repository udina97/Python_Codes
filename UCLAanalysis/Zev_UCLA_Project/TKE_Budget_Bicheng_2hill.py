"""
Program Name: TKE_Budget_Bicheng.py
Program purpose: This program computes the TKE budget .

Program Author: Marc Calaf.

Date created: 14 July 2023
Last date modified: 

To Do:
    

"""


import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import xarray as xr


os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import tavg_xy, terrainfollowing_3D, read_out_files, terrainfollowing_2D



#%% Loading data variables:
    
#path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/'
path = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_hill2h_3D/'


NumVar = 49
uscale = 0.4
canopyH = 39


#Amazon Hill Canopy
Nx = 320; Ny = 160; Nz = 290; Nz_SLayer = int(Nz/2)
Lx = 2000; Ly = 1000; Lz = 580
Zi = 580


Dx = Lx/Nx; Dy = Ly/Ny; Dz = Lz/Nz


class dm:
    nx = Nx
    ny = Ny
    nz = Nz
    lx = Lx
    ly = Ly
    lz = Lz
    dx = Dx
    dy = Dy
    dz = Dz
    zi = Zi
    

# var_name = ['u','u2','u2v','u2w','u3','uFcx','utxx','utxy','utxz','uv',
#             'uv2','uvw','uw','uw2','v','v2','v2w','v3','vFcy','vtxy',
#             'vtyy','vtyz','vw','vw2','w','w2','w3','wFcz','wtxz','wtyz',
#             'wtzz','p','p2','pdudx','pdvdy','pdwdz','pu','pv','pw','txx',
#             'txy','txz','tyy','tyz','tzz','Fcx','Fcy','Fcz','dissip']

items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')

#%% Loading the topography data:

# from pymatreader import read_mat
from scipy.io import loadmat

topodata = loadmat(path+'matlab/'+'topo.mat')
dist = np.mean(np.transpose(topodata['Z_gd'],(2,1,0)),axis=1)
    
#%% Loading the 2D .out data

path_in = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/amazon_canopy_hill2h/output/'
path_out = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/UCLAdata/amazon_canopy_hill2/'

variables = ['u','u2','u2v','u2w','u3','uFcx','utxx','utxy','utxz','uv',
            'uv2','uvw','uw','uw2','v','v2','v2w','v3','vFcy','vtxy',
            'vtyy','vtyz','vw','vw2','w','w2','w3','wFcz','wtxz','wtyz',
            'wtzz','p','p2','pdudx','pdvdy','pdwdz','pu','pv','pw','txx',
            'txy','txz','tyy','tyz','tzz','Fcx','Fcy','Fcz','dissip']

nt = 24

data_out = read_out_files(path_in, path_out, variables, Nx, Nz, nt, False)

data_tavg = dict()

for i in range(len(variables)):
    data_tavg[variables[i]] = np.mean(data_out[variables[i]],axis=2)

#%% pcolor test plots

var = 'p'
# var2 = 'avgW'
# tmp = copy.deepcopy(data_tke['p']-(data_tke['uu']+data_tke['vv']+data_tke['ww']))
tmp = copy.deepcopy(data_tavg[var])
# tmp2 = copy.deepcopy(data_tke[var2])
tmp[dist<0] = float('nan')

x_ax = np.arange(0,Nx)*Dx
y_ax = np.arange(0,Ny)*Dy
z_ax = np.arange(0,Nz)*Dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 150 #int(nz/2)
zslice = 50

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,:].T,cmap= 'coolwarm')
# plt1 = axs.streamplot(X,Y,tmp[:,yslice,:].T,tmp2[:,yslice,:].T)
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')
# axs.plot((x_ax),(np.mean(topodata_tke['intf'],axis=1) + h_canopy )/dm.zi,'--k')
# axs.set_ylim(z_ax[0],z_ax[-1])

# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis] * phi_c
  dphidx_c[-1, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

# def get_dphidy(phi, wn):
#   phi_c = np.fft.rfft(phi, axis=-2, norm="ortho")
#   dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
#   dphidy_c[:, -1, :] = 0
#   return np.fft.irfft(dphidy_c, axis=-2, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:, :-1] = (phi[:, 1:]-phi[:, :-1]) / (dm.dz/dm.zi)
  dphidz[:, -1] = dphidz[:, -2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, 0] = 0
  phi_h[:, 1:] = 0.5*(phi_c[:, :-1] + phi_c[:, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :-1] = 0.5*(phi_h[:, :-1]+phi_h[:, 1:])
  phi_c[:, -1] = phi_c[:, -2]
  return phi_c


#%% Calculate the Reynolds Stresses:


# Initialization
terms_ptb = dict()
terms_drv = dict()
terms_bdg = dict.fromkeys(items_bdg, None)

wn_x = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
# wn_y = 2*np.pi*np.fft.rfftfreq(dm.ny, dm.dy/dm.zi)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data_tavg['u'])
v_h = uvpnode2wnode(data_tavg['v'])

var_ptb = ['u2_t', 'uv_t', 'uw_t', 'v2_t', 'vw_t', 'w2_t', 'tke', 'tke_SGS']


terms_ptb['u2_t'] = data_tavg['u2'] - data_tavg['u']**2
terms_ptb['uv_t'] = data_tavg['uv'] - data_tavg['u']*data_tavg['v']
terms_ptb['uw_t'] = data_tavg['uw'] - u_h*data_tavg['w']
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data_tavg['v2'] - data_tavg['v']**2
terms_ptb['vw_t'] = data_tavg['vw'] - v_h*data_tavg['w']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data_tavg['w2'] -data_tavg['w']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']) / 2

#%% Flow Overview plots:
    
z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

#Mean Velocity profiles
fig, axs=plt.subplots(1,3, constrained_layout=True)

axs[0].plot(np.mean(data_tavg['u'],axis=(0)),z_ax,label='u')
axs[1].plot(np.mean(data_tavg['v'],axis=(0)),z_ax,label='v')
axs[2].plot(np.mean(data_tavg['w'],axis=(0)),z_ax,label='w')

axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
plt.legend()
plt.show()

var_Rij = ['Rxx', 'Ryy', 'Rzz', 'Rxy', 'Rxz', 'Ryz']

Rij = dict()
Rij['Rxx'] = data_tavg['u2'] - data_tavg['u']**2
Rij['Ryy'] = data_tavg['v2'] - data_tavg['v']**2
Rij['Rzz'] = data_tavg['w2'] - data_tavg['w']**2
Rij['Rxy'] = data_tavg['uv'] - data_tavg['u']*data_tavg['v']
Rij['Rxz'] = data_tavg['uw'] - u_h*data_tavg['w']
Rij['Ryz'] = data_tavg['vw'] - v_h*data_tavg['w']

for i in range(len(var_Rij)):
    Rij[var_Rij[i]][(dist < 0)] = float("nan")


#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.nanmean(Rij['Rxx'],axis=(0)),z_ax)
axs[0,1].plot(np.nanmean(Rij['Ryy'],axis=(0)),z_ax)
axs[0,2].plot(np.nanmean(Rij['Rzz'],axis=(0)),z_ax)
axs[1,0].plot(np.nanmean(Rij['Rxy'],axis=(0)),z_ax)
axs[1,1].plot(np.nanmean(Rij['Rxz'],axis=(0)),z_ax)
axs[1,2].plot(np.nanmean(Rij['Ryz'],axis=(0)),z_ax)

axs[0,0].set_ylim(0,1);axs[0,1].set_ylim(0,1);axs[0,2].set_ylim(0,1)
axs[1,0].set_ylim(0,1);axs[1,1].set_ylim(0,1);axs[1,2].set_ylim(0,1)

axs[0,0].set_xlabel('Rxx');axs[0,1].set_xlabel('Ryy');axs[0,2].set_xlabel('Rzz')
axs[1,0].set_xlabel('Rxy');axs[1,1].set_xlabel('Rxz');axs[1,2].set_xlabel('Ryz')

plt.show()


#%% Calculate the TKE budget

# Calculate the derivatives (all on uvp-nodes)
terms_drv['dudx'] = get_dphidx(data_tavg['u'], wn_x)
# terms_drv['dudy'] = get_dphidy(data['u'], wn_y)
terms_drv['dudz'] = get_dphidz(data_tavg['u'], dm)
terms_drv['dudz'] = wnode2uvpnode(terms_drv['dudz'])

terms_drv['dvdx'] = get_dphidx(data_tavg['v'], wn_x)
# terms_drv['dvdy'] = get_dphidy(data['v'], wn_y)
terms_drv['dvdz'] = get_dphidz(data_tavg['v'], dm)
terms_drv['dvdz'] = wnode2uvpnode(terms_drv['dvdz'])

terms_drv['dwdx'] = get_dphidx(data_tavg['w'], wn_x)
terms_drv['dwdx'] = wnode2uvpnode(terms_drv['dwdx'])
# terms_drv['dwdy'] = get_dphidy(data['w'], wn_y)
# terms_drv['dwdy'] = wnode2uvpnode(terms_drv['dwdy'])
terms_drv['dwdz'] = get_dphidz(data_tavg['w'], dm)

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')
ue = data_tavg['u']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
# ve = data_tavg['v']*terms_ptb['tke']
# dvedy = get_dphidy(ve, wn_y)
tkez = uvpnode2wnode(terms_ptb['tke'])
we = data_tavg['w']*tkez
dwedz = get_dphidz(we, dm)
terms_bdg['adv_h'] = -duedx #- dvedy
terms_bdg['adv_v'] = -dwedz

#SGS
ue_sgs = data_tavg['u']*terms_ptb['tke_SGS']
due_sgsdx = get_dphidx(ue_sgs, wn_x)
tke_sgsz = uvpnode2wnode(terms_ptb['tke_SGS'])
we_sgs = data_tavg['w']*tke_sgsz
dwe_sgsdz = get_dphidz(we_sgs, dm)
terms_bdg['advSGS_h'] = -due_sgsdx #- dvedy
terms_bdg['advSGS_v'] = -dwe_sgsdz

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['advSGS_h'] + terms_bdg['advSGS_v']

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = wnode2uvpnode(data_tavg['uw'])
vw_c = wnode2uvpnode(data_tavg['vw'])
w_c = wnode2uvpnode(data_tavg['w'])
w2_c = wnode2uvpnode(data_tavg['w2'])

terms_ptb['u3_t'] = data_tavg['u3'] - 3*data_tavg['u']*data_tavg['u2'] + 2*data_tavg['u']**3
terms_ptb['uv2_t'] = data_tavg['uv2'] - 2*data_tavg['v']*data_tavg['uv']\
  + 2*data_tavg['u']*data_tavg['v']**2 - data_tavg['u']*data_tavg['v2']
terms_ptb['uw2_t'] = data_tavg['uw2'] - 2*w_c*uw_c + 2*data_tavg['u']*w_c**2\
  - data_tavg['u']*w2_c
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data_tavg['utxx'] - data_tavg['u']*data_tavg['txx']
terms_ptb['vtxy_t'] = data_tavg['vtxy'] - data_tavg['v']*data_tavg['txy']
terms_ptb['wtxz_t'] = data_tavg['wtxz'] - data_tavg['w']*data_tavg['txz']
terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(0.5*(terms_ptb['utxx_t']+terms_ptb['vtxy_t']               #there was no 0.5 here, why?
  +terms_ptb['wtxz_t']), wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

# terms_ptb['u2v_t'] = data['u2v'] - 2*data['u']*data['uv']\
#   + 2*data['v']*data['u']**2 - data['v']*data['u2']
# terms_ptb['v3_t'] = data['v3'] - 3*data['v']*data['v2'] + 2*data['v']**3
# terms_ptb['vw2_t'] = data['vw2'] - 2*w_c*vw_c + 2*data['v']*w_c**2\
#   - data['v']*w2_c
# ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
# dvedy = get_dphidy(ve, wn_y)
# terms_ptb['utxy_t'] = data['utxy'] - data['u']*data['txy']
# terms_ptb['vtyy_t'] = data['vtyy'] - data['v']*data['tyy']
# terms_ptb['wtyz_t'] = data['wtyz'] - data['w']*data['tyz']
# terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
# dutaudy = get_dphidy(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
#   +terms_ptb['wtyz_t'], wn_y)
# terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = uvpnode2wnode(data_tavg['u2'])
v2_h = uvpnode2wnode(data_tavg['v2'])
terms_ptb['wu2_t'] = data_tavg['u2w'] - 2*u_h*data_tavg['uw'] + 2*data_tavg['w']*u_h**2\
  - data_tavg['w']*u2_h
terms_ptb['wv2_t'] = data_tavg['v2w'] - 2*v_h*data_tavg['vw'] + 2*data_tavg['w']*v_h**2\
  - data_tavg['w']*v2_h
terms_ptb['w3_t'] = data_tavg['w3'] - 3*data_tavg['w']*data_tavg['w2'] + 2*data_tavg['w']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dm)
terms_ptb['utxz_t'] = data_tavg['utxz'] - u_h*data_tavg['txz']
terms_ptb['vtyz_t'] = data_tavg['vtyz'] - v_h*data_tavg['tyz']
terms_ptb['wtzz_t'] = data_tavg['wtzz'] - data_tavg['w']*data_tavg['tzz']
dutaudz = get_dphidz(0.5*(terms_ptb['utxz_t']+terms_ptb['vtyz_t']               #there was no 0.5 here, why?
  +terms_ptb['wtzz_t']), dm)
terms_bdg['uturb_v'] = -dwedz-dutaudz

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
terms_ptb['pu_t'] = data_tavg['pu'] - data_tavg['p']*data_tavg['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

terms_ptb['pv_t'] = data_tavg['pv'] - data_tavg['p']*data_tavg['v']
# dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)

p_h = uvpnode2wnode(data_tavg['p'])
terms_ptb['pw_t'] = data_tavg['pw'] - p_h*data_tavg['w']
dpwdz = get_dphidz(terms_ptb['pw_t'], dm)

terms_bdg['pturb_h'] = -dpudx  #-dpvdy
terms_bdg['pturb_v'] = -dpwdz

terms_bdg['ptrans'] = terms_bdg['pturb_h'] + terms_bdg['pturb_v']

# Calculate the dissipation rate (all on uvp-nodes)
print('Calculating the dissipation term')
terms_drv['S11'] = terms_drv['dudx']
terms_drv['S12'] = 0.5*(terms_drv['dvdx'])  #terms_drv['dudy'] + 
terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
# terms_drv['S22'] = terms_drv['dvdy']
terms_drv['S23'] = 0.5*(terms_drv['dvdz'])  #+ terms_drv['dwdy']
terms_drv['S33'] = terms_drv['dwdz']

#- data_tavg['tyy']*terms_drv['S22']\
    
terms_bdg['dissip'] = data_tavg['dissip']\
  - data_tavg['txx']*terms_drv['S11']\
  - data_tavg['tzz']*terms_drv['S33']\
  - 2*data_tavg['txy']*terms_drv['S12'] - 2*data_tavg['txz']*terms_drv['S13']\
  - 2*data_tavg['tyz']*terms_drv['S23']

terms_bdg['canopy'] = data_tavg['wFcz'] - data_tavg['w']*data_tavg['Fcz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data_tavg['uFcx']-data_tavg['u']*data_tavg['Fcx'])\
  +(data_tavg['vFcy']-data_tavg['v']*data_tavg['Fcy'])
  
terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] #+ terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] #+ terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] #+ terms_ptb['vw_t']*terms_drv['dwdy']
  + data_tavg['txx']*terms_drv['S11'] #+ data['tyy']*terms_drv['S22']
  + 2*data_tavg['txy']*terms_drv['S12'] + data_tavg['txz']*terms_drv['dwdx']
  #+ data['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  + data_tavg['txz']*terms_drv['dudz'] + data_tavg['tyz']*terms_drv['dvdz']
  + data_tavg['tzz']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v']

terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] + terms_bdg['canopy'] + terms_bdg['dissip']\
                    + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
                    + terms_bdg['adv_h'] + terms_bdg['adv_v']

print('*'*80)

# np.save(path_out + 'TKE_terms.npy', terms_bdg) 

var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
           'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']

for i in range(len(var_bdg)):
    terms_bdg[var_bdg[i]][(dist < 0)] = float("nan")
    
#%% Remove topography to create terrain following averages

var_topo = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
            'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']

terms_topo = dict()

for i in range(len(var_topo)):
    terms_topo[var_topo[i]] = terrainfollowing_2D(terms_bdg[var_bdg[i]])


#%% Profiles of TKE Budget

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/UCLA_tke/hill2/'

z_ax = np.arange(0,Nz)*dm.dz/canopyH

plt.figure(figsize=(4,8))
plt.plot(np.nanmean(terms_bdg['res'],axis=(0)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
plt.plot(np.nanmean(terms_bdg['dissip'],axis=(0)),z_ax,c='green',marker='v',markevery=5,label='Diss')
plt.plot(np.nanmean(terms_bdg['canopy'],axis=(0)),z_ax,c='orange',marker='o',markevery=5,label='Can')
plt.plot(np.nanmean(terms_bdg['prod'],axis=(0)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
# plt.plot((np.nanmean(terms_bdg['res'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot((np.nanmean(terms_bdg['adv'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot((np.nanmean(terms_bdg['ttrans'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot((np.nanmean(terms_bdg['ptrans'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot((np.nanmean(terms_bdg['dissip'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot((np.nanmean(terms_bdg['canopy'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot((np.nanmean(terms_bdg['prod'],axis=(0))/np.nanmean(-terms_bdg['totdis'],axis=(0)))[7:],z_ax,c='blue',marker='D',markevery=7,label='Prod')
plt.hlines(canopyH/canopyH,-60,80,linestyle='--',colors='k')
plt.hlines(2*(canopyH/canopyH),-60,80,linestyle='--',colors='k')
plt.vlines(1,0,5,linestyle='--',colors='grey')
plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(0,10)
# plt.xlim(-2,2.5)
plt.title(r'Hill2')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_t \right \rangle$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper left')
plt.show()

# plt.savefig(path_fig+'TKE_Budget_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')

#%% pcolor test plots

tmp = copy.deepcopy(terms_bdg['prod']+terms_bdg['totdis'])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,Nx)*Dx
z_ax = np.arange(0,Nz)*Dz

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,:].T,cmap= 'bwr',vmin=-30,vmax=30)
# axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data_tavg['u'],dist,dm).T, DispFluct_withTopo(data_tavg['w'],dist,dm).T,
#                density = 1,color=[0.7,0.7,0.7])
# axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
axs.set_ylim(z_ax[0],z_ax[-1])
# axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.5])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'pu_avg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%Anisotropy analysis

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy_Clustering2D
from Stats import  ReynoldsStress, DispFluct, ReynoldsStressUVP, ReynoldsStress_2D

anisotropy_analysis = 'true'
anisotropy_compute = 'false'

Rstress = xr.DataArray(np.ones(shape = (Nx,Nz,6),order='F'),\
                        dims=('x','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress_2D(Nx,Nz,data_tavg['u'],data_tavg['v'],data_tavg['w'],data_tavg['u2'],\
                          data_tavg['v2'],data_tavg['w2'],data_tavg['uv'],data_tavg['uw'],data_tavg['vw'])


#Anisotropy analysis:
if (anisotropy_analysis == 'true'):
    
    if (anisotropy_compute == 'true'):
        
        [xB,yB,AnisType_1D] = Anisotropy_Clustering2D(Nx,Nz,Rstress)

        yB_1D = np.ndarray.flatten(yB)
        xB_1D = np.ndarray.flatten(xB)


        Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Nz,3),order='F'),\
                            dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
            
        Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

        os.chdir(path_out)
        Anisotropy_clustering.to_netcdf('Anisotropy_clustering.nc')
        
    else:
            
        Anisotropy_clustering = xr.open_dataarray(path_out + 'Anisotropy_clustering.nc')

        xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
        yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
        AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 

    #------------ End of the IF statement.
#--- End of Anisotropy analysis.

#%% New Colormap:
    
iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]

#Transform the HEX colors to RGB.
from PIL import ImageColor

iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')

for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")

iva_colors_RGB = iva_colors_RGB[:,:]/(256)

#Transform the array of colors to a list of values. 
colors = iva_colors_RGB.tolist()
#----------------------------------------------------

#The next few lines create a new colormap using IVA's colors:
from matplotlib.colors import LinearSegmentedColormap,ListedColormap

inbetween_color_amount = 10

# the 10 is from the original 10 colors, the 4 is for R, G, B, A
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))

# add first one already
newcolvals[0] = colors[0]

for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
    
newcolvals

cmap = ListedColormap(newcolvals, name='from_list', N=None)

#%% Some Pcolor figures for yB 

iva_colors = ["#410d00","#831901","#983e00","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]

yB = np.reshape(yB_1D,(Nx,Nz))
yB[(dist[:,:]<0)] = float('nan')
xB = np.reshape(xB_1D,(Nx,Nz))
xB[(dist[:,:]<0)] = float('nan')
# height = 350
z = np.arange(0,Nz)*Dz
z_on_h = z/(39)

fig, axs = plt.subplots(nrows=1,ncols=1)

# sc = axs.contourf(np.arange(0,Nx)*Dx,z_on_h,np.transpose(yB[:,:]),cmap=cmap,levels=[0,0.33,0.36,0.38,0.9],alpha=0.5)
# p = axs.contour(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(((terms_bdg['prod']) - (terms_bdg['totdis']))[:,yslice,:]),levels=[-100,-5,-1,1,5,100])
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[:,yslice,:]),levels=[0,0.3,0.32,0.34,0.36,0.38,0.4,0.9],colors=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black'])#vmin = 0, vmax = np.sqrt(3)/2)
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(xB[:,yslice,:]),cmap=cmap, levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs.plot(np.arange(0,Nx)*Dx,topodata['iintf']*Dz/canopyH,color='k')
axs.plot(np.arange(0,Nx)*Dx,(topodata['iintf']*Dz+canopyH)/canopyH,color='k')
# axs.clabel(p, p.levels, inline=True, fontsize=10)
sc = axs.pcolormesh(np.arange(0,Nx)*Dx,np.arange(0,Nz)*Dz/canopyH,np.transpose(yB[:,:]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

cbar = plt.colorbar(sc,label='yB')
    
axs.set_ylabel(r'$z/h$',fontsize=15)
axs.set_xlabel(r'$x/z_i$',fontsize=15)

plt.tight_layout()


#%% Scatter plot of yB and TKE term

yB = np.reshape(yB_1D,(Nx,Nz))
yB[(dist[:,:]<0)] = float('nan')

tmpDIS = copy.deepcopy(terms_bdg['totdis']) # copy.deepcopy(np.nanmean(terms_bdg['totdis'],axis=(1))) 
tmpRES = copy.deepcopy((terms_bdg['prod']) + (terms_bdg['totdis'])) # copy.deepcopy(np.nanmean((terms_bdg['prod']) - (terms_bdg['totdis']),axis=(1))) 
tmpTUR = copy.deepcopy((terms_bdg['uturb_h'] + terms_bdg['uturb_v'])) # copy.deepcopy(np.nanmean((terms_bdg['uturb_h'] + terms_bdg['uturb_v']),axis=(1))) 
tmpPRE = copy.deepcopy((terms_bdg['pturb_h'] + terms_bdg['pturb_v'])) # copy.deepcopy(np.nanmean((terms_bdg['pturb_h'] + terms_bdg['pturb_v']),axis=(1))) 
tmpADV = copy.deepcopy((terms_bdg['adv_v'] + terms_bdg['adv_h'])) # copy.deepcopy(np.nanmean((terms_bdg['adv_v'] + terms_bdg['adv_h']),axis=(1)))  

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

terms = [(tmpRES/abs(terms_bdg['totdis']))*100,((terms_bdg['uturb_h'] + terms_bdg['uturb_v'])/abs(terms_bdg['totdis']))*100,\
         ((terms_bdg['pturb_h'] + terms_bdg['pturb_v'])/abs(terms_bdg['totdis']))*100, ((terms_bdg['adv_v'] + terms_bdg['adv_h'])/abs(terms_bdg['totdis']))*100]

n = 3
    
TKE_term = terms[n]
tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
tmp_TKE_term[(dist<0)] = float('nan')


level = 3

fig, axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)
sc = axs.scatter(yB[:,:][(dist[:,:]>39) & (dist[:,:]<level*39)],tmp_TKE_term[:,:][(dist[:,:]>39) & (dist[:,:]<level*39)],c=yB[:,:][(dist[:,:]>39) & (dist[:,:]<level*39)],cmap=cmap, vmin=0,vmax=np.sqrt(3)/2)
axs.set_xlabel('yB')
axs.set_ylabel('TKE term')
axs.text(0.65, 0.95, f'r = {round(np.corrcoef(yB[:,:][(dist[:,:]>39) & (dist[:,:]<level*39)].flatten(),tmp_TKE_term[:,:][(dist[:,:]>39) & (dist[:,:]<level*39)].flatten())[0,1],2)}',\
        transform=axs.transAxes, fontsize=14,verticalalignment='top')
cbar = plt.colorbar(sc,label='yB')
axs.set_title(f'{labels[n]} - h to {level}h')

#%% Pcolor plots of TKE terms with contours of yB - yslices

tmpDIS = copy.deepcopy(terms_bdg['totdis'])
tmpRES = copy.deepcopy((terms_bdg['prod']) + (terms_bdg['totdis']))
tmpTUR = copy.deepcopy((terms_bdg['uturb_h'] + terms_bdg['uturb_v']))
tmpPRE = copy.deepcopy((terms_bdg['pturb_h'] + terms_bdg['pturb_v']))
tmpADV = copy.deepcopy((terms_bdg['adv_v'] + terms_bdg['adv_h']))

# tmpDIS[(abs(terms_bdg['totdis'])<1)] = float('nan')
tmpRES[(abs((terms_bdg['prod']) + (terms_bdg['totdis']))<1)] = float('nan')
tmpTUR[(abs((terms_bdg['prod']) + (terms_bdg['totdis']))<1)] = float('nan')
tmpPRE[(abs((terms_bdg['prod']) + (terms_bdg['totdis']))<1)] = float('nan')
tmpADV[(abs((terms_bdg['prod']) + (terms_bdg['totdis']))<1)] = float('nan')

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

terms = [((tmpRES)/abs(tmpDIS)),tmpTUR/abs(tmpDIS),\
         tmpPRE/abs(tmpDIS),(tmpADV)/abs(tmpDIS)]
    
levels=[-100,-25,-1,1,25,100]
colors=['blue','green','white','yellow','red']


fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

for i in range(0,len(axs)):
    for j in range(0,len(axs[0])):
        if i == 0:
            n = j
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            tmp_TKE_term[(dist<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*Dx,np.arange(0,Nz)*Dz/canopyH,(tmp_TKE_term[:,:]*100).T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
            sc = axs[i,j].contour(np.arange(0,Nx)*Dx,np.arange(0,Nz)*Dz/canopyH,np.transpose(yB[:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
            axs[i,j].plot(np.arange(0,Nx)*Dx,topodata['iintf']*Dz/canopyH,color='k')
            axs[i,j].plot(np.arange(0,Nx)*Dx,(topodata['iintf']*Dz+canopyH)/canopyH,color='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
        else:
            n = j+2
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            tmp_TKE_term[(dist<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*Dx,np.arange(0,Nz)*Dz/canopyH,(tmp_TKE_term[:,:]*100).T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
            sc = axs[i,j].contour(np.arange(0,Nx)*Dx,np.arange(0,Nz)*Dz/canopyH,np.transpose(yB[:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
            axs[i,j].plot(np.arange(0,Nx)*Dx,topodata['iintf']*Dz/canopyH,color='k')
            axs[i,j].plot(np.arange(0,Nx)*Dx,(topodata['iintf']*Dz+canopyH)/canopyH,color='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
    
axs[0,0].set_ylabel(r'$z/h$',fontsize=15),axs[1,0].set_ylabel(r'$z/h$',fontsize=15)
axs[1,0].set_xlabel(r'$x/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$x/z_i$',fontsize=15)


#%%Computing the advection index from Chamecki et al 2023 an plotting vertical profile

def ChameckiIndex(Nz_SLayer,coord,dist,Adv,TotDis):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    AdvTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    TotDisTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    Ia = np.zeros((Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        
        loc = coord[i]
        
        AdvTwr[i,:] = Adv[loc,int(np.where(dist[loc,:]>0)[0][0]):int(np.where(dist[loc,:]>0)[0][0])+Nz_SLayer]
        TotDisTwr[i,:] = TotDis[loc,int(np.where(dist[loc,:]>0)[0][0]):int(np.where(dist[loc,:]>0)[0][0])+Nz_SLayer]
        
    Ia = np.mean(np.abs(AdvTwr),axis=0)/np.mean(TotDisTwr,axis=0) 
    
    return Ia

coord = [(77),(78),(79),(80),(81),(82),(83),(237),(238),(239),(240),(241),(242),(243),(244)]
# coord = [(80),(240)]

terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
         terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
         terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

Adv = terms[5]
TotDis = +terms[1]

z = np.arange(0,Nz)*Dz/canopyH

Ia = ChameckiIndex(Nz_SLayer, coord, dist, Adv, TotDis)

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
axs.plot(Ia[0:50],z[0:50])
axs.set_xlabel('Ia')
axs.set_ylabel('z/h')

#%% Graphical Representation of U,W,TKE:

#Set values below the topography equal to NAN:
data_tavg['u'][(dist < 0)] = float("nan")
data_tavg['w'][(dist < 0)] = float("nan")
terms_ptb['tke'][(dist < 0)] = float("nan")

# yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,data_tavg['u'].T,cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,data_tavg['w'].T,cmap='YlGnBu',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,terms_ptb['tke'].T,cmap='YlGnBu',shading='gouraud')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])

#axs[0].hlines(h_canopy/dm.zi,x_ax[0],x_ax[-1],linestyle=':',colors='k')
axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])
axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\overline{u}(x,y_{nz/2},z)/u_*$')

#axs[1].hlines(h_canopy/dm.zi,x_ax[0],x_ax[-1],linestyle=':',colors='k')
axs[1].set_ylabel(r'$z/z_i$');#axs[1].set_xlabel(r'$x/z_i$') 
axs[1].set_title(r'$\overline{w}(x,y_{nz/2},z)/u_*$')

#axs[2].hlines(h_canopy/dm.zi,x_ax[0],x_ax[-1],linestyle=':',colors='k')
axs[2].set_xlabel(r'$x/z_i$'); axs[2].set_ylabel(r'$z/z_i$')
axs[2].set_title(r'$\overline{e}(x,y_{nz/2},z)/u_*^2$')

axs[0].plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[1].plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[2].plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')


plt.show()

# plt.savefig(path_fig+'u_w_tke_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')


#%% Fix for the flat case where we are missing data for the dissipation:
    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
# yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Set values below the topography equal to NAN:

#TKE Production:
Prod = terms_bdg['prod_h'] + terms_bdg['prod_v']

Dissip = terms_bdg['dissip'] + terms_bdg['canopy']
# Dissip = np.zeros((Nz,Ny,Nx),dtype = 'float')

#TKE Dissipation:
# Dissip[h_canopy:-1,:,:] = - Prod[h_canopy:-1,:,:]
# Dissip[0:h_canopy,:,:] = - Prod[h_canopy,:,:]

#TKE Residual:
Res = (Prod + Dissip)
#norm_Res = Res/Dissip

#Set the values of the variables under the topography as NaN
Prod[(dist < 0)] = float("nan")
Dissip[(dist < 0)] = float("nan")
Res[(dist < 0)] = float("nan")


# Dissip[0:h_canopy,:,:] = -Prod[h_canopy,:,:]

fig, axs=plt.subplots(3,1,figsize=(6,6),constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,Prod.T,vmin = 0,vmax = 60, cmap='Reds',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip.T,vmin = -60,vmax = 0,cmap='Blues_r',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,Res.T,vmin = -40,vmax = 40,cmap='bwr',shading='gouraud')


axs[0].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[1].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[2].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')

axs[0].set_ylim([0,1]);axs[1].set_ylim([0,1]);axs[2].set_ylim([0,1])
axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])

axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\mathcal{P}(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')

axs[1].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[1].set_title(r'$\epsilon(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')

axs[2].set_xlabel(r'$x/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[2].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[2].set_title(r'$(\mathcal{P}-\epsilon)\,\,(z_i/u_*^3)$')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])



plt.show()

# plt.savefig(path_fig+'Test_P_D_Res_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')


#%% Graphical Representation of Production, Dissipation, and Residual:

    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
# yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Set values below the topography equal to NAN:

#TKE Production:
Prod = terms_bdg['prod_h'] + terms_bdg['prod_v']

#TKE Dissipation:
Dissip = terms_bdg['dissip'] + terms_bdg['canopy']

#TKE Residual:
Res = (Prod + Dissip)
#norm_Res = Res/Dissip

#Set the values of the variables under the topography as NaN
Prod[(dist < 0)] = float("nan")
Dissip[(dist < 0)] = float("nan")
Res[(dist < 0)] = float("nan")

fig, axs=plt.subplots(3,1,figsize=(6,6),constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,Prod.T,vmin = 0,vmax = 60, cmap='Reds',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip.T,vmin = -60,vmax = 0,cmap='Blues_r',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,Res.T,vmin = -40,vmax = 40,cmap='bwr',shading='gouraud')


axs[0].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[1].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[2].plot(x_ax,(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')

axs[0].set_ylim([0,1]);axs[1].set_ylim([0,1]);axs[2].set_ylim([0,1])
axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])

axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\mathcal{P}(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')

axs[1].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[1].set_title(r'$\epsilon(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')

axs[2].set_xlabel(r'$x/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[2].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[2].set_title(r'$(\mathcal{P}-\epsilon)\,\,(z_i/u_*^3)$')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])



plt.show()

# plt.savefig(path+'Figures/'+'P_D_Res_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')

#%% Graphical Representation of Residual/Dissipation:

    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
# yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Set values below the topography equal to NAN:

#TKE Production:
Prod = terms_bdg['prod_h'] + terms_bdg['prod_v']

#TKE Dissipation:
Dissip = terms_bdg['dissip'] + terms_bdg['canopy']

#Set the values of the variables under the topography as NaN
Prod[(dist < 0)] = float("nan")
Dissip[(dist < 0)] = float("nan")


#TKE Ratio:
Residual = (Prod + Dissip)/(-Dissip)

fig, axs=plt.subplots(2,1,figsize=(8,6),constrained_layout=True)
#plt1 = axs[0].contourf(x_ax,z_ax,Residual[:,yslice,:],levels=[-1.5, -1, -0.5, -0.1, 0.1, 0.5, 1, 1.5],cmap = 'bwr')
plt1 = axs[0].pcolormesh(x_ax,z_ax,Residual.T,vmin = -1.5,vmax = 1.5,cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip.T,vmin = -60,vmax = 20,cmap='Reds',shading='gouraud')



axs[0].plot(x_ax,np.mean(topodata['intf'],axis=1) + h_canopy,'--k')
axs[1].plot(x_ax,np.mean(topodata['intf'],axis=1) + h_canopy,'--k')


axs[0].set_ylim([0,1]);axs[1].set_ylim([0,1])
axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1])

axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\mathcal{P}/\epsilon \,(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')

axs[1].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[1].set_title(r'$\epsilon(x,y_{nz/2},z)\,\,(z_i/u_*^3)$')


fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])




plt.show()

# plt.savefig(path_fig+'P_D_Res_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')

#%% Diepsersive TKE Terms:

def XYavg_withTopo(phi,dist,condition):
    phi[(dist < 0)] = float("nan")
    if condition=='true':
        XYavg_phi = np.nanmean(phi,axis=(0))
    else:
        phi_tmp = terrainfollowing_2D(phi)
        XYavg_phi = np.nanmean(phi_tmp,axis=(0))
    return XYavg_phi

def get_dphi_avgdz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1] = (phi[1:]-phi[:-1]) / (dm.dz/dm.zi)
  dphidz[-1] = dphidz[-2]
  return dphidz

def XYavg_wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:-1] = 0.5*(phi_h[:-1]+phi_h[1:])
  phi_c[-1] = phi_c[-2]
  return phi_c

def DispFluct_withTopo(phi,dist,dm,condition):
    phi[(dist < 0)] = float("nan")
    if condition=='true':
        tmp = np.nanmean(phi,axis=(0))
        DF = np.zeros((dm.nx, dm.nz),dtype='float')
        for k in range(0,Nz):
            DF[:, k] = phi[:, k] - tmp[k]
    else:
        phi_tmp = terrainfollowing_2D(phi)
        tmp = np.nanmean(phi_tmp,axis=(0))
        DF = np.zeros((dm.nx, dm.nz),dtype='float')
        for k in range(0,Nz):
            DF[:, k] = phi_tmp[:, k] - tmp[k]
      
    return DF

    
# Initialization
items_Dispbdg = ('prod','Dprod','Ttransp','Dtransp')

terms_avgdrv = dict()
terms_DispCorr = dict()
terms_Dispbdg = dict.fromkeys(items_Dispbdg, None)

#condition for the averaging, if 'true' then planar average, if 'false' then terrain following average
condition  = 'false'

terms_avgdrv['du_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data_tavg['u'],dist,condition), dm)
terms_avgdrv['du_avgdz'] = XYavg_wnode2uvpnode(terms_avgdrv['du_avgdz'])
terms_avgdrv['dv_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data_tavg['v'],dist,condition), dm)
terms_avgdrv['dv_avgdz'] = XYavg_wnode2uvpnode(terms_avgdrv['dv_avgdz'])
terms_avgdrv['dw_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data_tavg['w'],dist,condition), dm)


#----------------------------------------------------------------------------
# Calculating the space averaged production terms (all on uvp-nodes)
print('Calculating the averaged production term')

terms_Dispbdg['prod'] = -(XYavg_withTopo(terms_ptb['uw_t'],dist,condition)*terms_avgdrv['du_avgdz'] 
                            + XYavg_withTopo(terms_ptb['vw_t'],dist,condition)*terms_avgdrv['dv_avgdz']
                            + XYavg_withTopo(terms_ptb['w2_t'],dist,condition)*terms_avgdrv['dw_avgdz']
                            + XYavg_withTopo(data_tavg['txz'],dist,condition)*terms_avgdrv['du_avgdz']
                            + XYavg_withTopo(data_tavg['tyz'],dist,condition)*terms_avgdrv['dv_avgdz']
                            + XYavg_withTopo(data_tavg['tzz'],dist,condition)*terms_avgdrv['dw_avgdz']
                            )

print('Calculating the Dispersive production term')


terms_DispCorr['prodCorr'] = -( (DispFluct_withTopo(terms_ptb['uw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudz'],dist,dm,condition)) 
                           + (DispFluct_withTopo(terms_ptb['vw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdz'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['w2_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdz'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['u2_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudx'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uv_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdx'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdx'],dist,dm,condition))
                           + (DispFluct_withTopo(data_tavg['txz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudz'],dist,dm,condition)) 
                           + (DispFluct_withTopo(data_tavg['tyz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdz'],dist,dm,condition))
                           + (DispFluct_withTopo(data_tavg['tzz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdz'],dist,dm,condition))
                           + (DispFluct_withTopo(data_tavg['txx'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudx'],dist,dm,condition))
                           + (DispFluct_withTopo(data_tavg['txy'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdx'],dist,dm,condition))
                           + (DispFluct_withTopo(data_tavg['txz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdx'],dist,dm,condition))
                           )

if condition=='true':
    terms_Dispbdg['Dprod'] = XYavg_withTopo(terms_DispCorr['prodCorr'],dist,condition)
else:
    terms_Dispbdg['Dprod'] = np.nanmean(terms_DispCorr['prodCorr'],axis=(0))


#----------------------------------------------------------------------------

# Calculate the space averaged turbulent transport term (vertical, all on uvp-nodes)

print('Calculating the averaged turbulent transport term')

# From the TKE transport term computed earlier compute first the space average and then the vertical gradient:
tmp = - (we + terms_ptb['utxz_t'] + terms_ptb['vtyz_t'] + terms_ptb['wtzz_t'])
terms_Dispbdg['Ttransp'] =  get_dphi_avgdz(XYavg_withTopo(tmp,dist,condition),dm)


print('Calculating the dispersive turbulent transport term')

#terms_DispCorr['transpCorr'] = - DispFluct_withTopo(data['w'],dist,dm)*DispFluct_withTopo(terms_ptb['tke'],dist,dm)
#terms_Dispbdg['Dtransp'] =  get_dphi_avgdz(XYavg_withTopo(terms_DispCorr['transpCorr'],dist),dm)

terms_DispCorr['transpCorr'] = - get_dphidz(DispFluct_withTopo(data_tavg['w'],dist,dm,condition)*(DispFluct_withTopo(terms_ptb['tke'],dist,dm,condition)+DispFluct_withTopo(terms_ptb['tke_SGS'],dist,dm,condition)),dm)

if condition=='true':
    terms_Dispbdg['Dtransp'] =  XYavg_withTopo(terms_DispCorr['transpCorr'],dist,condition)
else:
    terms_Dispbdg['Dtransp'] =  np.nanmean(terms_DispCorr['transpCorr'],axis=(0))

#%% Profile of Dispersive and Non-dispersive production

z_ax = np.arange(0,Nz)*dm.dz/h_canopy#(dm.dz/dm.zi)

plt.figure(figsize=(4,8))
# plt.plot(terms_Dispbdg['prod']/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='grey',marker='o',markevery=5,label='<P>')
# plt.plot(terms_Dispbdg['Dprod']/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='black',marker='s',markevery=5,label='<P">')
plt.plot((terms_Dispbdg['prod'] + terms_Dispbdg['Dprod'])/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='grey',marker='o',markevery=5,label='<P> + <P">')
plt.plot(np.nanmean(terms_topo['prod']/(-terms_topo['totdis']),axis=(0)),z_ax,c='black',marker='s',markevery=5,label='<P>')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.vlines(1,0,5,linestyle='-.',colors='grey')
plt.vlines(0,0,5,linestyle='-.',colors='grey')
plt.ylim(0,5)
plt.xlim(-2,2)
if condition=='true':
    plt.title(r'Hill2 - Planar')
else:
    plt.title(r'Hill2 - Terrain')
plt.xlabel(r'$\mathcal{P}\,\,(z_i/u_*^3)$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper right')
plt.show()

# plt.savefig(path_fig+'Dispersive_Prod_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')

# if condition=='true':
#     plt.savefig(path_fig+'Dispersive_Prod_Planar.png',dpi=300,facecolor='white', edgecolor='white')
# else:
#     plt.savefig(path_fig+'Dispersive_Prod_Terrain.png',dpi=300,facecolor='white', edgecolor='white')

plt.figure(figsize=(4,8))
# plt.plot(terms_Dispbdg['Ttransp']/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='grey',marker='o',markevery=5,label='<T>')
# plt.plot(terms_Dispbdg['Dtransp']/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='black',marker='s',markevery=5,label='<T">')
plt.plot((terms_Dispbdg['Ttransp'] + terms_Dispbdg['Dtransp'])/np.nanmean(-terms_topo['totdis'],axis=(0)),z_ax,c='grey',marker='o',markevery=5,label='<T> + <T">')
plt.plot(np.nanmean(terms_topo['ttrans']/(-terms_topo['totdis']),axis=(0)),z_ax,c='black',marker='s',markevery=5,label='<T>')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.vlines(1,0,5,linestyle='-.',colors='grey')
plt.vlines(0,0,5,linestyle='-.',colors='grey')
plt.ylim(0,5)
plt.xlim(-2,2)
if condition=='true':
    plt.title(r'Hill2 - Planar')
else:
    plt.title(r'Hill2 - Terrain')
plt.xlabel(r'$\mathcal{T}\,\,(z_i/u_*^3)$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper right')
plt.show()

# plt.savefig(path_fig+'Dispersive_Trans_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')

# if condition=='true':
#     plt.savefig(path_fig+'Dispersive_Trans_Planar.png',dpi=300,facecolor='white', edgecolor='white')
# else:
#     plt.savefig(path_fig+'Dispersive_Trans_Terrain.png',dpi=300,facecolor='white', edgecolor='white')

#%% Graphical representation for testing purposes:

#Plot 1: Profiles comparing both approaches for the Production:

z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

fig, axs=plt.subplots(1,2, constrained_layout=True)
axs[0].plot(XYavg_withTopo(terms_bdg['prod_h'],dist),z_ax,label='<P_h>',color='k',marker='o',markerfacecolor="None",markevery=20)
axs[0].plot(XYavg_withTopo(terms_bdg['prod_v'],dist),z_ax,label='<P_v>',color=[0.8,0.8,0.8],marker='s',markerfacecolor="None",markevery=20)
axs[0].plot(terms_Dispbdg['prod'],z_ax,label='<P>',color=[0.6,0.6,0.6],marker='v',markerfacecolor="None",markevery=(10,20))
axs[0].plot(terms_Dispbdg['Dprod'],z_ax,label='<DP>',color=[0.4,0.4,0.4],marker='*',markerfacecolor="None", markevery=(10,20))

axs[1].plot((XYavg_withTopo(terms_bdg['prod_h'],dist) + XYavg_withTopo(terms_bdg['prod_v'],dist)),z_ax,label='P_1',color='k')
axs[1].plot((terms_Dispbdg['prod'] + terms_Dispbdg['Dprod']),z_ax,label='P_2',linestyle='dashed',color=[0.8,0.8,0.8])

axs[0].hlines(h_canopy/dm.zi,-30,70,linestyle='--',colors='k')
axs[0].set_ylim(0,1)
axs[0].set_xlim(-30,70)
axs[0].legend()

axs[1].hlines(h_canopy/dm.zi,-30,70,linestyle='--',colors='k')
axs[0].set_xlim(-30,70)
axs[1].set_ylim(0,1)
axs[1].legend()
plt.show()

#Plot 2: Profiles comparing both approaches for the Turbulent Transport:

z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

fig, axs=plt.subplots(1,2, constrained_layout=True)
axs[0].plot(XYavg_withTopo(terms_bdg['uturb_h'],dist),z_ax,label='<T_h>',color='k')
axs[0].plot(XYavg_withTopo(terms_bdg['uturb_v'],dist),z_ax,label='<T_v>',color=[0.8,0.8,0.8])
axs[0].plot(terms_Dispbdg['Ttransp'],z_ax,label='<T>',linestyle='dashed',color=[0.6,0.6,0.6])
axs[0].plot(terms_Dispbdg['Dtransp'],z_ax,label='<DT>',linestyle ='-.',color=[0.4,0.4,0.4])

axs[1].plot((XYavg_withTopo(terms_bdg['uturb_h'],dist) + XYavg_withTopo(terms_bdg['uturb_v'],dist)),z_ax,label='T_1',color='k')
axs[1].plot((terms_Dispbdg['Ttransp'] + terms_Dispbdg['Dtransp']),z_ax,label='T_2',linestyle='dashed',color=[0.8,0.8,0.8])

axs[0].hlines(h_canopy/dm.zi,-40,80,linestyle='--',colors='k')
axs[0].set_ylim(0,1)
axs[0].legend()

axs[1].hlines(h_canopy/dm.zi,-40,80,linestyle='--',colors='k')
axs[1].set_ylim(0,1)
axs[1].legend()
plt.show()


#Plot 3: 2D slice illustrating the dispersive production

tmp = terms_bdg['prod_h'] + terms_bdg['prod_v']

fig, axs=plt.subplots(2,1, constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax*dm.zi,terms_DispCorr['prodCorr'].T,vmin = -40, vmax= 40, cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax*dm.zi,tmp.T,cmap='YlGnBu',shading='gouraud')


axs[0].plot(x_ax,np.mean(topodata['intf'],axis=1) + h_canopy,'--k')
axs[1].plot(x_ax,np.mean(topodata['intf'],axis=1) + h_canopy,'--k')

axs[0].set_ylim([0,200]);axs[1].set_ylim([0,200])

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])


#%% Graphical representation for presenting purposes:

#Plot 1: Profiles comparing both approaches for the Production:

z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

fig, axs=plt.subplots(1,2, constrained_layout=True)
axs[0].plot(terms_Dispbdg['prod'],z_ax,label=r'$<T_{\mathcal{P}}>$',color=[0,0,0],marker='o',markerfacecolor="None",markevery=(10,20))
axs[0].plot(terms_Dispbdg['Dprod'],z_ax,label=r'$<D_{\mathcal{P}}>$',color=[0.5,0.5,0.5],marker='s',markerfacecolor="None", markevery=(10,20))

axs[1].plot(terms_Dispbdg['Ttransp'],z_ax,label=r'$<T_{\mathcal{T}}>$',color=[0,0,0],marker='o',markerfacecolor="None",markevery=(10,20))
axs[1].plot(terms_Dispbdg['Dtransp'],z_ax,label=r'$<D_{\mathcal{T}}>$',color=[0.5,0.5,0.5],marker='s',markerfacecolor="None", markevery=(10,20))

axs[0].hlines(h_canopy/dm.zi,-30,70,linestyle=':',colors='k')
axs[0].set_ylim(0,1)
axs[0].set_xlim(-30,70)
axs[0].set_xlabel(r'$\mathcal{P}\,\,(z_i/u_*^3)$')
axs[0].set_ylabel(r'$z/z_i$')
axs[0].legend()

axs[1].hlines(h_canopy/dm.zi,-40,40,linestyle=':',colors='k')
axs[1].set_xlim(-40,40)
axs[1].set_ylim(0,1)
axs[1].set_xlabel(r'$\mathcal{T}\,\,(z_i/u_*^3)$')
axs[1].legend()
plt.show()

# plt.savefig(path_fig+'Prod_Transp_profiles.png',dpi=300,facecolor='white', edgecolor='white')


#Plot 2: 2D slice illustrating the dispersive production

fig, axs=plt.subplots(2,1,figsize=(6,6),constrained_layout=True)
plt1 = axs[0].pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['prodCorr'].T,vmin = -10, vmax= 10, cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['transpCorr'].T,vmin = -5, vmax= 5, cmap='bwr',shading='gouraud')

axs[0].plot((x_ax/dm.zi),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')
axs[1].plot((x_ax/dm.zi),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')

axs[0].set_ylim([0,1])
axs[0].set_ylabel(r'$z/z_i$')
axs[0].set_title(r'$D_{\mathcal{P}} (x,y_{ny/2},z)$')
axs[1].set_xlabel(r'$x/z_i$')
axs[1].set_ylim([0,1])
axs[1].set_ylabel(r'$z/z_i$')
axs[1].set_title(r'$D_{\mathcal{T}} (x,y_{ny/2},z)$')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])

# plt.savefig(path_fig+'Prod_Transp_2DSlice.png',dpi=300,facecolor='white', edgecolor='white')


#Plot 3: 2D slice illustrating the dispersive production only 

fig, axs=plt.subplots(1,1,figsize=(6,3),constrained_layout=True)
plt1 = axs.pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['prodCorr'].T,vmin = -10, vmax= 10, cmap='bwr',shading='gouraud')

axs.plot((x_ax/dm.zi),(np.mean(topodata['intf'],axis=1) + h_canopy)/dm.zi,'--k')

axs.set_ylim([0,1])
axs.set_ylabel(r'$z/z_i$')
axs.set_title(r'$D_{\mathcal{P}} (x,y_{ny/2},z)$')
axs.set_xlabel(r'$x/z_i$')

fig.colorbar(plt1,ax=axs)


# plt.savefig(path_fig+'Disp_Prod_2DSlice.png',dpi=300,facecolor='white', edgecolor='white')

#%% Computing vorticity:

order = 1

vort_x = terms_drv['dvdz']
vort_y = terms_drv['dudz'] - terms_drv['dwdx']
vort_z = terms_drv['dvdx']


fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

#y-vorticity vertical slice
plt1 = axs.pcolormesh(x_ax,z_ax,vort_y.T,cmap= 'bwr',vmin=-50,vmax = 50)
axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data_tavg['u'],dist,dm).T, DispFluct_withTopo(data_tavg['w'],dist,dm).T,
               density = 1,color=[0.7,0.7,0.7])
axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs[0].hlines(values_zlh[n]/zi,x_ax[0],x_ax[-1],linestyles='--',color='gray')
axs.set_ylim([0,1])
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
plt.show()

# plt.savefig(path_fig+'Vorticity_Streamlines.png',dpi=300,facecolor='white', edgecolor='white')

#%% Compensated circulation


gamma_star = np.nansum(abs(vort_y),axis=(0,1))/(uscale*dm.zi)

#%%

var_vel = ['u', 'v', 'w']

velocity = dict()
velocity['u'] = data_tavg['u']
velocity['v'] = data_tavg['v']
velocity['w'] = data_tavg['w']
for i in range(len(var_vel)):
    velocity[var_vel[i]][(dist < 0)] = float("nan")

uw_d = np.nanmean(velocity['u']*velocity['w'],axis=(0)) - np.nanmean(velocity['u'],axis=(0))*np.nanmean(velocity['w'],axis=(0))

u_d = np.zeros((Nx,Nz),'d',order='F')
v_d = np.zeros((Nx,Nz),'d',order='F')
w_d = np.zeros((Nx,Nz),'d',order='F')

for k in range(0,Nz):
    u_d[:,k] = velocity['u'][:,k] - np.nanmean(velocity['u'],axis=(0))[k]
    v_d[:,k] = velocity['v'][:,k] - np.nanmean(velocity['v'],axis=(0))[k]
    w_d[:,k] = velocity['w'][:,k] - np.nanmean(velocity['w'],axis=(0))[k]
    
    
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

uw_t = terms_ptb['uw_t']
uw_t[dist<0] = float('nan')

# fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

# plt1 = axs.pcolormesh(x_ax,z_ax,(u_d*w_d).T,cmap= 'bwr')
# axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
# # axs.set_title('Vorticity - y & Dispersive Streamlines')
# # axs[0].hlines(values_zlh[n]/zi,x_ax[0],x_ax[-1],linestyles='--',color='gray')
# # axs.set_ylim([0,1])
# axs.set_ylabel(r'$z/z_i$')
# axs.set_xlabel(r'$x/z_i$')
# fig.colorbar(plt1,ax=axs)
# plt.show()

# z_ax = np.arange(0,Nz)*dm.dz/h_canopy#(dm.dz/dm.zi)

plt.figure(figsize=(4,8))
plt.plot(np.nanmean(-uw_t,axis=(0)),z_ax,c='pink',marker='o',markevery=5,label='<uw_t>')
plt.plot(uw_d,z_ax,c='red',marker='o',markevery=5,label='uw_d')
# plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
# plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
# plt.vlines(1,0,5,linestyle='--',colors='grey')
# plt.vlines(-1,0,5,linestyle='--',colors='grey')
# plt.ylim(0,5)
# plt.xlim(-2,2.5)
plt.title(r'Hill2')
# plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_t \right \rangle$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper left')
plt.show()




















































