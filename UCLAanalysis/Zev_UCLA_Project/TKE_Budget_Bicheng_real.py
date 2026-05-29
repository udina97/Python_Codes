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


os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import tavg_xy, terrainfollowing_3D



#%% Loading data variables:
    
#path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/'
path = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_real_3D/'


NumVar = 49
uscale = 0.4
h_canopy = 39


#Amazon Hill Canopy
Nx = 376; Ny = 376; Nz = 270 
Lx = 3000; Ly = 3000; Lz = 540
Zi = 540


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
    

var_name = ['u','u2','u2v','u2w','u3','uFcx','utxx','utxy','utxz','uv',
            'uv2','uvw','uw','uw2','v','v2','v2w','v3','vFcy','vtxy',
            'vtyy','vtyz','vw','vw2','w','w2','w3','wFcz','wtxz','wtyz',
            'wtzz','p','p2','pdudx','pdvdy','pdwdz','pu','pv','pw','txx',
            'txy','txz','tyy','tyz','tzz','Fcx','Fcy','Fcz','dissip']

items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')

data = dict() #we create an empty dictionary.

for i in range(0,NumVar):
    file_name = path + var_name[i] + '.npy'
    
    print(f'Reading variable = {var_name[i]}')
    
    #Each Variable is assigned as one element of the dictionary.
    data[var_name[i]] = np.load(file_name, allow_pickle=False)

#%% Loading the topography data:

# from pymatreader import read_mat
from scipy.io import loadmat

topodata = loadmat(path+'matlab/'+'topo.mat')
dist = topodata['Z_gd']

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = complex(0, 1) * wn[np.newaxis, np.newaxis, :] * phi_c
  dphidx_c[:, :, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-2, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=-2, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1, :] = (phi[1:, :]-phi[:-1, :]) / (dm.dz/dm.zi)
  dphidz[-1, :] = dphidz[-2, :]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[0, :, :] = 0
  phi_h[1:, :, :] = 0.5*(phi_c[:-1, :, :] + phi_c[1:, :, :])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:-1, :, :] = 0.5*(phi_h[:-1, :, :]+phi_h[1:, :, :])
  phi_c[-1, :, :] = phi_c[-2, :, :]
  return phi_c


#%% Calculate the Reynolds Stresses:


# Initialization
terms_ptb = dict()
terms_drv = dict()
terms_bdg = dict.fromkeys(items_bdg, None)

wn_x = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
wn_y = 2*np.pi*np.fft.rfftfreq(dm.ny, dm.dy/dm.zi)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data['u'])
v_h = uvpnode2wnode(data['v'])


terms_ptb['u2_t'] = data['u2'] - data['u']**2
terms_ptb['uv_t'] = data['uv'] - data['u']*data['v']
terms_ptb['uw_t'] = data['uw'] - u_h*data['w']
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data['v2'] - data['v']**2
terms_ptb['vw_t'] = data['vw'] - v_h*data['w']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data['w2'] -data['w']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data['txx']+data['tyy']+data['tzz']) / 2

#%% Flow Overview plots:
    
z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

#Mean Velocity profiles
fig, axs=plt.subplots(1,3, constrained_layout=True)

axs[0].plot(np.mean(data['u'],axis=(1,2)),z_ax,label='u')
axs[1].plot(np.mean(data['v'],axis=(1,2)),z_ax,label='v')
axs[2].plot(np.mean(data['w'],axis=(1,2)),z_ax,label='w')

axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
plt.legend()
plt.show()

var_Rij = ['Rxx', 'Ryy', 'Rzz', 'Rxy', 'Rxz', 'Ryz']


Rij = dict()
Rij['Rxx'] = data['u2'] - data['u']**2
Rij['Ryy'] = data['v2'] - data['v']**2
Rij['Rzz'] = data['w2'] - data['w']**2
Rij['Rxy'] = data['uv'] - data['u']*data['v']
Rij['Rxz'] = data['uw'] - u_h*data['w']
Rij['Ryz'] = data['vw'] - v_h*data['w']

for i in range(len(var_Rij)):
    Rij[var_Rij[i]][(dist < 0)] = float("nan")

#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.nanmean(Rij['Rxx'],axis=(1,2)),z_ax)
axs[0,1].plot(np.nanmean(Rij['Ryy'],axis=(1,2)),z_ax)
axs[0,2].plot(np.nanmean(Rij['Rzz'],axis=(1,2)),z_ax)
axs[1,0].plot(np.nanmean(Rij['Rxy'],axis=(1,2)),z_ax)
axs[1,1].plot(np.nanmean(Rij['Rxz'],axis=(1,2)),z_ax)
axs[1,2].plot(np.nanmean(Rij['Ryz'],axis=(1,2)),z_ax)

axs[0,0].set_ylim(0,1);axs[0,1].set_ylim(0,1);axs[0,2].set_ylim(0,1)
axs[1,0].set_ylim(0,1);axs[1,1].set_ylim(0,1);axs[1,2].set_ylim(0,1)

axs[0,0].set_xlabel('Rxx');axs[0,1].set_xlabel('Ryy');axs[0,2].set_xlabel('Rzz')
axs[1,0].set_xlabel('Rxy');axs[1,1].set_xlabel('Rxz');axs[1,2].set_xlabel('Ryz')

plt.show()


#%% Calculate the TKE budget

# Calculate the derivatives (all on uvp-nodes)
terms_drv['dudx'] = get_dphidx(data['u'], wn_x)
terms_drv['dudy'] = get_dphidy(data['u'], wn_y)
terms_drv['dudz'] = get_dphidz(data['u'], dm)
terms_drv['dudz'] = wnode2uvpnode(terms_drv['dudz'])

terms_drv['dvdx'] = get_dphidx(data['v'], wn_x)
terms_drv['dvdy'] = get_dphidy(data['v'], wn_y)
terms_drv['dvdz'] = get_dphidz(data['v'], dm)
terms_drv['dvdz'] = wnode2uvpnode(terms_drv['dvdz'])

terms_drv['dwdx'] = get_dphidx(data['w'], wn_x)
terms_drv['dwdx'] = wnode2uvpnode(terms_drv['dwdx'])
terms_drv['dwdy'] = get_dphidy(data['w'], wn_y)
terms_drv['dwdy'] = wnode2uvpnode(terms_drv['dwdy'])
terms_drv['dwdz'] = get_dphidz(data['w'], dm)

# Calculate the advection term (all on uvp-nodes)
print('Calculating the advection term')
ue = data['u']*terms_ptb['tke']
duedx = get_dphidx(ue, wn_x)
ve = data['v']*terms_ptb['tke']
dvedy = get_dphidy(ve, wn_y)
tkez = uvpnode2wnode(terms_ptb['tke'])
we = data['w']*tkez
dwedz = get_dphidz(we, dm)
terms_bdg['adv_h'] = -duedx - dvedy
terms_bdg['adv_v'] = -dwedz

#SGS
ue_sgs = data['u']*terms_ptb['tke_SGS']
due_sgsdx = get_dphidx(ue_sgs, wn_x)
ve_sgs = data['v']*terms_ptb['tke_SGS']
dve_sgsdy = get_dphidy(ve_sgs, wn_y)
tke_sgsz = uvpnode2wnode(terms_ptb['tke_SGS'])
we_sgs = data['w']*tke_sgsz
dwe_sgsdz = get_dphidz(we_sgs, dm)
terms_bdg['advSGS_h'] = -due_sgsdx - dve_sgsdy
terms_bdg['advSGS_v'] = -dwe_sgsdz

terms_bdg['adv'] = terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['advSGS_h'] + terms_bdg['advSGS_v']

# Calculate the turbulent transport term (horizontal, all on uvp-nodes)
print('Calculating the horizontal turbulent transport term')
uw_c = wnode2uvpnode(data['uw'])
vw_c = wnode2uvpnode(data['vw'])
w_c = wnode2uvpnode(data['w'])
w2_c = wnode2uvpnode(data['w2'])

terms_ptb['u3_t'] = data['u3'] - 3*data['u']*data['u2'] + 2*data['u']**3
terms_ptb['uv2_t'] = data['uv2'] - 2*data['v']*data['uv']\
  + 2*data['u']*data['v']**2 - data['u']*data['v2']
terms_ptb['uw2_t'] = data['uw2'] - 2*w_c*uw_c + 2*data['u']*w_c**2\
  - data['u']*w2_c
ue = 0.5*(terms_ptb['u3_t']+terms_ptb['uv2_t']+terms_ptb['uw2_t'])
duedx = get_dphidx(ue, wn_x)
terms_ptb['utxx_t'] = data['utxx'] - data['u']*data['txx']
terms_ptb['vtxy_t'] = data['vtxy'] - data['v']*data['txy']
terms_ptb['wtxz_t'] = data['wtxz'] - data['w']*data['txz']
terms_ptb['wtxz_t'] = wnode2uvpnode(terms_ptb['wtxz_t'])
dutaudx = get_dphidx(0.5*(terms_ptb['utxx_t']+terms_ptb['vtxy_t']
  +terms_ptb['wtxz_t']), wn_x)
terms_bdg['uturb_h'] = -duedx-dutaudx

terms_ptb['u2v_t'] = data['u2v'] - 2*data['u']*data['uv']\
  + 2*data['v']*data['u']**2 - data['v']*data['u2']
terms_ptb['v3_t'] = data['v3'] - 3*data['v']*data['v2'] + 2*data['v']**3
terms_ptb['vw2_t'] = data['vw2'] - 2*w_c*vw_c + 2*data['v']*w_c**2\
  - data['v']*w2_c
ve = 0.5*(terms_ptb['u2v_t']+terms_ptb['v3_t']+terms_ptb['vw2_t'])
dvedy = get_dphidy(ve, wn_y)
terms_ptb['utxy_t'] = data['utxy'] - data['u']*data['txy']
terms_ptb['vtyy_t'] = data['vtyy'] - data['v']*data['tyy']
terms_ptb['wtyz_t'] = data['wtyz'] - data['w']*data['tyz']
terms_ptb['wtyz_t'] = wnode2uvpnode(terms_ptb['wtyz_t'])
dutaudy = get_dphidy(0.5*(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
  +terms_ptb['wtyz_t']), wn_y)
terms_bdg['uturb_h'] += -dvedy-dutaudy

# Calculate the turbulent transport term (vertical, all on uvp-nodes)
print('Calculating the vertical turbulent transport term')
u2_h = uvpnode2wnode(data['u2'])
v2_h = uvpnode2wnode(data['v2'])
terms_ptb['wu2_t'] = data['u2w'] - 2*u_h*data['uw'] + 2*data['w']*u_h**2\
  - data['w']*u2_h
terms_ptb['wv2_t'] = data['v2w'] - 2*v_h*data['vw'] + 2*data['w']*v_h**2\
  - data['w']*v2_h
terms_ptb['w3_t'] = data['w3'] - 3*data['w']*data['w2'] + 2*data['w']**3
we = 0.5*(terms_ptb['wu2_t']+terms_ptb['wv2_t']+terms_ptb['w3_t'])
dwedz = get_dphidz(we, dm)
terms_ptb['utxz_t'] = data['utxz'] - u_h*data['txz']
terms_ptb['vtyz_t'] = data['vtyz'] - v_h*data['tyz']
terms_ptb['wtzz_t'] = data['wtzz'] - data['w']*data['tzz']
dutaudz = get_dphidz(0.5*(terms_ptb['utxz_t']+terms_ptb['vtyz_t']
  +terms_ptb['wtzz_t']), dm)
terms_bdg['uturb_v'] = -dwedz-dutaudz

terms_bdg['ttrans'] = terms_bdg['uturb_h'] + terms_bdg['uturb_v']

# Calculate the pressure transport term (all on uvp-nodes)
print('Calculating the pressure transport term')
terms_ptb['pu_t'] = data['pu'] - data['p']*data['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

terms_ptb['pv_t'] = data['pv'] - data['p']*data['v']
dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)

p_h = uvpnode2wnode(data['p'])
terms_ptb['pw_t'] = data['pw'] - p_h*data['w']
dpwdz = get_dphidz(terms_ptb['pw_t'], dm)

terms_bdg['pturb_h'] = -dpudx-dpvdy
terms_bdg['pturb_v'] = -dpwdz

terms_bdg['ptrans'] = terms_bdg['pturb_h'] + terms_bdg['pturb_v']

# Calculate the dissipation rate (all on uvp-nodes)
print('Calculating the dissipation term')
terms_drv['S11'] = terms_drv['dudx']
terms_drv['S12'] = 0.5*(terms_drv['dudy'] + terms_drv['dvdx'])
terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
terms_drv['S22'] = terms_drv['dvdy']
terms_drv['S23'] = 0.5*(terms_drv['dvdz'] + terms_drv['dwdy'])
terms_drv['S33'] = terms_drv['dwdz']

terms_bdg['dissip'] = data['dissip']\
  - data['txx']*terms_drv['S11'] - data['tyy']*terms_drv['S22']\
  - data['tzz']*terms_drv['S33']\
  - 2*data['txy']*terms_drv['S12'] - 2*data['txz']*terms_drv['S13']\
  - 2*data['tyz']*terms_drv['S23']

terms_bdg['canopy'] = data['wFcz'] - data['w']*data['Fcz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data['uFcx']-data['u']*data['Fcx'])\
  +(data['vFcy']-data['v']*data['Fcy'])
  
terms_bdg['totdis'] = terms_bdg['dissip'] + terms_bdg['canopy']

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] + terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] + terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] + terms_ptb['vw_t']*terms_drv['dwdy']
  + data['txx']*terms_drv['S11'] + data['tyy']*terms_drv['S22']
  + 2*data['txy']*terms_drv['S12'] + data['txz']*terms_drv['dwdx']
  + data['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  + data['txz']*terms_drv['dudz'] + data['tyz']*terms_drv['dvdz']
  + data['tzz']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

terms_bdg['prod'] = terms_bdg['prod_h'] + terms_bdg['prod_v']

terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] + terms_bdg['canopy'] + terms_bdg['dissip']\
                    + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
                    + terms_bdg['adv_h'] + terms_bdg['adv_v']

print('*'*80)

var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
           'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']

for i in range(len(var_bdg)):
    terms_bdg[var_bdg[i]][(dist < 0)] = float("nan")

#%% Remove topography to create terrain following averages

var_topo = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
            'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']

terms_topo = dict()

for i in range(len(var_topo)):
    terms_topo[var_topo[i]] = terrainfollowing_3D(terms_bdg[var_bdg[i]])


#%% Profiles of TKE Budget

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/UCLA_tke/real/'

z_ax = np.arange(0,Nz)*dm.dz/h_canopy

plt.figure(figsize=(4,8))
# plt.plot(np.nanmean(terms_bdg['res'],axis=(1,2)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(1,2)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(1,2)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(1,2)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.nanmean(terms_bdg['dissip'],axis=(1,2)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.nanmean(terms_bdg['canopy'],axis=(1,2)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(1,2)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
plt.plot(np.nanmean(terms_bdg['res'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='pink',marker='o',markevery=5,label='res')
plt.plot(np.nanmean(terms_bdg['adv'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='red',marker='s',markevery=5,label='Adv')
plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
plt.plot(np.nanmean(terms_bdg['dissip'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='green',marker='v',markevery=5,label='Diss')
plt.plot(np.nanmean(terms_bdg['canopy'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='orange',marker='o',markevery=5,label='Can')
plt.plot(np.nanmean(terms_bdg['prod'],axis=(1,2))/np.nanmean(-terms_bdg['totdis'],axis=(1,2)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.vlines(1,0,5,linestyle='--',colors='grey')
plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(0,10)
plt.xlim(-2,2.5)
plt.title(r'Real')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/h_C$')
plt.legend(loc='upper left')
plt.show()

# plt.savefig(path_fig+'TKE_Budget_PlanarH10.png',dpi=300,facecolor='white', edgecolor='white')


#%% Graphical Representation of U,W,TKE:

#Set values below the topography equal to NAN:
data['u'][(dist < 0)] = float("nan")
data['w'][(dist < 0)] = float("nan")
terms_ptb['tke'][(dist < 0)] = float("nan")

yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,data['u'][:,yslice,:],cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,data['w'][:,yslice,:],cmap='YlGnBu',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,terms_ptb['tke'][:,yslice,:],cmap='YlGnBu',shading='gouraud')

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

axs[0].plot((x_ax),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[1].plot((x_ax),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[2].plot((x_ax),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')


plt.show()

# plt.savefig(path_fig+'u_w_tke_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')


#%% Fix for the flat case where we are missing data for the dissipation:
    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
yslice = int(dm.ny/2)
x_ax = np.arange(0,Nx)*Dx/dm.zi
z_ax = np.arange(0,Nz)*Dz/dm.zi

#Set values below the topography equal to NAN:

#TKE Production:
Prod = terms_bdg['prod_h'] + terms_bdg['prod_v']

Dissip = np.zeros((Nz,Ny,Nx),dtype = 'float')

#TKE Dissipation:
Dissip[h_canopy:-1,:,:] = - Prod[h_canopy:-1,:,:]
Dissip[0:h_canopy,:,:] = - Prod[h_canopy,:,:]

#TKE Residual:
Res = (Prod + Dissip)
#norm_Res = Res/Dissip

#Set the values of the variables under the topography as NaN
Prod[(dist < 0)] = float("nan")
Dissip[(dist < 0)] = float("nan")
Res[(dist < 0)] = float("nan")


Dissip[0:h_canopy,:,:] = -Prod[h_canopy,:,:]

fig, axs=plt.subplots(3,1,figsize=(6,6),constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,Prod[:,yslice,:],vmin = 0,vmax = 60, cmap='Reds',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip[:,yslice,:],vmin = -60,vmax = 0,cmap='Blues_r',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,Res[:,yslice,:],vmin = -40,vmax = 40,cmap='bwr',shading='gouraud')


axs[0].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[1].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[2].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')

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

# plt.savefig(path+'Figures/'+'Test_P_D_Res_2Dslice.png',dpi=300,facecolor='white', edgecolor='white')


#%% Graphical Representation of Production, Dissipation, and Residual:

    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
yslice = int(dm.ny/2)
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
plt1 = axs[0].pcolormesh(x_ax,z_ax,Prod[:,yslice,:],vmin = 0,vmax = 60, cmap='Reds',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip[:,yslice,:],vmin = -60,vmax = 0,cmap='Blues_r',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,Res[:,yslice,:],vmin = -40,vmax = 40,cmap='bwr',shading='gouraud')


axs[0].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[1].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[2].plot(x_ax,(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')

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

#%% Graphical Representation of Residual/Dissipation:

    
import scipy as sp
import scipy.ndimage

#y-Slice at which we will plot the results.
yslice = int(dm.ny/2)
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
plt1 = axs[0].pcolormesh(x_ax,z_ax,Residual[:,yslice,:],vmin = -1.5,vmax = 1.5,cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip[:,yslice,:],vmin = -60,vmax = 20,cmap='Reds',shading='gouraud')



axs[0].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')
axs[1].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')


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
        XYavg_phi = np.nanmean(phi,axis=(1,2))
    else:
        phi_tmp = terrainfollowing_3D(phi)
        XYavg_phi = np.nanmean(phi_tmp,axis=(1,2))
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
        tmp = np.nanmean(phi,axis=(1,2))
        DF = np.zeros((dm.nz,dm.ny,dm.nx),dtype='float')
        for k in range(0,Nz):
            DF[k,:,:] = phi[k,:,:] - tmp[k]
    else:
        phi_tmp = terrainfollowing_3D(phi)
        tmp = np.nanmean(phi_tmp,axis=(1,2))
        DF = np.zeros((dm.nz,dm.ny,dm.nx),dtype='float')
        for k in range(0,Nz):
            DF[k,:,:] = phi_tmp[k,:,:] - tmp[k]
      
    return DF

    
# Initialization
items_Dispbdg = ('prod','Dprod','Ttransp','Dtransp')

terms_avgdrv = dict()
terms_DispCorr = dict()
terms_Dispbdg = dict.fromkeys(items_Dispbdg, None)

condition = 'true'


terms_avgdrv['du_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data['u'],dist,condition), dm)
terms_avgdrv['du_avgdz'] = XYavg_wnode2uvpnode(terms_avgdrv['du_avgdz'])
terms_avgdrv['dv_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data['v'],dist,condition), dm)
terms_avgdrv['dv_avgdz'] = XYavg_wnode2uvpnode(terms_avgdrv['dv_avgdz'])
terms_avgdrv['dw_avgdz'] = get_dphi_avgdz(XYavg_withTopo(data['w'],dist,condition), dm)


#----------------------------------------------------------------------------
# Calculating the space averaged production terms (all on uvp-nodes)
print('Calculating the averaged production term')

terms_Dispbdg['prod'] = -(XYavg_withTopo(terms_ptb['uw_t'],dist,condition)*terms_avgdrv['du_avgdz'] 
                            + XYavg_withTopo(terms_ptb['vw_t'],dist,condition)*terms_avgdrv['dv_avgdz']
                            + XYavg_withTopo(terms_ptb['w2_t'],dist,condition)*terms_avgdrv['dw_avgdz']
                            + XYavg_withTopo(data['txz'],dist,condition)*terms_avgdrv['du_avgdz']
                            + XYavg_withTopo(data['tyz'],dist,condition)*terms_avgdrv['dv_avgdz']
                            + XYavg_withTopo(data['tzz'],dist,condition)*terms_avgdrv['dw_avgdz']
                            )

print('Calculating the Dispersive production term')


terms_DispCorr['prodCorr'] = -((DispFluct_withTopo(terms_ptb['u2_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudx'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uv_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudy'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudz'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uv_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdx'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['v2_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdy'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['vw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdz'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['uw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdx'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['vw_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdy'],dist,dm,condition))
                           + (DispFluct_withTopo(terms_ptb['w2_t'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdz'],dist,dm,condition))
                           + (DispFluct_withTopo(data['txx'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudx'],dist,dm,condition))
                           + (DispFluct_withTopo(data['txy'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudy'],dist,dm,condition))
                           + (DispFluct_withTopo(data['txz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dudz'],dist,dm,condition))
                           + (DispFluct_withTopo(data['txy'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdx'],dist,dm,condition))
                           + (DispFluct_withTopo(data['tyy'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdy'],dist,dm,condition))
                           + (DispFluct_withTopo(data['tyz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dvdz'],dist,dm,condition))
                           + (DispFluct_withTopo(data['txz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdx'],dist,dm,condition))
                           + (DispFluct_withTopo(data['tyz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdy'],dist,dm,condition))
                           + (DispFluct_withTopo(data['tzz'],dist,dm,condition)*DispFluct_withTopo(terms_drv['dwdz'],dist,dm,condition))
                           )

if condition=='true':
    terms_Dispbdg['Dprod'] = XYavg_withTopo(terms_DispCorr['prodCorr'],dist,condition)
else:
    terms_Dispbdg['Dprod'] = np.nanmean(terms_DispCorr['prodCorr'],axis=(1,2))

#----------------------------------------------------------------------------

# Calculate the space averaged turbulent transport term (vertical, all on uvp-nodes)

print('Calculating the averaged turbulent transport term')

# From the TKE transport term computed earlier compute first the space average and then the vertical gradient:
tmp = - (we + terms_ptb['utxz_t'] + terms_ptb['vtyz_t'] + terms_ptb['wtzz_t'])
terms_Dispbdg['Ttransp'] =  get_dphi_avgdz(XYavg_withTopo(tmp,dist,condition),dm)


print('Calculating the dispersive turbulent transport term')

#terms_DispCorr['transpCorr'] = - DispFluct_withTopo(data['w'],dist,dm)*DispFluct_withTopo(terms_ptb['tke'],dist,dm)
#terms_Dispbdg['Dtransp'] =  get_dphi_avgdz(XYavg_withTopo(terms_DispCorr['transpCorr'],dist),dm)

terms_DispCorr['transpCorr'] = - get_dphidz(DispFluct_withTopo(data['w'],dist,dm,condition)*(DispFluct_withTopo(terms_ptb['tke'],dist,dm,condition)+DispFluct_withTopo(terms_ptb['tke_SGS'],dist,dm,condition)),dm)

if condition=='true':
    terms_Dispbdg['Dtransp'] =  XYavg_withTopo(terms_DispCorr['transpCorr'],dist,condition)
else:
    terms_Dispbdg['Dtransp'] =  np.nanmean(terms_DispCorr['transpCorr'],axis=(1,2))

#%% Profile of Dispersive and Non-dispersive production

z_ax = np.arange(0,Nz)*dm.dz/h_canopy#(dm.dz/dm.zi)

plt.figure(figsize=(4,8))
plt.plot(terms_Dispbdg['prod'],z_ax,c='grey',marker='o',markevery=5,label='<P>')
plt.plot(terms_Dispbdg['Dprod'],z_ax,c='black',marker='s',markevery=5,label='<P">')
# plt.plot(terms_Dispbdg['prod'] + terms_Dispbdg['Dprod'],z_ax,c='grey',marker='o',markevery=5,label='<P> + <P">')
# plt.plot(np.nanmean(terms_topo['prod'],axis=(1,2)),z_ax,c='black',marker='s',markevery=5,label='<P>')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.ylim(0,10)
plt.xlim(-40,60)
if condition=='true':
    plt.title(r'Real - Planar')
else:
    plt.title(r'Real - Terrain')
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
plt.plot(terms_Dispbdg['Ttransp'],z_ax,c='grey',marker='o',markevery=5,label='<T>')
plt.plot(terms_Dispbdg['Dtransp'],z_ax,c='black',marker='s',markevery=5,label='<T">')
# plt.plot(terms_Dispbdg['Ttransp'] + terms_Dispbdg['Dtransp'],z_ax,c='grey',marker='o',markevery=5,label='<T> + <T">')
# plt.plot(np.nanmean(terms_topo['ttrans'],axis=(1,2)),z_ax,c='black',marker='s',markevery=5,label='<T>')
plt.hlines(h_canopy/h_canopy,-60,80,linestyle='--',colors='k')
plt.hlines(2*(h_canopy/h_canopy),-60,80,linestyle='--',colors='k')
plt.ylim(0,10)
plt.xlim(-40,60)
if condition=='true':
    plt.title(r'Real - Planar')
else:
    plt.title(r'Real - Terrain')
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
plt1 = axs[0].pcolormesh(x_ax,z_ax*dm.zi,terms_DispCorr['prodCorr'][:,yslice,:],vmin = -40, vmax= 40, cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax*dm.zi,tmp[:,yslice,:],cmap='YlGnBu',shading='gouraud')


axs[0].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')
axs[1].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')

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
plt1 = axs[0].pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['prodCorr'][:,yslice,:],vmin = -10, vmax= 10, cmap='bwr',shading='gouraud')
plt2 = axs[1].pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['transpCorr'][:,yslice,:],vmin = -5, vmax= 5, cmap='bwr',shading='gouraud')

axs[0].plot((x_ax/dm.zi),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs[1].plot((x_ax/dm.zi),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')

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
plt1 = axs.pcolormesh((x_ax/dm.zi),z_ax,terms_DispCorr['prodCorr'][:,yslice,:],vmin = -10, vmax= 10, cmap='bwr',shading='gouraud')

axs.plot((x_ax/dm.zi),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')

axs.set_ylim([0,1])
axs.set_ylabel(r'$z/z_i$')
axs.set_title(r'$D_{\mathcal{P}} (x,y_{ny/2},z)$')
axs.set_xlabel(r'$x/z_i$')

fig.colorbar(plt1,ax=axs)


# plt.savefig(path_fig+'Disp_Prod_2DSlice.png',dpi=300,facecolor='white', edgecolor='white')

#%% Computing vorticity:

order = 1

vort_x = terms_drv['dwdy'] - terms_drv['dvdz']
vort_y = terms_drv['dudz'] - terms_drv['dwdx']
vort_z = terms_drv['dvdx'] - terms_drv['dudy']


fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

#y-vorticity vertical slice
plt1 = axs.pcolormesh(x_ax,z_ax,vort_y[:,yslice,:],cmap= 'bwr',vmin=-50,vmax = 50)
axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data['u'],dist,dm)[:,yslice,:], DispFluct_withTopo(data['w'],dist,dm)[:,yslice,:],
               density = 1,color=[0.7,0.7,0.7])
axs.plot((x_ax),(topodata['intf'][:,yslice] + h_canopy)/dm.zi,'--k')
axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs[0].hlines(values_zlh[n]/zi,x_ax[0],x_ax[-1],linestyles='--',color='gray')
axs.set_ylim([0,1])
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
plt.show()

# plt.savefig(path_fig+'Vorticity_Streamlines.png',dpi=300,facecolor='white', edgecolor='white')




