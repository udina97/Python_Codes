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


os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")



#%% Loading data variables:
    
path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/'
#path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_flat_3D/'


NumVar = 49
uscale = 0.4
h_canopy = 39

#Amazon Flat Canopy
#Nx = 320; Ny = 160; Nz = 260 
#Lx = 2000; Ly = 1000; Lz = 520
#Zi = 520

#Amazon Real Canopy
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


Rij = dict()
Rij['Rxx'] = data['u2'] - data['u']**2
Rij['Ryy'] = data['v2'] - data['v']**2
Rij['Rzz'] = data['w2'] - data['w']**2
Rij['Rxy'] = data['uv'] - data['u']*data['v']
Rij['Rxz'] = data['uw'] - u_h*data['w']
Rij['Ryz'] = data['vw'] - v_h*data['w']


#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.mean(Rij['Rxx'],axis=(1,2)),z_ax)
axs[0,1].plot(np.mean(Rij['Ryy'],axis=(1,2)),z_ax)
axs[0,2].plot(np.mean(Rij['Rzz'],axis=(1,2)),z_ax)
axs[1,0].plot(np.mean(Rij['Rxy'],axis=(1,2)),z_ax)
axs[1,1].plot(np.mean(Rij['Rxz'],axis=(1,2)),z_ax)
axs[1,2].plot(np.mean(Rij['Ryz'],axis=(1,2)),z_ax)

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
dutaudx = get_dphidx(terms_ptb['utxx_t']+terms_ptb['vtxy_t']
  +terms_ptb['wtxz_t'], wn_x)
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
dutaudy = get_dphidy(terms_ptb['utxy_t']+terms_ptb['vtyy_t']
  +terms_ptb['wtyz_t'], wn_y)
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
dutaudz = get_dphidz(terms_ptb['utxz_t']+terms_ptb['vtyz_t']
  +terms_ptb['wtzz_t'], dm)
terms_bdg['uturb_v'] = -dwedz-dutaudz

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

# Calculate the production (all on uvp-nodes)
print('Calculating the production term')
terms_bdg['prod_h'] =\
  -(terms_ptb['u2_t']*terms_drv['dudx'] + terms_ptb['uv_t']*terms_drv['dudy']
  + terms_ptb['uv_t']*terms_drv['dvdx'] + terms_ptb['v2_t']*terms_drv['dvdy']
  + terms_ptb['uw_t']*terms_drv['dwdx'] + terms_ptb['vw_t']*terms_drv['dwdy']
  #+ data['txx']*terms_drv['S11'] + data['tyy']*terms_drv['S22']
  #+ 2*data['txy']*terms_drv['S12'] + data['txz']*terms_drv['dwdx']
  #+ data['tyz']*terms_drv['dwdy']
  )
terms_bdg['prod_v'] =\
  -(terms_ptb['uw_t']*terms_drv['dudz'] + terms_ptb['vw_t']*terms_drv['dvdz']
  + terms_ptb['w2_t']*terms_drv['dwdz']
  #+ data['txz']*terms_drv['dudz'] + data['tyz']*terms_drv['dvdz']
  #+ data['tzz']*terms_drv['S33']
  )
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

print('*'*80)


#%% Profiles of TKE Budget



z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

plt.figure()
plt.plot(np.mean(terms_bdg['dissip'],axis=(1,2)),z_ax,label='Dissip')
plt.plot(np.mean(terms_bdg['adv_h'],axis=(1,2)),z_ax,label='Advec_h')
plt.plot(np.mean(terms_bdg['adv_v'],axis=(1,2)),z_ax,label='Advec_v')
plt.plot(np.mean(terms_bdg['prod_h'],axis=(1,2)),z_ax,label='prod_h')
plt.plot(np.mean(terms_bdg['prod_v'],axis=(1,2)),z_ax,label='prod_v')
plt.plot(np.mean(terms_bdg['uturb_h'],axis=(1,2)),z_ax,label='uturb_h')
plt.plot(np.mean(terms_bdg['uturb_v'],axis=(1,2)),z_ax,label='uturb_v')
plt.plot(np.mean(terms_bdg['pturb_h'],axis=(1,2)),z_ax,label='pturb_h')
plt.plot(np.mean(terms_bdg['pturb_v'],axis=(1,2)),z_ax,label='pturb_v')
plt.plot(np.mean(terms_bdg['canopy'],axis=(1,2)),z_ax,label='canopy')
plt.hlines(h_canopy/dm.zi,-40,80,linestyle='--',colors='k')
plt.ylim(0,1)
plt.legend()
plt.show()



#%% Loading the topography data:

from pymatreader import read_mat

topodata = read_mat(path+'topo.mat')
dist = topodata['Z_gd']

#%% Graphical Representation of U,W,TKE:

#Set values below the topography equal to NAN:
data['u'][(dist < 0)] = float("nan")
data['w'][(dist < 0)] = float("nan")
terms_ptb['tke'][(dist < 0)] = float("nan")

yslice = 188
x_ax = np.arange(0,Nx)*Dx

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1, constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,data['u'][:,yslice,:],cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,data['w'][:,yslice,:],cmap='YlGnBu',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,terms_ptb['tke'][:,yslice,:],cmap='YlGnBu',shading='gouraud')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])



#%% Graphical Representation of Production and Dissipation:

#Set values below the topography equal to NAN:

#TKE Production:
Prod = terms_bdg['prod_h'] + terms_bdg['prod_v']
Prod[(dist < 0)] = float("nan")

#TKE Dissipation:
Dissip = terms_bdg['dissip']
Dissip[(dist < 0)] = float("nan")

#TKE Residual:
Res = Prod + Dissip
Res[(dist < 0)] = float("nan")

yslice = 188
norm_Res = np.zeros((Nz,Nx),dtype='float',order = 'F')
for i in range(0,Nx):
    for k in range(0,Nz):
        norm_Res[k,i] = Res[k,yslice,i]/Dissip[k,yslice,i]


x_ax = np.arange(0,Nx)*Dx
z_ax = np.arange(0,Nz)*Dz

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1, constrained_layout=True)
plt1 = axs[0].pcolormesh(x_ax,z_ax,Prod[:,yslice,:],vmin = 0,vmax = 80, cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(x_ax,z_ax,Dissip[:,yslice,:],vmin = -40,vmax = 0,cmap='YlGnBu',shading='gouraud')
#plt3 = axs[2].pcolormesh(x_ax,z_ax,Res[:,yslice,:],vmin = -80,vmax = 80,cmap='bwr',shading='gouraud')
plt3 = axs[2].pcolormesh(x_ax,z_ax,norm_Res,vmin = -1.5, vmax = 1.5, cmap='bwr',shading='gouraud')

axs[0].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')
axs[1].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')
axs[2].plot(x_ax,topodata['intf'][:,yslice] + h_canopy,'--k')

axs[0].set_ylim([0,200]);axs[1].set_ylim([0,200]);axs[2].set_ylim([0,200])

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])













