"""
Program Author: Marc Calaf.

This program uses Giulia's LES output data, to compute the TKE Budget.

Date created: 7 November 2023
Last date modified: 7 November 2023

To Do: 

"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr


os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


#%% Loading data variables:
    
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_hom-eq_9mps_ug/TKE_RAV_Output/'
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_g1200_9mps_ug/TKE_RAV_Output/'
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_g800_9mps_ug/TKE_RAV_Output/'
path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_g400_9mps_ug/TKE_RAV_Output/'

data_LES = xr.open_dataarray(path+'Data_Momentum_4TKE.nc')

NumVar_LES = 64
NumVar = 49
uscale = 0.313 #For the 9 m/s case
h_canopy = 39


Nx = 256; Ny = 256; Nz = 256 
Zi = 1000
Lx = 2*np.pi*Zi; Ly = 2*np.pi*Zi; Lz = 1*Zi
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
    

var_name = ['u','v','w','p',\
    'uu','vv','ww','uv','uw','vw',\
    'dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
    'txx','tyy','tzz','txy','txz','tyz',\
    'uuu','uvv','uww','vuu','vvv','vww','wuu','wvv','www',\
    'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz',\
    'vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
    'up','vp','wp',\
    'dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']
  

data = dict() #we create an empty dictionary.

for var in var_name:   
    
    #Each Variable is assigned as one element of the dictionary.
    data[var] = np.array(data_LES.loc[:,:,:,var])
    print(var)


items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')


del data_LES

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dm.dz/dm.zi)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :,1:])
  phi_c[:, :, -1] = phi_c[:, :, -2]
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


terms_ptb['u2_t'] = data['uu'] - data['u']**2
terms_ptb['uv_t'] = data['uv'] - data['u']*data['v']
terms_ptb['uw_t'] = data['uw'] - u_h*data['w']
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data['vv'] - data['v']**2
terms_ptb['vw_t'] = data['vw'] - v_h*data['w']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data['ww'] -data['w']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2


#%% Flow Overview plots:
    
z_ax = np.arange(0,Nz)*(dm.dz/dm.zi)

#Mean Velocity profiles
fig, axs=plt.subplots(1,3, constrained_layout=True)

axs[0].plot(np.mean(data['u'],axis=(0,1)),z_ax,label='u')
axs[1].plot(np.mean(data['v'],axis=(0,1)),z_ax,label='v')
axs[2].plot(np.mean(data['w'],axis=(0,1)),z_ax,label='w')

axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
plt.legend()
plt.show()


Rij = dict()
Rij['Rxx'] = data['uu'] - data['u']**2
Rij['Ryy'] = data['vv'] - data['v']**2
Rij['Rzz'] = data['ww'] - data['w']**2
Rij['Rxy'] = data['uv'] - data['u']*data['v']
Rij['Rxz'] = data['uw'] - u_h*data['w']
Rij['Ryz'] = data['vw'] - v_h*data['w']


#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.mean(Rij['Rxx'],axis=(0,1)),z_ax)
axs[0,1].plot(np.mean(Rij['Ryy'],axis=(0,1)),z_ax)
axs[0,2].plot(np.mean(Rij['Rzz'],axis=(0,1)),z_ax)
axs[1,0].plot(np.mean(Rij['Rxy'],axis=(0,1)),z_ax)
axs[1,1].plot(np.mean(Rij['Rxz'],axis=(0,1)),z_ax)
axs[1,2].plot(np.mean(Rij['Ryz'],axis=(0,1)),z_ax)

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
w2_c = wnode2uvpnode(data['ww'])

terms_ptb['u3_t'] = data['uuu'] - 3*data['u']*data['uu'] + 2*data['u']**3
terms_ptb['uv2_t'] = data['uvv'] - 2*data['v']*data['uv']\
  + 2*data['u']*data['v']**2 - data['u']*data['vv']
terms_ptb['uw2_t'] = data['uww'] - 2*w_c*uw_c + 2*data['u']*w_c**2\
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

terms_ptb['u2v_t'] = data['vuu'] - 2*data['u']*data['uv']\
  + 2*data['v']*data['u']**2 - data['v']*data['uu']
terms_ptb['v3_t'] = data['vvv'] - 3*data['v']*data['vv'] + 2*data['v']**3
terms_ptb['vw2_t'] = data['vww'] - 2*w_c*vw_c + 2*data['v']*w_c**2\
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
u2_h = uvpnode2wnode(data['uu'])
v2_h = uvpnode2wnode(data['vv'])
terms_ptb['wu2_t'] = data['wuu'] - 2*u_h*data['uw'] + 2*data['w']*u_h**2\
  - data['w']*u2_h
terms_ptb['wv2_t'] = data['wvv'] - 2*v_h*data['vw'] + 2*data['w']*v_h**2\
  - data['w']*v2_h
terms_ptb['w3_t'] = data['www'] - 3*data['w']*data['ww'] + 2*data['w']**3
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
#Need to correct the LES pressure to be the real pressure (pr).
#P_LES = p_r + 1/2(uu + vv + ww_uvp) + SGS_TKE.
#There is no need to correct the SGS TKE because it is negligible.

data['p_r'] = data['p'] - 1/2*(data['uu'] + data['vv'] + wnode2uvpnode(data['ww'])) 
data['up_r'] = data['up'] - 1/2*(data['uuu'] + data['uvv'] + wnode2uvpnode(data['uww']))\
    - 1/3*(data['utxx'] + data['utyy'] + data['utzz']) 
data['vp_r'] = data['vp'] - 1/2*(data['vuu'] + data['vvv'] + wnode2uvpnode(data['vww']))\
    - 1/3*(data['vtxx'] + data['vtyy'] + data['vtzz'])
data['wp_r'] = data['wp'] - 1/2*(uvpnode2wnode(data['wuu']) + uvpnode2wnode(data['wvv']) + data['www'])\
    - 1/3*(data['wtxx'] + data['wtyy'] + data['wtzz'])

terms_ptb['pu_t'] = data['up_r'] - data['p_r']*data['u']
dpudx = get_dphidx(terms_ptb['pu_t'], wn_x)

terms_ptb['pv_t'] = data['vp_r'] - data['p_r']*data['v']
dpvdy = get_dphidy(terms_ptb['pv_t'], wn_y)

p_h = uvpnode2wnode(data['p_r'])
terms_ptb['pw_t'] = data['wp_r'] - p_h*data['w']
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

#terms_bdg['dissip'] = data['dissip']\
#  - data['txx']*terms_drv['S11'] - data['tyy']*terms_drv['S22']\
#  - data['tzz']*terms_drv['S33']\
#  - 2*data['txy']*terms_drv['S12'] - 2*data['txz']*terms_drv['S13']\
#  - 2*data['tyz']*terms_drv['S23']

#MC: This is a test. Bicheng's version is the one above! Here I use Giulia's approach.
terms_bdg['dissip'] = (data['dxx'] + data['dyy'] + data['dzz'] +\
    data['dxy'] + data['dxz'] + data['dyz']) \
  - data['txx']*terms_drv['S11'] - data['tyy']*terms_drv['S22']\
  - data['tzz']*terms_drv['S33']\
  - 2*data['txy']*terms_drv['S12'] - 2*data['txz']*terms_drv['S13']\
  - 2*data['tyz']*terms_drv['S23']


terms_bdg['canopy'] = data['wfdz'] - data['w']*data['fdz']
terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
terms_bdg['canopy'] += (data['ufdx']-data['u']*data['fdx'])\
  +(data['vfdy']-data['v']*data['fdy'])

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
  + terms_ptb['w2_t']*terms_drv['dwdz'])\
  + data['txz']*terms_drv['dudz'] + data['tyz']*terms_drv['dvdz']\
  + data['tzz']*terms_drv['S33']
  
terms_bdg['prod_dudz'] = -terms_ptb['uw_t']*terms_drv['dudz']

print('*'*80)

Budget = -terms_bdg['dissip'] + terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['prod_h'] + \
    terms_bdg['prod_v'] + terms_bdg['uturb_h'] + terms_bdg['uturb_v'] + terms_bdg['pturb_h'] + \
    terms_bdg['pturb_v'] - terms_bdg['canopy']


#Save the TKE budget terms using numpy:
np.save(path + 'TKE_terms.npy', terms_bdg) 

#To read the file later use:
'''
terms_bdg = np.load('TKE_terms.npy',allow_pickle='TRUE').item()
'''    

#%% Shift budget to the w-nodes:

terms_bdg_wnode = dict()

terms_bdg_wnode['dissip'] = uvpnode2wnode(terms_bdg['dissip'])
terms_bdg_wnode['adv_h'] = uvpnode2wnode(terms_bdg['adv_h'])
terms_bdg_wnode['adv_v'] = uvpnode2wnode(terms_bdg['adv_v'])
terms_bdg_wnode['prod_h'] = uvpnode2wnode(terms_bdg['prod_h'])
terms_bdg_wnode['prod_v'] = uvpnode2wnode(terms_bdg['prod_v'])
terms_bdg_wnode['uturb_h'] = uvpnode2wnode(terms_bdg['uturb_h'])
terms_bdg_wnode['uturb_v'] = uvpnode2wnode(terms_bdg['uturb_v'])
terms_bdg_wnode['pturb_h'] = uvpnode2wnode(terms_bdg['pturb_h'])
terms_bdg_wnode['pturb_v'] = uvpnode2wnode(terms_bdg['pturb_v'])
terms_bdg_wnode['canopy'] = uvpnode2wnode(terms_bdg['canopy'])
    
Budget_wnode = -terms_bdg_wnode['dissip'] + terms_bdg_wnode['adv_h'] + terms_bdg_wnode['adv_v'] + terms_bdg_wnode['prod_h'] + \
    terms_bdg_wnode['prod_v'] + terms_bdg_wnode['uturb_h'] + terms_bdg_wnode['uturb_v'] + terms_bdg_wnode['pturb_h'] + \
    terms_bdg_wnode['pturb_v'] - terms_bdg_wnode['canopy']




#%% Profiles of TKE Budget



z_ax = np.arange(0,Nz)*(dm.dz/h_canopy) + 0.5*(dm.dz/h_canopy)

plt.figure()
plt.plot(-np.mean(terms_bdg['dissip'],axis=(0,1)),z_ax,label='Dissip')
plt.plot(np.mean(terms_bdg['adv_h'],axis=(0,1)),z_ax,label='Advec_h')
plt.plot(np.mean(terms_bdg['adv_v'],axis=(0,1)),z_ax,label='Advec_v')
plt.plot(np.mean(terms_bdg['prod_h'],axis=(0,1)),z_ax,label='Prod_h')
plt.plot(np.mean(terms_bdg['prod_v'],axis=(0,1)),z_ax,label='Prod_v')
plt.plot(np.mean(terms_bdg['uturb_h'],axis=(0,1)),z_ax,label='T-Trans_h')
plt.plot(np.mean(terms_bdg['uturb_v'],axis=(0,1)),z_ax,label='T-Trans_v')
plt.plot(np.mean(terms_bdg['pturb_h'],axis=(0,1)),z_ax,label='p-Trans_h')
plt.plot(np.mean(terms_bdg['pturb_v'],axis=(0,1)),z_ax,label='p-Trans_v')
plt.plot(-np.mean(terms_bdg['canopy'],axis=(0,1)),z_ax,label='Canopy')
plt.plot(np.mean(Budget,axis=(0,1)),z_ax,color='k',label='Budget')
plt.hlines(1,-1500,1500,linestyle='--',colors='k')
plt.ylim(0,5)
plt.xlim(-1500,1500)
#plt.xlim(-40,80)
plt.xlabel(r'$de/dt\,\,(z_i/u_*^3)$')
plt.ylabel(r'$z/h$')
plt.legend()
plt.show()
plt.tight_layout()

plt.savefig(path+'Figures/'+'TKE_Budget.png',dpi=300,facecolor='white', edgecolor='white')


z_ax_wnode = np.arange(0,Nz)*(dm.dz/h_canopy)

plt.figure()
plt.plot(-np.mean(terms_bdg_wnode['dissip'],axis=(0,1)),z_ax_wnode,label='Dissip')
plt.plot(np.mean(terms_bdg_wnode['adv_h'],axis=(0,1)),z_ax_wnode,label='Advec_h')
plt.plot(np.mean(terms_bdg_wnode['adv_v'],axis=(0,1)),z_ax_wnode,label='Advec_v')
plt.plot(np.mean(terms_bdg_wnode['prod_h'],axis=(0,1)),z_ax_wnode,label='Prod_h')
plt.plot(np.mean(terms_bdg_wnode['prod_v'],axis=(0,1)),z_ax_wnode,label='Prod_v')
plt.plot(np.mean(terms_bdg_wnode['uturb_h'],axis=(0,1)),z_ax_wnode,label='T-Trans_h')
plt.plot(np.mean(terms_bdg_wnode['uturb_v'],axis=(0,1)),z_ax_wnode,label='T-Trans_v')
plt.plot(np.mean(terms_bdg_wnode['pturb_h'],axis=(0,1)),z_ax_wnode,label='p-Trans_h')
plt.plot(np.mean(terms_bdg_wnode['pturb_v'],axis=(0,1)),z_ax_wnode,label='p-Trans_v')
plt.plot(-np.mean(terms_bdg_wnode['canopy'],axis=(0,1)),z_ax_wnode,label='Canopy')
plt.plot(np.mean(Budget_wnode,axis=(0,1)),z_ax_wnode,color='k',label='Budget')
plt.hlines(1,-1500,1500,linestyle='--',colors='k')
plt.ylim(0,5)
plt.xlim(-1500,1500)
#plt.xlim(-40,80)
plt.xlabel(r'$de/dt\,\,(z_i/u_*^3)$')
plt.ylabel(r'$z/h$')
plt.legend()
plt.show()
plt.tight_layout()

plt.savefig(path+'Figures/'+'TKE_Budget_wnode.png',dpi=300,facecolor='white', edgecolor='white')
