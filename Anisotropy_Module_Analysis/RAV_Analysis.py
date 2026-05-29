#%%Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode

#%%Import Data

aniso = 'patch256_aniso_1ms'
classic = 'patch256_classic_1ms'
# aniso = 'homo_unstable_aniso_1ms'
# classic = 'homo_unstable_classic_1ms'

path_aniso = '/scratch/general/nfs1/u1450851/LES_Sims/'+aniso
path_classic = '/scratch/general/nfs1/u1450851/LES_Sims/'+classic

mom3D_a = xr.open_dataarray(path_aniso+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_a = xr.open_dataarray(path_aniso+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_a = xr.open_dataarray(path_aniso+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_a = xr.open_dataarray(path_aniso+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

mom3D_c = xr.open_dataarray(path_classic+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_c = xr.open_dataarray(path_classic+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_c = xr.open_dataarray(path_classic+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_c = xr.open_dataarray(path_classic+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Compute ustar and heat flux

uw = mom3D_a[:,:,:,14].data - uvpnode2wnode(mom3D_a[:,:,:,0].data)*mom3D_a[:,:,:,2].data - mom3D_a[:,:,:,20].data
vw = mom3D_a[:,:,:,15].data - uvpnode2wnode(mom3D_a[:,:,:,1].data)*mom3D_a[:,:,:,2].data - mom3D_a[:,:,:,21].data
ustar_a = (uw**2 + vw**2)**(0.25)
shear_a = np.sqrt(uw**2 + vw**2)
hf_a = sc3D_a[:,:,:,4].data - uvpnode2wnode(sc3D_a[:,:,:,0].data)*mom3D_a[:,:,:,2].data - sc3D_a[:,:,:,7].data

uw = mom3D_c[:,:,:,14].data - uvpnode2wnode(mom3D_c[:,:,:,0].data)*mom3D_c[:,:,:,2].data - mom3D_c[:,:,:,20].data
vw = mom3D_c[:,:,:,15].data - uvpnode2wnode(mom3D_c[:,:,:,1].data)*mom3D_c[:,:,:,2].data - mom3D_c[:,:,:,21].data
ustar_c = (uw**2 + vw**2)**(0.25)
shear_c = np.sqrt(uw**2 + vw**2)
hf_c = sc3D_c[:,:,:,4].data - uvpnode2wnode(sc3D_c[:,:,:,0].data)*mom3D_c[:,:,:,2].data - sc3D_c[:,:,:,7].data

#%%Domain averaged profiles

Ua = np.sqrt((mom3D_a[:,:,:,0].data)**2 + (mom3D_a[:,:,:,1].data)**2)#/wnode2uvpnode(ustar_a)
Uc = np.sqrt((mom3D_c[:,:,:,0].data)**2 + (mom3D_c[:,:,:,1].data)**2)#/wnode2uvpnode(ustar_c)

fig,axs = plt.subplots(1,4,tight_layout=True,sharey=True)

axs[0].plot(np.mean(Ua,axis=(0,1)),z_uvp,c='k',label='A')
axs[0].plot(np.mean(Uc,axis=(0,1)),z_uvp,c='r',label='C')
axs[1].plot(np.mean(wnode2uvpnode(shear_a),axis=(0,1)),z_uvp,c='k',label='A')
axs[1].plot(np.mean(wnode2uvpnode(shear_c),axis=(0,1)),z_uvp,c='r',label='C')
axs[2].plot(np.mean(sc3D_a[:,:,:,0],axis=(0,1)),z_uvp,c='k',label='A')
axs[2].plot(np.mean(sc3D_c[:,:,:,0],axis=(0,1)),z_uvp,c='r',label='C')
axs[3].plot(np.mean(wnode2uvpnode(hf_a),axis=(0,1)),z_uvp,c='k',label='A')
axs[3].plot(np.mean(wnode2uvpnode(hf_c),axis=(0,1)),z_uvp,c='r',label='C')

axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
axs[0].set_xlabel(r"$U$",fontsize=14)
axs[0].set_ylim(0,z_uvp[-1])
axs[1].set_xlabel(r"$\overline{u'w'}$",fontsize=14)
axs[2].set_xlabel(r"$\theta$",fontsize=14)
axs[3].set_xlabel(r"$\overline{w'\theta'}$",fontsize=14)
fig.suptitle(f"{nx}",fontsize=14)
for i in range(len(axs)):
    axs[i].legend()

plt.show()

#%%Compute mean velocity profile

ustar_a = np.mean(mom2D_a[:,:,0].data)
ustar_c = np.mean(mom2D_c[:,:,0].data)
U_a = np.mean(np.sqrt((mom3D_a[:,:,:,0].data)**2 + (mom3D_a[:,:,:,1].data)**2),axis=(0,1))
U_c = np.mean(np.sqrt((mom3D_c[:,:,:,0].data)**2 + (mom3D_c[:,:,:,1].data)**2),axis=(0,1))
fit_levels = [5, 7, 9]

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
axs.plot(U_a/ustar_a,z_uvp,c='r')
axs.plot(U_c/ustar_c,z_uvp,c='k')
axs.plot((U_a/ustar_a)[fit_levels],z_uvp[fit_levels],'ro',label='Aniso Fit Levels')
axs.plot((U_c/ustar_c)[fit_levels],z_uvp[fit_levels],'ko',label='Aniso Fit Levels')
axs.set_yscale('log')
axs.set_xlabel('U / u*',fontsize=14)
axs.set_ylabel('z / zi',fontsize=14)
axs.set_title('Mean Velocity Profile',fontsize=14)
plt.show()

#%%Define function to compute ustar through fitting
from scipy.optimize import curve_fit
def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, u, v, twr=False):    

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]
    kappa = 0.4

    # Levels to use for logarithmic fit
    fit_levels = [10, 11, 12]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

#%%Compute ustar through fitting

z0hi,ustar,U_mean,U_data,z_data,u_fit = compute_ustar(20, z_uvp*zi, np.mean(mom3D_c[:,:,:,0].data,axis=(0,1))*uscale,
                                                       np.mean(mom3D_c[:,:,:,1].data,axis=(0,1))*uscale)
# print('Fitted ustar:', ustar)
# print('Fitted z0hi:', z0hi)

# %%Plot mean velocity profile with fitted ustar

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
z0hi,ustar,U_mean,U_data,z_data,u_fit = compute_ustar(20, z_uvp*zi, np.mean(mom3D_a[:,:,:,0].data,axis=(0,1))*uscale,
                                                       np.mean(mom3D_a[:,:,:,1].data,axis=(0,1))*uscale)
print('Fitted ustar:', ustar)
print('Fitted z0hi:', z0hi)
U_a = np.mean(np.sqrt((mom3D_a[:,:,:,0].data)**2 + (mom3D_a[:,:,:,1].data)**2),axis=(0,1))
axs.plot(U_a/(ustar/uscale),z_uvp,c='r')
axs.plot(1/0.4*np.log(z_uvp/(z0hi/zi)),z_uvp,'r--')
z0hi,ustar,U_mean,U_data,z_data,u_fit = compute_ustar(20, z_uvp*zi, np.mean(mom3D_c[:,:,:,0].data,axis=(0,1))*uscale,
                                                       np.mean(mom3D_c[:,:,:,1].data,axis=(0,1))*uscale)
print('Fitted ustar:', ustar)
print('Fitted z0hi:', z0hi)
U_c = np.mean(np.sqrt((mom3D_c[:,:,:,0].data)**2 + (mom3D_c[:,:,:,1].data)**2),axis=(0,1))
axs.plot(U_c/(ustar/uscale),z_uvp,c='k')
axs.plot(1/0.4*np.log(z_uvp/(z0hi/zi)),z_uvp,'k--')
# axs.plot((U_a/ustar_a)[fit_levels],z_uvp[fit_levels],'ro',label='Aniso Fit Levels')
# axs.plot((U_c/ustar_c)[fit_levels],z_uvp[fit_levels],'ko',label='Aniso Fit Levels')
axs.set_yscale('log')
axs.set_xlabel('U / u*',fontsize=14)
axs.set_ylabel('z / zi',fontsize=14)
axs.set_title('Mean Velocity Profile',fontsize=14)
plt.show()

# %%
