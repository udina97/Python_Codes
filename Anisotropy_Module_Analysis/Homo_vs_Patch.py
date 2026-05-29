#%%Import libraries
 
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from functions import get_dphidx, get_dphidy, get_dphidz, uvpnode2wnode, wnode2uvpnode
from Anisotropy_Functions import ColorAnisotropy,Anisotropy
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
cmap = ColorAnisotropy()

# %%Simulation parameters

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

# %%Import data

# homo = 'homo256_aniso_9ms'
# patch = 'patch256_aniso_9ms'
homo = 'homo256_classic_9ms'
patch = 'patch256_classic_9ms'

path_homo = '/scratch/general/nfs1/u1450851/LES_Sims/'+homo
path_patch = '/scratch/general/nfs1/u1450851/LES_Sims/'+patch

mom3D_h = xr.open_dataarray(path_homo+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_h = xr.open_dataarray(path_homo+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_h = xr.open_dataarray(path_homo+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_h = xr.open_dataarray(path_homo+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

mom3D_p = xr.open_dataarray(path_patch+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_p = xr.open_dataarray(path_patch+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_p = xr.open_dataarray(path_patch+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_p = xr.open_dataarray(path_patch+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

step = 100000   

snap3D_h = read_checkpoint_aniso(path_homo+'/output_checkpoint/',[step],nx,ny,nz)
snap2D_h = read_checkpoint_sfc_L(path_homo+'/output_checkpoint/',[step],nx,ny)

snap3D_p = read_checkpoint_aniso(path_patch+'/output_checkpoint/',[step],nx,ny,nz)
snap2D_p = read_checkpoint_sfc_L(path_patch+'/output_checkpoint/',[step],nx,ny)

#%%Compute fluxes for the RAV data

uw = mom3D_h[:,:,:,14].data - uvpnode2wnode(mom3D_h[:,:,:,0].data)*mom3D_h[:,:,:,2].data - mom3D_h[:,:,:,20].data
vw = mom3D_h[:,:,:,15].data - uvpnode2wnode(mom3D_h[:,:,:,1].data)*mom3D_h[:,:,:,2].data - mom3D_h[:,:,:,21].data
ustar_h = (uw**2 + vw**2)**(0.25)
shear_h = np.sqrt(uw**2 + vw**2)
hf_h = sc3D_h[:,:,:,4].data - uvpnode2wnode(sc3D_h[:,:,:,0].data)*mom3D_h[:,:,:,2].data - sc3D_h[:,:,:,7].data

uw = mom3D_p[:,:,:,14].data - uvpnode2wnode(mom3D_p[:,:,:,0].data)*mom3D_p[:,:,:,2].data - mom3D_p[:,:,:,20].data
vw = mom3D_p[:,:,:,15].data - uvpnode2wnode(mom3D_p[:,:,:,1].data)*mom3D_p[:,:,:,2].data - mom3D_p[:,:,:,21].data
ustar_p = (uw**2 + vw**2)**(0.25)
shear_p = np.sqrt(uw**2 + vw**2)
hf_p = sc3D_p[:,:,:,4].data - uvpnode2wnode(sc3D_p[:,:,:,0].data)*mom3D_p[:,:,:,2].data - sc3D_p[:,:,:,7].data

# %%Compute mean profiles for the RAV data

Uh = np.sqrt((mom3D_h[:,:,:,0].data)**2 + (mom3D_h[:,:,:,1].data)**2)#/wnode2uvpnode(ustar_a)
Up = np.sqrt((mom3D_p[:,:,:,0].data)**2 + (mom3D_p[:,:,:,1].data)**2)#/wnode2uvpnode(ustar_c)

fig,axs = plt.subplots(1,4,tight_layout=True,sharey=True)

axs[0].plot(np.mean(Uh,axis=(0,1)),z_uvp,c='k',label='H')
axs[0].plot(np.mean(Up,axis=(0,1)),z_uvp,c='r',label='P')
axs[1].plot(np.mean(wnode2uvpnode(shear_h),axis=(0,1)),z_uvp,c='k',label='H')
axs[1].plot(np.mean(wnode2uvpnode(shear_p),axis=(0,1)),z_uvp,c='r',label='P')
axs[2].plot(np.mean(sc3D_h[:,:,:,0],axis=(0,1)),z_uvp,c='k',label='H')
axs[2].plot(np.mean(sc3D_p[:,:,:,0],axis=(0,1)),z_uvp,c='r',label='P')
axs[3].plot(np.mean(wnode2uvpnode(hf_h),axis=(0,1)),z_uvp,c='k',label='H')
axs[3].plot(np.mean(wnode2uvpnode(hf_p),axis=(0,1)),z_uvp,c='r',label='P')

axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
axs[0].set_xlabel(r"$U$",fontsize=14)
axs[0].set_ylim(0,z_uvp[-1])
axs[1].set_xlabel(r"$\overline{u'w'}$",fontsize=14)
axs[2].set_xlabel(r"$\theta$",fontsize=14)
axs[3].set_xlabel(r"$\overline{w'\theta'}$",fontsize=14)
for i in range(len(axs)):
    axs[i].legend()

plt.show()

# %%
