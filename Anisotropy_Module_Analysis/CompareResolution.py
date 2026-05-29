#%%Load libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso,\
read_checkpoint_sfc, read_checkpoint_sfc_L

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

# %%Simulation parameters

sp128 = dict(); sp256 = dict()
sp128['nx'] = 128; sp256['nx'] = 256
sp128['ny'] = 128; sp256['ny'] = 256
sp128['nz'] = 128; sp256['nz'] = 256

sp128['lx'] = 2*np.pi; sp256['lx'] = 2*np.pi
sp128['ly'] = 2*np.pi; sp256['ly'] = 2*np.pi 
sp128['lz'] = 2; sp256['lz'] = 2

sp128['dx'] = sp128['lx']/sp128['nx']; sp256['dx'] = sp256['lx']/sp256['nx']
sp128['dy'] = sp128['ly']/sp128['ny']; sp256['dy'] = sp256['ly']/sp256['ny']
sp128['dz'] = sp128['lz']/sp128['nz']; sp256['dz'] = sp256['lz']/sp256['nz']

zi = 1000
uscale = 0.4
Tscale = 290
sp128['x'] = np.arange(0,sp128['nx'])*sp128['dx']; sp256['x'] = np.arange(0,sp256['nx'])*sp256['dx']
sp128['y'] = np.arange(0,sp128['ny'])*sp128['dy']; sp256['y'] = np.arange(0,sp256['ny'])*sp256['dy']
sp128['z_uvp'] = np.arange(0,sp128['nz'])*sp128['dz'] + sp128['dz']/2; sp256['z_uvp'] = np.arange(0,sp256['nz'])*sp256['dz'] + sp256['dz']/2
sp128['z_w'] = np.arange(0,sp128['nz'])*sp128['dz']; sp256['z_w'] = np.arange(0,sp256['nz'])*sp256['dz']

# %%Import data

path_to_data = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim128 = 'patch128_aniso_1ms_v2'
sim256 = 'patch256_aniso_1ms'

step = 100000

d128 = read_checkpoint_aniso(path_to_data+sim128+'/output_checkpoint/',[step],sp128['nx'],sp128['ny'],sp128['nz'])
d128_sfc = read_checkpoint_sfc_L(path_to_data+sim128+'/output_checkpoint/',[step],sp128['nx'],sp128['nz'])

d256 = read_checkpoint_aniso(path_to_data+sim256+'/output_checkpoint/',[step],sp256['nx'],sp256['ny'],sp256['nz'])
d256_sfc = read_checkpoint_sfc_L(path_to_data+sim256+'/output_checkpoint/',[step],sp256['nx'],sp256['ny'])


# %%Plot surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(sp256['x'],sp256['y'],(d256_sfc['sfcVAL']*Tscale).T,cmap='hot_r',vmin=285,vmax=295)
cbar = plt.colorbar(p)
axs.set_xlabel(r"$x/z_i$",fontsize=14)
axs.set_ylabel(r"$y/z_i$",fontsize=14)
plt.show()

# %%Compute anisotropy for the 128 resolution

nx = sp128['nx']; ny = sp128['ny']; nz = sp128['nz']
uu = d128['uu_new'][:,:,:nz] - d128['u_new'][:,:,:nz]*d128['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = d128['vv_new'][:,:,:nz] - d128['v_new'][:,:,:nz]*d128['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = d128['ww_new'][:,:,:nz] - d128['w_new'][:,:,:nz]*d128['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = d128['uv_new'][:,:,:nz] - d128['u_new'][:,:,:nz]*d128['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = d128['uw_new'][:,:,:nz] - d128['u_new'][:,:,:nz]*d128['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = d128['vw_new'][:,:,:nz] - d128['v_new'][:,:,:nz]*d128['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke128 = uu + vv + ww

[xB_128,yB_128,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])


# %%Compute anisotropy for the 256 resolution
nx = sp256['nx']; ny = sp256['ny']; nz = sp256['nz']
uu = d256['uu_new'][:,:,:nz] - d256['u_new'][:,:,:nz]*d256['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = d256['vv_new'][:,:,:nz] - d256['v_new'][:,:,:nz]*d256['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = d256['ww_new'][:,:,:nz] - d256['w_new'][:,:,:nz]*d256['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = d256['uv_new'][:,:,:nz] - d256['u_new'][:,:,:nz]*d256['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = d256['uw_new'][:,:,:nz] - d256['u_new'][:,:,:nz]*d256['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = d256['vw_new'][:,:,:nz] - d256['v_new'][:,:,:nz]*d256['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke256 = uu + vv + ww

[xB_256,yB_256,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

# %%Plot histogram of yB at the surface

lvl = 0

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(yB_128[:,:,lvl].flatten(),bins=50,density=True,alpha=0.4,label='128')
axs.hist(yB_256[:,:,lvl].flatten(),bins=50,density=True,alpha=0.4,color='red',label='256')
axs.set_xlabel(r"$y_B$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

# %%Plot pcolor of anisotropy

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))
axs[0].pcolormesh(sp128['x'],sp128['y'],yB_128[:,:,0].T,cmap=cmap,vmin=0,vmax=0.3)
axs[1].pcolormesh(sp256['x'],sp256['y'],yB_256[:,:,0].T,cmap=cmap,vmin=0,vmax=0.3)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("128",fontsize=14)
axs[1].set_title("256",fontsize=14)

plt.show()

# %%TKE histogram

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(tke128[:,:,0].flatten(),bins=50,density=True,alpha=0.4,label='128')
axs.hist(tke256[:,:,0].flatten(),bins=50,density=True,alpha=0.4,color='red',label='256')
axs.set_xlabel(r"$TKE$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

# %%TKE colormaps

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))
axs[0].pcolormesh(sp128['x'],sp128['y'],tke128[:,:,0].T,cmap='viridis',vmin=0,vmax=10)
axs[1].pcolormesh(sp256['x'],sp256['y'],tke256[:,:,0].T,cmap='viridis',vmin=0,vmax=10)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("128",fontsize=14)
axs[1].set_title("256",fontsize=14)

plt.show()

# %%

fig,axs = plt.subplots(1,1,tight_layout=True)
bins = np.logspace(np.log10(1e-2), np.log10(1e1), 100)
axs.hist(d128_sfc['phi_m'].flatten(),bins=bins,density=True,alpha=0.4,label='128')
axs.hist(d256_sfc['phi_m'].flatten(),bins=bins,density=True,alpha=0.4,color='red',label='256')
axs.set_xlabel(r"$\phi$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xscale('log')
axs.legend()

plt.show()

# %%

fig,axs = plt.subplots(1,1,tight_layout=True)
# bins = np.logspace(np.log10(1e-1), np.log10(1e1), 100)
axs.hist((d128_sfc['psi_m'][(d128_sfc['psi_m']>0)]).flatten(),bins=100,density=True,alpha=0.4,label='128')
axs.hist((d256_sfc['psi_m'][(d256_sfc['psi_m']>0)]).flatten(),bins=100,density=True,alpha=0.4,color='red',label='256')
axs.set_xlabel(r"$\psi$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
# axs.set_xscale('log')
axs.legend()

plt.show()

# %%
