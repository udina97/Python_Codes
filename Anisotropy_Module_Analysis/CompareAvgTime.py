#%%Import libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso,\
read_checkpoint_sfc, read_checkpoint_sfc_L

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 64
ny = 64
nz = 64

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

path_to_data = '/scratch/general/nfs1/u1450851/LES_Sims/'#'Patch/v2/128/'
sim30 = 'test_A'
sim10 = 'test_B'

step = 30000

d30 = read_checkpoint_aniso(path_to_data+sim30+'/output_checkpoint/',[step],nx,ny,nz)
d30_sfc = read_checkpoint_sfc_L(path_to_data+sim30+'/output_checkpoint/',[step],nx,ny)

d10 = read_checkpoint_aniso(path_to_data+sim10+'/output_checkpoint/',[step],nx,ny,nz)
d10_sfc = read_checkpoint_sfc_L(path_to_data+sim10+'/output_checkpoint/',[step],nx,ny)

#%%Plot the surface temperature pattern

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(x,y,(d10_sfc['sfcVAL']*Tscale).T,vmin=285,vmax=295,cmap='hot_r')
axs.set_xlabel(r"$x/z_i$",fontsize=20)
axs.set_ylabel(r"$y/z_i$",fontsize=20)
axs.set_title(r"Surface Temperature",fontsize=20)
cbar = plt.colorbar(p)
axs.tick_params(axis='both',which='major',labelsize=15)
cbar.set_label(label="T [K]",size=15)
cbar.ax.tick_params(labelsize=14)

plt.show()

#%% Flux gradient functions

phiM_30 = d30_sfc['phi_m']
phiH_30 = d30_sfc['phi_h']
psiM_30 = d30_sfc['psi_m']
psiH_30 = d30_sfc['psi_h']

phiM_10 = d10_sfc['phi_m']
phiH_10 = d10_sfc['phi_h']
psiM_10 = d10_sfc['psi_m']
psiH_10 = d10_sfc['psi_h']

L30 = d30_sfc['L']
zoverL_30 = (dz/2)/L30
zoverL_u_30 = zoverL_30[(zoverL_30<0)]
zoverL_s_30 = zoverL_30[(zoverL_30>0)]

L10 = d10_sfc['L']
zoverL_10 = (dz/2)/L10
zoverL_u_10 = zoverL_10[(zoverL_10<0)]
zoverL_s_10 = zoverL_10[(zoverL_10>0)]

# %%Scatter plot of phi/psi vs z/L

fig,axs = plt.subplots(1,1,tight_layout=True)
bins = np.logspace(np.log10(1e-4), np.log10(1e4), 50)
axs.hist(abs(zoverL_u_30),bins=bins,density=True,alpha=0.4,label='30min')
axs.hist(abs(zoverL_u_10),bins=bins,density=True,alpha=0.4,color='red',label='10min')
axs.set_xscale('log')
axs.set_xlabel(r"$|\zeta|$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

#%%---------------------------Anisotropy and TKE-----------------------------------------------------
# %%Compute anisotropy 30 minute window

uu = d30['uu_new'][:,:,:nz] - d30['u_new'][:,:,:nz]*d30['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = d30['vv_new'][:,:,:nz] - d30['v_new'][:,:,:nz]*d30['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = d30['ww_new'][:,:,:nz] - d30['w_new'][:,:,:nz]*d30['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = d30['uv_new'][:,:,:nz] - d30['u_new'][:,:,:nz]*d30['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = d30['uw_new'][:,:,:nz] - d30['u_new'][:,:,:nz]*d30['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = d30['vw_new'][:,:,:nz] - d30['v_new'][:,:,:nz]*d30['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke30 = uu + vv + ww

[xB_30,yB_30,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])


# %%Compute anisotropy 10 minute window

uu = d10['uu_new'][:,:,:nz] - d10['u_new'][:,:,:nz]*d10['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = d10['vv_new'][:,:,:nz] - d10['v_new'][:,:,:nz]*d10['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = d10['ww_new'][:,:,:nz] - d10['w_new'][:,:,:nz]*d10['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = d10['uv_new'][:,:,:nz] - d10['u_new'][:,:,:nz]*d10['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = d10['uw_new'][:,:,:nz] - d10['u_new'][:,:,:nz]*d10['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = d10['vw_new'][:,:,:nz] - d10['v_new'][:,:,:nz]*d10['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke10 = uu + vv + ww

[xB_10,yB_10,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

# %%Plot histogram of the surface value of yB

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(yB_30[:,:,0].flatten(),bins=50,density=True,alpha=0.4,label='30min')
axs.hist(yB_10[:,:,0].flatten(),bins=50,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$y_B$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

# %%Pcolormesh of yB at the surface

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))
axs[0].pcolormesh(x,y,yB_30[:,:,0].T,cmap=cmap,vmin=0,vmax=0.3)
axs[1].pcolormesh(x,y,yB_10[:,:,0].T,cmap=cmap,vmin=0,vmax=0.3)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("30 min",fontsize=14)
axs[1].set_title("10 min",fontsize=14)

plt.show()

# %%Compare the TKE at the surface

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))
axs[0].pcolormesh(x,y,tke30[:,:,0].T,cmap='viridis',vmin=0)
axs[1].pcolormesh(x,y,tke10[:,:,0].T,cmap='viridis',vmin=0)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("30 min",fontsize=14)
axs[1].set_title("10 min",fontsize=14)

plt.show()

#%%TKE histogram

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(tke30[:,:,0].flatten(),bins=50,density=True,alpha=0.4,label='30min')
axs.hist(tke10[:,:,0].flatten(),bins=50,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$TKE$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

#%%-------------------------------Wind Fields-----------------------------------------------------------

#%%Compare the velocity field components at the surface, pcolorplots

var = 'u'
lvl = 0

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))
axs[0].pcolormesh(x,y,d30[var][:,:,lvl].T,cmap='viridis',vmin=-5,vmax=5)
axs[1].pcolormesh(x,y,d10[var][:,:,lvl].T,cmap='viridis',vmin=-5,vmax=5)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("30 min",fontsize=14)
axs[1].set_title("10 min",fontsize=14)

plt.show()

#%%Histograms of velocity

var = 'w'
lvl = 10

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(d30[var][:,:,lvl].flatten(),bins=50,density=True,alpha=0.4,label='30min')
axs.hist(d10[var][:,:,lvl].flatten(),bins=50,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$u_i$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

#%%Variances

var30 = d30['ww_new'][:,:,:nz] - d30['w_new'][:,:,:nz]*d30['w_new'][:,:,:nz]
var10 = d10['ww_new'][:,:,:nz] - d10['w_new'][:,:,:nz]*d10['w_new'][:,:,:nz]

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(var30[:,:,0].flatten(),bins=50,density=True,alpha=0.4,label='30min')
axs.hist(var10[:,:,0].flatten(),bins=50,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$u_i'u_i'$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

#%%-----------------------------Flux gradient relations--------------------------------------------------------------

# %%Plot histograms of the phi functions at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)
bins = np.logspace(np.log10(1e-2), np.log10(1e2), 100)
axs.hist(phiM_30.flatten(),bins=bins,density=True,alpha=0.4,label='30min')
axs.hist(phiM_10.flatten(),bins=bins,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$\phi$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xscale('log')
axs.legend()

plt.show()

# %%Plot histograms of the psi functions at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)
# bins = np.logspace(np.log10(1e-2), np.log10(1e2), 100)
axs.hist(psiM_30.flatten(),bins=100,density=True,alpha=0.4,label='30min')
axs.hist(psiM_10.flatten(),bins=100,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$\psi$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
# axs.set_xscale('log')
axs.legend()

plt.show()

# %%Plot pcolor of ustar at sfc

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,5),sharey=True)
axs[0].pcolormesh(x,y,d30_sfc['ustar'].T,cmap='viridis',vmin=0,vmax=1)
axs[1].pcolormesh(x,y,d10_sfc['ustar'].T,cmap='viridis',vmin=0,vmax=1)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("30 min",fontsize=14)
axs[1].set_title("10 min",fontsize=14)

plt.show()

# %%Plot histogram of ustar

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist(d30_sfc['ustar'].flatten(),bins=100,density=True,alpha=0.4,label='30min')
axs.hist(d10_sfc['ustar'].flatten(),bins=100,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$ustar$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
# axs.set_xscale('log')
axs.legend()
plt.show()

# %%Plot pcolor of the heatflux

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,5),sharey=True)
axs[0].pcolormesh(x,y,(d30_sfc['sfcFLUX']*uscale*Tscale*1000).T,cmap='viridis',vmin=-100,vmax=400)
axs[1].pcolormesh(x,y,(d10_sfc['sfcFLUX']*uscale*Tscale*1000).T,cmap='viridis',vmin=-100,vmax=400)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=14)

axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title("30 min",fontsize=14)
axs[1].set_title("10 min",fontsize=14)

plt.show()

# %%Plot histogram of the surface heat flux

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.hist((d30_sfc['sfcFLUX']*uscale*Tscale*1000).flatten(),bins=100,density=True,alpha=0.4,label='30min')
axs.hist((d10_sfc['sfcFLUX']*uscale*Tscale*1000).flatten(),bins=100,density=True,alpha=0.4,color='red',label='10min')
axs.set_xlabel(r"$\overline{w'T'}$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
# axs.set_xscale('log')
axs.legend()

plt.show()

# %%
