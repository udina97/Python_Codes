#%%import modules
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Import data for a coarse and a fine resolution

coarse = 'test_tkesgs_2Delta'
fine = 'test_tkesgs_128'

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

step = 50000   

data_coarse = read_checkpoint_mom(path+coarse+'/output_checkpoint/',[step],32,32,32)
data_fine = read_checkpoint_mom(path+fine+'/output_checkpoint/',[step],128,128,128)

#%%Some useful things

nx_coarse = 32 ; nx_fine = 128
ny_coarse = 32 ; ny_fine = 128
nz_coarse = 32 ; nz_fine = 128

lx = 2*np.pi
ly = 2*np.pi
lz = 1

dx_coarse = lx/nx_coarse ; dx_fine = lx/nx_fine
dy_coarse = ly/ny_coarse ; dy_fine = ly/ny_fine
dz_coarse = lz/nz_coarse ; dz_fine = lz/nz_fine

zi = 1000
uscale = 0.45
Tscale = 290
x_coarse = np.arange(0,nx_coarse)*dx_coarse
y_coarse = np.arange(0,ny_coarse)*dy_coarse
z_uvp_coarse = np.arange(0,nz_coarse)*dz_coarse + dz_coarse/2
z_w_coarse = np.arange(0,nz_coarse)*dz_coarse
x_fine = np.arange(0,nx_fine)*dx_fine
y_fine = np.arange(0,ny_fine)*dy_fine
z_uvp_fine = np.arange(0,nz_fine)*dz_fine + dz_fine/2
z_w_fine = np.arange(0,nz_fine)*dz_fine

#%%Plot pcolor of anisotropy

fig,axs = plt.subplots(1,2,tight_layout=True, figsize=(10,5))

axs[0].pcolormesh(x_coarse,z_uvp_coarse,np.median(data_coarse['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x_coarse,z_uvp_coarse,checkpnt['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x_coarse,z_w_coarse,checkpnt['yB'][nx//2,:,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x_coarse,y_coarse,checkpnt['yB'][:,:,15].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs[1].pcolormesh(x_fine,z_uvp_fine,np.median(data_fine['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x_fine,z_uvp_fine,checkpnt['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x_fine,z_w_fine,checkpnt['yB'][nx//2,:,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x_fine,y_fine,checkpnt['yB'][:,:,15].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

plt.show()

#%%PDF of yB at the closest horizontal plane to a target height

target_height_m = 100.0

def closest_height_level(z_coord, target_height_m, zi_value):
    z_coord = np.asarray(z_coord)

    # The vertical coordinates in this script are nondimensional z/zi.
    # Convert them to meters before selecting the closest level.
    z_m = z_coord*zi_value

    level = int(np.nanargmin(np.abs(z_m - target_height_m)))
    return level, z_m[level]

def compute_pdf(data, n=1000):
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]

    x_pdf = np.linspace(np.min(data), np.max(data), n)
    kde = gaussian_kde(data)
    pdf = kde(x_pdf)

    return x_pdf, pdf

zlevel_coarse, zheight_coarse = closest_height_level(z_uvp_coarse, target_height_m, zi)
zlevel_fine, zheight_fine = closest_height_level(z_uvp_fine, target_height_m, zi)

yB_coarse_plane = np.asarray(data_coarse['yB'][:, :, zlevel_coarse]).ravel()
yB_fine_plane = np.asarray(data_fine['yB'][:, :, zlevel_fine]).ravel()
yB_coarse_plane = yB_coarse_plane[np.isfinite(yB_coarse_plane)]
yB_fine_plane = yB_fine_plane[np.isfinite(yB_fine_plane)]

x_pdf_coarse, pdf_coarse = compute_pdf(yB_coarse_plane)
x_pdf_fine, pdf_fine = compute_pdf(yB_fine_plane)

fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(6, 4))

# axs.hist(yB_coarse_plane, bins=50, density=True, alpha=0.35, color='tab:blue',
#          label=f'coarse hist: z={zheight_coarse:.1f} m')
axs.plot(x_pdf_coarse, pdf_coarse, color='tab:blue', lw=2,
         label='coarse KDE')

# axs.hist(yB_fine_plane, bins=500, density=True, alpha=0.35, color='tab:red',
#          label=f'fine hist: z={zheight_fine:.1f} m')
axs.plot(x_pdf_fine, pdf_fine, color='tab:red', lw=2,
         label='fine KDE')

axs.set_xlabel(r'$y_B$', fontsize=14)
axs.set_ylabel('PDF', fontsize=14)
axs.set_title(r'$y_B$ PDF from horizontal plane near ' + f'{target_height_m:.0f} m', fontsize=14)
axs.legend()
axs.set_xlim(0, np.sqrt(3)/2)

plt.show()

# %%

yB_fine = data_fine['yB']

# %%
