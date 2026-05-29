#%%Load libraries

import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import math

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from Anisotropy_Functions import ColorAnisotropy

#%%Set Path to simulation

path_to_sim = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 't_hom_1ms_noaniso_uvp'
path_data = path_to_sim + sim + '/data/'

#%%Set simulation parameters

nx = 64
ny = 64
nz = 64
lx = 2*np.pi
ly = 2*np.pi
lz = 2
dx = lx/nx
dy = ly/ny
dz = lz/nz

uscale = 0.4
dt = 0.1
zi = 1000
Ug = 1
Tscale = 290

#%%Load the data
iter = 90000
data_momentum = xr.open_dataarray(path_data+'Momentum3D/'+'Data_Momentum_'+str(iter)+'.nc')
data_scalar = xr.open_dataarray(path_data+'Scalar3D/'+'Data_Scalar_'+str(iter)+'.nc')
data_aniso = xr.open_dataarray(path_data+'Anisotropy/'+'Data_Anisotropy_'+str(iter)+'.nc')

#%%Plot

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.pcolormesh(np.arange(0,nx),np.arange(0,nz),data_momentum.data[:,24,:,2].T,cmap='jet')
plt.show()
# %%

yB = data_aniso.data[:,:,:,1]
xB = data_aniso.data[:,:,:,0]

has_negatives = np.any(yB > np.sqrt(3)/2)

print(has_negatives)


# %% Plot Anisotropy

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,yB[:,ny//2,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
cbar = plt.colorbar(p)
plt.show()

# %%
