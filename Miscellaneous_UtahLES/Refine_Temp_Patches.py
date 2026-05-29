#%%Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt

#%%Path to temperature patch file

path_to_file = "/uufs/chpc.utah.edu/common/home/u1450851/LES_code/LES-2Ddecomp/build/input/"
file = "surface_SC1.dat"

sfcT = np.fromfile(path_to_file+file,dtype=np.float64)

#%%Reshape the surface temperature array
nx = 128
ny = 128
sfcT = sfcT.reshape((nx,ny),order='F')

# %%Plot the surface Temperature

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(np.arange(0,nx),np.arange(0,ny),sfcT.T,vmin=285,vmax=295,cmap='hot_r')
axs.set_xlabel(r"$nx$",fontsize=20)
axs.set_ylabel(r"$ny$",fontsize=20)
axs.set_title(r"Surface Temperature",fontsize=20)
cbar = plt.colorbar(p)
axs.tick_params(axis='both',which='major',labelsize=15)
cbar.set_label(label="T [K]",size=15)
cbar.ax.tick_params(labelsize=14)

plt.show()

# %%Modify the array resolution to a different one

nx_new = 128
ny_new = 128

# sfcT256 = np.repeat(np.repeat(sfcT,2,axis=0),2,axis=1)
sfcT128 = sfcT.reshape(nx_new, 2, ny_new, 2).mean(axis=(1, 3))


# %%Plot the new high resolution surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(np.arange(0,nx_new),np.arange(0,ny_new),sfcT128.T,vmin=285,vmax=295,cmap='hot_r')
axs.set_xlabel(r"$nx$",fontsize=20)
axs.set_ylabel(r"$ny$",fontsize=20)
axs.set_title(r"Surface Temperature",fontsize=20)
cbar = plt.colorbar(p)
axs.tick_params(axis='both',which='major',labelsize=15)
cbar.set_label(label="T [K]",size=15)
cbar.ax.tick_params(labelsize=14)

plt.show()

# %%Save the new file
filename = 'surface_SC1_256to128.dat'

with open(path_to_file+filename,'wb') as f:
    sfcT128.T.tofile(f)

# %%
