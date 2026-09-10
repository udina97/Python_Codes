#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:42:15 2025

@author: u1450851
"""

import numpy as np
import os

def load2decomp_initialcond(folder, nx, ny, nz):
    """
    Load initial condition files.

    Parameters
    ----------
    folder : str
        Path to data folder.
    nx, ny, nz : int
        Grid resolution in x, y, z directions.

    Returns
    -------
    initcond : dict of numpy.ndarray
        Dictionary with keys 'u', 'v', 'w' containing 3D velocity components.
    """
    fs3D1 = ['u', 'v', 'w']
    fn = os.path.join(folder, 'initial_conditions.dat')

    print("="*62)
    print("reading initial conditions file:")
    print(fn)
    print("="*62)

    # Read binary data as doubles
    with open(fn, 'rb') as f:
        tmp = np.fromfile(f, dtype=np.float64)

    len3D = nx * ny * nz
    initcond = {}
    displ = 0

    for key in fs3D1:
        data = tmp[displ : displ + len3D]
        initcond[key] = data.reshape((nx, ny, nz), order='F')  # Fortran order
        displ += len3D

    return initcond

#######################################################################################################################################

import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_initcond(initcond, fs, z_uvp, z_w):
    """
    Plot initial condition vertical profiles.

    Parameters
    ----------
    initcond : dict
        Dictionary with 3D arrays for 'u', 'v', 'w', and optionally 'T', 'q'.
    fs : list of str
        Fields to plot (e.g. ['u', 'v', 'w']).
    z_uvp : 1D array
        z-coordinate for u, v profiles and derived quantities.
    z_w : 1D array
        z-coordinate for w profile.
    cf : str
        Case folder name for output.
    """
    nfs = len(fs)
    npanels = nfs + 1

    fig = plt.figure(figsize=(npanels * 3.5, 5))
    gs = gridspec.GridSpec(1, npanels, wspace=0.3)

    # Compute horizontal averages
    prof = {key: np.mean(np.mean(initcond[key], axis=1), axis=0) for key in initcond}

    # Plot u
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(prof['u'], z_uvp, '-')
    ax.set_xlabel(r'$u/u_*$')
    ax.set_ylabel(r'$z/z_i$')
    ax.grid(True)
    ax.legend([r'$\langle u \rangle_{xy}(z)$'], loc='upper left', frameon=False)

    # Plot v
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(prof['v'], z_uvp, '-')
    ax.set_xlabel(r'$v/u_*$')
    ax.grid(True)
    ax.legend([r'$\langle v \rangle_{xy}(z)$'], loc='upper left', frameon=False)
    ax.set_yticklabels([])

    # Plot w
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(prof['w'], z_w, '-')
    ax.set_xlabel(r'$w/u_*$')
    ax.grid(True)
    ax.legend([r'$\langle w \rangle_{xy}(z)$'], loc='upper left', frameon=False)
    ax.set_yticklabels([])

    # Plot wind direction angle (arctangent of v/u)
    ax = fig.add_subplot(gs[0, 3])
    wind_angle = np.degrees(np.arctan2(prof['v'], prof['u']))
    ax.plot(wind_angle, z_uvp, '-')
    ax.set_xlabel(r'$^\circ$')
    ax.grid(True)
    ax.legend([r'$\alpha(z)$'], loc='upper left', frameon=False)
    ax.set_yticklabels([])

    # Optional: Plot T if available
    if 'T' in prof:
        ax = fig.add_subplot(gs[0, 4])
        ax.plot(prof['T'], z_uvp, '-')
        ax.set_xlabel(r'$T/T_0$')
        ax.grid(True)
        ax.legend([r'$\langle T \rangle_{xy}(z)$'], loc='upper left', frameon=False)
        ax.set_yticklabels([])

    # Optional: Plot q if available
    if 'q' in prof:
        ax = fig.add_subplot(gs[0, 5])
        ax.plot(prof['q'], z_uvp, '-')
        ax.set_xlabel(r'$q$')
        ax.grid(True)
        ax.legend([r'$\langle q \rangle_{xy}(z)$'], loc='upper left', frameon=False)
        ax.set_yticklabels([])

    # Save to PDF
    # output_dir = os.path.join('fig', cf)
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, 'initial_profile.pdf')
    # plt.savefig(output_path, bbox_inches='tight')
    # plt.close(fig)
    
##################################################################################################################################

def load2decomp_initialcond_SC(folder, fs3D1, nx, ny, nz):
    """
    Load initial condition files for specified fields.

    Parameters
    ----------
    folder : str
        Path to the data folder.
    fs3D1 : list of str
        List of field names to extract from the binary file.
    nx, ny, nz : int
        Grid resolution in x, y, z directions.

    Returns
    -------
    initcond : dict
        Dictionary with keys from fs3D1 containing 3D NumPy arrays.
    """
    fn = os.path.join(folder, 'initial_conditions.dat')
    
    print("=" * 62)
    print("reading initial conditions file:")
    print(fn)
    print("=" * 62)

    # Read binary file (double precision)
    with open(fn, 'rb') as f:
        tmp = np.fromfile(f, dtype=np.float64)

    len3D = nx * ny * nz
    displ = 0
    initcond = {}

    for key in fs3D1:
        data = tmp[displ : displ + len3D]
        initcond[key] = data.reshape((nx, ny, nz), order='F')  # Fortran-style reshape
        displ += len3D

    return initcond


#%%Load initial conditions

nx = 64
ny = 64
nz = 64

lx = 2*np.pi
ly = 2*np.pi
lz = 1

dx = lx/nx
dy = ly/ny
dz = lz/nz

z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

# path = '/uufs/chpc.utah.edu/common/home/calaf-group3/Giulia_research3/miscellaneous/'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'

sim = 'test_mod_grad'

fs3D1 = ['u','v','w']

ic = load2decomp_initialcond(path+sim+'/', nx, ny, nz)
# ic_s = load2decomp_initialcond_SC(path+sim+'/', fs3D1, nx, ny, nz)

plot_initcond(ic, fs3D1, z_uvp, z_w)


#%%

def load_2Dfield(Nx,Ny,path):
    "Calling function to upload data from the Output_Rav_SC files"

    import numpy as np
    import struct


    print('Defining arrays')

    SurfTemp = np.zeros((Nx*Ny),'d')   #This is a temporal variable used as a bridge.
    SurfTemp2D = np.zeros((Nx,Ny),'d') #This is the final variable where the 2D field is stored.

    print('Opening the Surface Temperature file')


    FileName = '../input/surface_SC1.dat'
    f = open(path+FileName,"rb")

    #Next we read the data and we dump it all, as strings into a new variable.

    print('Done reading the binari data')

    data_raw = f.read(Nx*Ny*8)


    #Next we read the strings from above into double-type numbers, that are stored in data.
    SurfTemp[0:(Nx*Ny)] = np.array(struct.unpack('d'*(Nx*Ny), data_raw))
    SurfTemp2D[:,:] = np.reshape(SurfTemp,[Nx,Ny],'F')

    #Now we can delete the main data matrix uploaded initially to free some memory
    del data_raw
    f.close()



    return(SurfTemp2D)

#%%
import numpy as np
import matplotlib.pyplot as plt

nx = 256
ny = 256
nz = 256

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/TempPatch/'
sim = '800patch_1ms_aniso_fix_local_grn'
sfcT = load_2Dfield(nx, ny, path+sim+'/input/')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p1 = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(sfcT).T,cmap='hot_r')

axs.set_ylabel(r'$y/z_i$', fontsize=18)
fig.suptitle('Surface temperature', fontsize=18)
axs.set_xlabel(r'$x/z_i$', fontsize=18)
axs.text(0.1, 0.85,r"$\overline{T}$="+f"{np.mean(sfcT):.2f}K", transform=axs.transAxes, fontsize=18)
cbar1 = plt.colorbar(p1)
cbar1.set_label(label='T [K]',fontsize=18)
cbar1.ax.tick_params(labelsize=15)
axs.tick_params(axis='both',labelsize=15)

# plt.savefig(path_fig+'sfcT.png',dpi=300)
plt.show()






































