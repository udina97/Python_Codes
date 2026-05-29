#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:42:17 2023

@author: benjamin
"""

#%% Plotting statistics and dignostics of LES outputs

def log_u_profile(Nz,Lz,path,data_name,u_scale):
    """
    Function to plot the logarithmic profile of u velocity.

    Parameters
    ----------
    Nz : Grid resolution z-dir
    Lz : Domain length z-dir
    path : path to the 3D momentum data

    Returns
    -------
    Logarithmic profile of u plot

    """
    import numpy as np
    import matplotlib
    import xarray as xr
    import os
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mticker
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data = xr.open_dataarray(data_name)
    
    dz = Lz/Nz
    z = np.arange(0,Nz)*dz + dz/2 #U is defined at the u,v,P nodes!!
    
    txz = data.data[:,:,:,20]
    txz_sfc = np.mean(txz[:,:,0],axis=(0,1))
    u = data.data[:,:,:,0]*u_scale/np.sqrt(txz_sfc)
    
    u_xyavg = np.mean(u,axis=(0,1))
    
    fig,ax = plt.subplots(figsize=(10,8),tight_layout=True)
    ax.plot(u_xyavg,z)
    ax.set_yscale('log')
    # ax.axhline(0,c='black')
    # ax.axvline(0,c='black')
    ax.set_title(r'Logarithmic profile of $<u>_{xy}$')
    ax.set_xlabel(r'U/u* [-]')
    ax.set_ylabel(r'z/zi [-]')
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    plt.show()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def check_mke(path,nRAV,Iter):
    """
    Plot the time series of Mean Kinetic Energy

    Parameters
    ----------
    path : path to Running diagnostics .txt file

    Returns
    -------
    Plot of mean kinetic energy vs time

    """
    
    import numpy as np
    import os
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mticker
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    diagnostics = np.genfromtxt('running_diagnostics.txt')
    
    meanMKE = np.mean(diagnostics[-nRAV*Iter:-1,3])
    
    fig,ax = plt.subplots(figsize=(8,6),tight_layout=True)
    ax.plot(diagnostics[:,0],diagnostics[:,3])
    ax.axvline(max(diagnostics[:,0]),color='grey', ls='--')
    ax.axvline(max(diagnostics[:,0])-nRAV*Iter,color='grey', ls='--')
    ax.axhline(meanMKE, color='grey', ls='--')
    ax.set_title('Check MKE for convergence')
    ax.set_xlabel(r'Iteration Step')
    ax.set_ylabel(r'MKE')
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    # ax.set_xticks(np.arange(0,max(diagnostics[:,0])+50000,100000))
    plt.show()
    
    return fig
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def check_ustar(path,nRAV,Iter):
    """
    Plot time series of ustar 

    Parameters
    ----------
    path : path to Running diagnostics .txt file

    Returns
    -------
    Plot of ustar vs time

    """
    
    import numpy as np
    import os
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mticker
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    diagnostics = np.genfromtxt('running_diagnostics.txt')
    
    mean_ustar = np.mean(diagnostics[-nRAV*Iter:-1,4])
    
    fig,ax = plt.subplots(figsize=(8,6),tight_layout=True)
    ax.plot(diagnostics[:,0],diagnostics[:,4])
    ax.axvline(max(diagnostics[:,0]),color='grey', ls='--')
    ax.axvline(max(diagnostics[:,0])-nRAV*Iter,color='grey', ls='--')
    ax.axhline(mean_ustar, color='grey', ls='--')
    ax.set_title('Check $u^{*}$ for convergence')
    ax.set_xlabel(r'Iteration Step')
    ax.set_ylabel(r'$u^{*}$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    # ax.set_xticks(np.arange(0,max(diagnostics[:,0])+50000,100000))
    plt.show()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def tau_wall_profile(Nx,Ny,Nz,Lz,path,data_name):
    """
    Plot the verical profile of Tau_wall = ReynoldsStress_xz + Tau_sgs_xz + Dispersive_xz

    Parameters
    ----------
    Nx : Grid resolution in x-dir
    Ny : Grid resolution in y-dir
    Nz : Grid resolution in z-dir
    Lz : Domain size z-dir
    path : path to Momentum data from LES output

    Returns
    -------
    Plot vertical profile of tau_wall

    """
    
    import numpy as np
    import matplotlib
    import os
    import xarray as xr
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mticker
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data = xr.open_dataarray(data_name)
    
    dz = Lz/Nz
    z = np.arange(0,Nz)*dz
    
    #Reynolds stress, Rxz at w nodes!!
    UW = data.data[:,:,:,14]
    U = data.data[:,:,:,0]
    W = data.data[:,:,:,2]
    U_w = np.zeros((Nx,Ny,Nz),'d',order='F') #interpolate U at the w nodes
    for k in range(1,Nz):
        U_w[:,:,k]=0.5*(U[:,:,k-1] + U[:,:,k])
    U_w[:,:,0] = 0 #velocity is 0 at the surface
    
    R_xz = -(UW - (U_w*W))
    
    #Sub-grid scale stress, Txz already at w nodes!!
    tau_sgs_xz = data.data[:,:,:,20]
   
    #Dispersive stress, Dxz at w nodes!!
    U_w_disp = np.zeros((Nx,Ny,Nz),'d',order='F') #compute dispersive velocity for U interpolated at w nodes!
    W_disp = np.zeros((Nx,Ny,Nz),'d',order='F')
    for k in range(0,Nz):
        U_w_disp[:,:,k] = np.squeeze(U_w[:,:,k])-np.mean(U_w[:,:,k],axis=(0,1))
        W_disp[:,:,k] = np.squeeze(W[:,:,k])-np.mean(W[:,:,k],axis=(0,1))
        
    D_xz = -np.mean(U_w_disp*W_disp,axis=(0,1))
    
    tau_wall = np.mean(R_xz, axis=(0,1)) + np.mean(tau_sgs_xz, axis=(0,1)) + D_xz #Twall defined at w nodes!!
    
    fig,ax = plt.subplots(figsize=(10,8),tight_layout=True)
    ax.plot(tau_wall,z,'k',label=r'$\tau_{wall}$')
    ax.plot(np.mean(R_xz, axis=(0,1)),z,'--g',label=r'$R_{XZ}$')
    ax.plot(np.mean(tau_sgs_xz, axis=(0,1)),z,'--r',label=r'$\tau_{SGS,XZ}$')
    ax.plot(D_xz,z,'--b',label=r'$D_{XZ}$')
    ax.axvline(1,ls='--',c='k')
    ax.set_xlabel(r'$Stress$')
    ax.set_ylabel(r'z [m]')
    ax.set_title(r'Vertical profile of $<\tau_{wall}>_{xy}$')
    ax.legend()
    
    plt.show()
    
    return fig
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_u_2D_xy(Nx,Ny,Lx,Ly,path,h,data_name):
    """
    Plot u velocity on a 2D plane at a fixed height z

    Parameters
    ----------
    path : path of Data_Momentum.nc
    h : Height at which to plot the 2D plane

    Returns
    -------
    Plot of a 2D XY plane for u component of velocity

    """
    
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data = xr.open_dataarray(data_name)
    
    u = data.data[:,:,:,0]
    
    dx = Lx/Nx
    dy = Ly/Ny
    
    x = np.arange(0,Nx)*dx
    y = np.arange(0,Ny)*dy
    
    fig,ax = plt.subplots()
    im1 = ax.pcolormesh(x,y,np.transpose(u[:,:,h]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'y [m]')
    ax.set_title(r'2D plot of u at level h = %i' %h)
    
    return fig
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_u_2D_xz(Nx,Nz,Lx,Lz,path,y,data_name):
    """
    Plot u velocity on a 2D plane at a fixed y

    Parameters
    ----------
    path : path of Data_Momentum.nc
    y : y location where to plot the 2D field

    Returns
    -------
    Plot of a 2D XZ plane for u component of velocity

    """
    
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data = xr.open_dataarray(data_name)
    
    u = data.data[:,:,:,0]
    
    dx = Lx/Nx
    dz = Lz/Nz
    
    x = np.arange(0,Nx)*dx
    z = np.arange(0,Nz)*dz
    
    fig,ax = plt.subplots()
    im1 = ax.pcolormesh(x,z,np.transpose(u[:,y,:]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'z [m]')
    ax.set_title(r'2D plot of u at y = %i' %y)
    
    return fig
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_ustar_2D(Nx,Ny,Lx,Ly,path,data_name):
    """
    Plot ustar on a 2D plane

    Parameters
    ----------
    path : path of Data_Momentum_2D.nc

    Returns
    -------
    Plot of a 2D XY plane for ustar

    """
    
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data2D = xr.open_dataarray(data_name)
    
    ustar = data2D.data[:,:,0]
    
    dx = Lx/Nx
    dy = Ly/Ny
    
    x = np.arange(0,Nx)*dx
    y = np.arange(0,Ny)*dy
    
    fig,ax = plt.subplots()
    im1 = ax.pcolormesh(x,y,np.transpose(ustar[:,:]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'y [m]')
    ax.set_title(r'2D plot of ustar')
    
    return fig
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_T_2D(Nx,Ny,Nz,Lx,Ly,Lz,h,path,data_name):
    """
    Plot T on a 2D plane

    Parameters
    ----------
    path : path of Data_Scalar.nc

    Returns
    -------
    Plot of a 2D XY plane for T

    """
    
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data2D = xr.open_dataarray(data_name)
    
    T = data2D.data[:,:,:,0]
    
    dx = Lx/Nx
    dy = Ly/Ny
    dz = Lz/Nz
    
    x = np.arange(0,Nx)*dx
    y = np.arange(0,Ny)*dy
    z = np.arange(0,Nz)*dz
    
    fig,ax = plt.subplots()
    im1 = ax.pcolormesh(x,y,np.transpose(T[:,:,h]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'y [m]')
    ax.set_title(r'2D plot of T')
    
    return fig
    
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_sfcval(Nx,Ny,Lx,Ly,path,data_name):
    """
    Plot T on a 2D plane

    Parameters
    ----------
    path : path of Data_Scalar.nc

    Returns
    -------
    Plot of a 2D XY plane for T

    """
    
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    data2D = xr.open_dataarray(data_name)
    
    sfcval = data2D.data[:,:,6]
    
    dx = Lx/Nx
    dy = Ly/Ny
    
    x = np.arange(0,Nx)*dx
    y = np.arange(0,Ny)*dy
    
    fig,ax = plt.subplots()
    im1 = ax.pcolormesh(x,y,np.transpose(sfcval[:,:]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'y [m]')
    ax.set_title(r'2D plot of Mav_sfcval')
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_test(Nx,Ny,Nz,Lx,Ly,path,h):
    """
    Compare dudz from dataTKE and data momentum

    Parameters
    ----------
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.
    Lx : TYPE
        DESCRIPTION.
    Ly : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    
    os.chdir(path)
    dataTKE = xr.open_dataarray('Data_TKEbudget3D.nc')
    dataMom = xr.open_dataarray('Data_Momentum.nc')
    
    pTKE = dataTKE.data[:,:,:,27]
    pMOM = dataMom.data[:,:,:,3]
    
    Txx_sgs = dataMom.data[:,:,:,16]    #components of the sub-grid scale stress tensor
    Tyy_sgs = dataMom.data[:,:,:,17]
    Tzz_sgs = dataMom.data[:,:,:,18]

    UU = dataMom.data[:,:,:,4]  
    VV = dataMom.data[:,:,:,5]
    WW = dataMom.data[:,:,:,6]

    WW_interp = np.zeros((Nx,Ny,Nz),'d',order='F')     #we interpolate the vertical velocity on the u,v nodes
    for k in range(0,Nz-1):
            WW_interp[:,:,k] = (WW[:,:,k] + WW[:,:,k+1])/2

    p0 = pMOM - (1/2)*(UU+VV+WW_interp)
    
    dx = Lx/Nx
    dy = Ly/Ny
    
    x = np.arange(0,Nx)*dx
    y = np.arange(0,Ny)*dy
    
    fig,ax = plt.subplots(1,2)
    im1 = ax[0].pcolormesh(x,y,np.transpose(pTKE[:,:,h]),shading='gouraud',cmap='coolwarm')
    im2 = ax[1].pcolormesh(x,y,np.transpose(p0[:,:,h]),shading='gouraud',cmap='coolwarm')
    cbar1 = fig.colorbar(im1, ax=ax[0], shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    cbar2 = fig.colorbar(im2, ax=ax[1], shrink=0.8,location='top',orientation='horizontal',pad=0.1)
    ax[0].set_xlabel(r'x [m]')
    ax[1].set_xlabel(r'x [m]')
    ax[0].set_ylabel(r'y [m]')
    ax[0].set_title(r'p TKE')
    ax[1].set_title(r'p MOM')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    