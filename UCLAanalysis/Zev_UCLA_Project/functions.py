#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:48:29 2023

@author: u1450851
"""

def tavg_xy(var,intf,dz,lz):
    '''
    This function computes the x-y average of a variable following the terrain (I think)

    Parameters
    ----------
    var : 3D variable to be averaged
    intf : not sure
    dz : vertical grid spacing
    lz : vertical domain length

    Returns
    -------
    out : 1D vector where at each vertical level var has been averaged following topography

    '''
    import numpy as np
    
    if len(var.shape)==3:
        [nz,ny,nx] = var.shape
    else:
        [nx,nz] = var.shape
        ny = 1
    
    tmp_avgxy = np.zeros((nx,ny))
    levels = np.arange(np.min(intf),lz-np.max(intf),dz)
    out = np.zeros(len(levels))
    for ll in range(0,len(levels)):
        for i in range(0,nx):
            for j in range(0,ny):
                for k in range(2,nz-1):
                    h = k*dz - intf[i,j]
                    if (h<levels[ll]+dz/2 and h>levels[ll]-dz/2):
                        ch = k*dz - (levels[ll] + intf[i,j])
                        if ch<0:
                            w1 = (dz - abs(ch))/dz
                            w2 = 1 - w1
                            var_tmp = var[i,j,k]*w1 + var[i,j,k+1]*w2
                        else:
                            w1 = (dz - abs(ch))/dz
                            w2 = 1 - w1
                            var_tmp = var[i,j,k]*w1 + var[i,j,k-1]*w2
                        tmp_avgxy[i,j] = var_tmp
                        break
        out[ll] = sum(sum(tmp_avgxy))/(nx*ny)
        tmp_avgxy = np.zeros((nx,ny))
    return out

#-----------------------------------------------------------------------------------------------------------------------

def terrainfollowing_3D(var):
    '''
    This function takes in a 3D field and return the same 3D field where the tpography has been removed

    Parameters
    ----------
    var : 3D variable
    iintf : not sure

    Returns
    -------
    var_t

    '''
    import numpy as np
    import math
    import copy
    
    sz = var.shape
    nx = sz[2]
    ny = sz[1]
    nz = sz[0]
    lol = copy.deepcopy(var)
    lol[np.isnan(lol)] = 999
    var_t = np.full((nz,ny,nx),np.nan)
    
    for i in range(nx):
        for j in range(ny):
            non_999_indices = np.where(lol[:, j, i] != 999)[0]
            if len(non_999_indices) > 0:
                var_t[:len(non_999_indices), j, i] = lol[non_999_indices, j, i]
    return var_t

#-----------------------------------------------------------------------------------------------------------------------

def terrainfollowing_2D(var):
    '''
    This function takes in a 2D field and return the same 2D field where the tpography has been removed

    Parameters
    ----------
    var : 2D variable

    Returns
    -------
    var_t

    '''
    import numpy as np
    import copy
    import math
    
    lol = copy.deepcopy(var)
    sz = lol.shape
    nx = sz[0]
    nz = sz[1]
    
    lol[np.isnan(lol)] = 999
    var_t = np.full((nx,nz),np.nan)
    
    for i in range(nx):
        non_999_indices = np.where(lol[i, :] != 999)[0]
        if len(non_999_indices) > 0:
            var_t[i, :len(non_999_indices)] = lol[i, non_999_indices]
                
    
    return var_t

#-----------------------------------------------------------------------------------------------------------------------

def read_out_files(path_in,path_out,variables,nx,nz,nt,condition):
    '''
    This function reads and saves the .out files from the UCLAdata into a structure

    Parameters
    ----------
    path_in : path to the output folder where the .out files are stored
    path_out : path where to save the structure
    variables : list of names for all the variables

    Returns
    -------
    None.

    '''
    
    import numpy as np
    import pickle
    
    if condition==True:
    
        data_strc = dict()
        
        for var in range(len(variables)):
            
            print(f'Reading file: {variables[var]}')
    
            f = open(path_in+'avg_'+variables[var]+'.out','r')
            data = f.readlines()
            f.close()
    
            array = np.zeros((nx,nz,nt),'d',order='F')
    
            for t in range(0,nt):
                for k in range(0,nz):
                    counter = k + t*nz
                    data_split = data[counter].split()
                    for i in range(1,nx+1):
                        array[i-1,k,t] = float(data_split[i])
                
            data_strc[variables[var]] = array 
            
        print('Saving and loading the data dictionary.')
        
        file = open(path_out+'2D_out_data.pkl','wb')
        pickle.dump(data_strc,file)
        file.close()
        
    else:
        
        print('Loading the data dictionary.')
        
        file = open(path_out+'2D_out_data.pkl','rb')
        data_strc = pickle.load(file)
        file.close()
        
    
    return data_strc

#----------------------------------------------------------------------------------------------------------------------

def UCLA_npy_to_netCDF(path_in,path_out,Nx,Ny,Nz,data_flag):
    
    import numpy as np
    import xarray as xr
    import os
    
    if data_flag == 'true':
        
        txx = np.load(path_in + 'txx.npy')
        txy = np.load(path_in + 'txy.npy')
        txz = np.load(path_in + 'txz.npy')
        tyy = np.load(path_in + 'tyy.npy')
        tyz = np.load(path_in + 'tyz.npy')
        tzz = np.load(path_in + 'tzz.npy')
        u = np.load(path_in + 'u.npy')
        v = np.load(path_in + 'v.npy')
        w = np.load(path_in + 'w.npy')
        u2 = np.load(path_in + 'u2.npy')
        v2 = np.load(path_in + 'v2.npy')
        w2 = np.load(path_in + 'w2.npy')
        uv = np.load(path_in + 'uv.npy')
        uw = np.load(path_in + 'uw.npy')
        vw = np.load(path_in + 'vw.npy')
        
        data = xr.DataArray(np.ones(shape = (Nz,Ny,Nx,15),order='F'),\
                                    dims=('z','y','x','variable'), coords = {'variable':['txx','tyy',\
                                    'tzz','txy','txz','tyz','u','v','w','u2','v2','w2','uv','uw','vw']})
            
            
        data[:,:,:,0] = txx; data[:,:,:,1] = tyy; data[:,:,:,2] = tzz; data[:,:,:,3] = txy; 
        data[:,:,:,4] = txz; data[:,:,:,5] = tyz; data[:,:,:,6] = u; data[:,:,:,7] = v; 
        data[:,:,:,8] = w; data[:,:,:,9] = u2; data[:,:,:,10] = v2; data[:,:,:,11] = w2; 
        data[:,:,:,12] = uv; data[:,:,:,13] = uw; data[:,:,:,14] = vw; 
        
        os.chdir(path_out)
        
        data.to_netcdf('data.nc')
    else:
        data = xr.open_dataarray(path_out + 'data.nc')
    
    return data

#----------------------------------------------------------------------------------------------------------------------------------------------

def build_phi(ipath, nx, ny, nz, mpiProc):
    import numpy as np
    # Initialize the phi tensor
    phi = np.zeros((nx, ny, nz), dtype=np.float64)

    for i in range(mpiProc):
        # Calculate the number of layers per MPI node
        nzi = nz // mpiProc

        # Open and read the binary file
        with open(f'{ipath}phi.c{i}', 'rb') as fid:
            data = np.fromfile(fid, dtype=np.float64, count=nx * ny * nzi)

        # Reshape and copy data to the global variable
        cvar = np.reshape(data, (nx, ny, nzi), order='F')
        phi[:, :, nzi * i:nzi * (i + 1)] = cvar

    return phi

#----------------------------------------------------------------------------------------------------------------------------------------------

def build_intf(phi, dz):
    import numpy as np
    nx, ny, nz = phi.shape
    intf = np.zeros((nx, ny), dtype=phi.dtype)
    iintf = np.zeros((nx, ny), dtype=int)
    
    for j in range(ny):
        for i in range(nx):
            init = 0
            for k in range(3, nz - 1):
                if phi[i, j, k] * phi[i, j, k + 1] <= 0 and init == 0:
                    intf[i, j] = (k - 1) * dz - phi[i, j, k-1]
                    iintf[i, j] = k
                    init = 1
            init = 0
            
    return intf, iintf

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_var(path, strvar, nx, ny, nzTot, nt, iintf, mpiProc, avgT=1):
    import numpy as np
    cvarGlob = np.zeros((nx, ny, nzTot, avgT), dtype=np.float64)
    
    nzi = nzTot // mpiProc
    
    for i in range(mpiProc):
        with open(f'{path}{strvar}.c{i}', 'rb') as fid:
            data = np.fromfile(fid, dtype=np.float64, count=nx * ny * nzi * nt)
            data = data[nx * ny * nzi * (nt - avgT):nx * ny * nzi * nt]

        cvar = np.reshape(data,(nx, ny, nzi, avgT),order='F')
        cvarGlob[:, :, (nzi * i):(nzi * (i + 1)), :] = cvar[:, :, 0:nzi, :]

    return cvarGlob

#------------------------------------------------------------------------------------------------------------------------------------------

def get_var_ts(path, strvar, nzTot, nt, mpiProc, avgT=1):
    import numpy as np
    cvarGlob = np.zeros((nzTot, avgT), dtype=np.float64)
    
    nzi = nzTot // mpiProc
    
    for i in range(mpiProc):
        with open(f'{path}{strvar}.c{i}', 'rb') as fid:
            data = np.fromfile(fid, dtype=np.float64, count=nzi * nt)
            data = data[nzi * (nt - avgT):nzi * nt]

        cvar = np.reshape(data,(nzi, avgT),order='F')
        cvarGlob[(nzi * i):(nzi * (i + 1)), :] = cvar[0:nzi, :]

    return cvarGlob

#---------------------------------------------------------------------------------------------------------------------------------------------

def get_snapvar(path, strvar, nx, ny, nzTot, iintf, mpiProc, jt=None):
    import numpy as np
    import os
    
    cvarGlob = np.zeros((nx, ny, nzTot), dtype=np.float64)
    
    nzi = nzTot // mpiProc
    
    for i in range(mpiProc):
        if jt is not None:
            filename = os.path.join(path, f"{strvar}_jt{jt:07d}.c{i}")
        else:
            filename = os.path.join(path, f"{strvar}.c{i}")
        
        with open(filename, 'rb') as fid:
            data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nzi)
            # data = data[nx * ny * nzi :nx * ny * nzi]

        cvar = np.reshape(data,(nx, ny, nzi),order='F')
        cvarGlob[:, :, (nzi * i):(nzi * (i + 1))] = cvar[:, :, 0:nzi]

    return cvarGlob

#---------------------------------------------------------------------------------------------------------------------------------------------

def get_2Dxz_slice(path, strvar, nx, nzTot, iintf, mpiProc, jt=None):
    import numpy as np
    import os
    
    cvarGlob = np.zeros((nx, nzTot), dtype=np.float64)
    
    nzi = nzTot // mpiProc
    
    for i in range(mpiProc):
        if jt is not None:
            filename = os.path.join(path, f"{strvar}_jt{jt:07d}.c{i}")
        else:
            filename = os.path.join(path, f"{strvar}.c{i}")
        
        with open(filename, 'rb') as fid:
            data = np.fromfile(fid, dtype=np.float32, count=nx * nzi)
            # data = data[nx * ny * nzi :nx * ny * nzi]

        cvar = np.reshape(data,(nx, nzi),order='F')
        cvarGlob[:, (nzi * i):(nzi * (i + 1))] = cvar[:, 0:nzi]

    return cvarGlob

#--------------------------------------------------------------------------------------------------------------------------------------------


def load_3d_UCLAdata(file_path):
    import os
    import numpy as np
    try:
        # Load the 3D data from the .npy file
        data = np.load(file_path).transpose(2, 1, 0)

        # Get dimensions (nx, ny, nz)
        nx, ny, nz = data.shape

        # Create a dictionary with the data using the filename as key
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data_dict = {
            file_name: data
        }

        return data_dict

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    
#--------------------------------------------------------------------------------------------------------------------------------------------

def load_2d_UCLAdata(file_path):
    import os
    import numpy as np
    try:
        # Load the 3D data from the .npy file
        data = np.load(file_path).transpose(2, 1, 0)

        # Get dimensions (nx, ny, nz)
        nx, ny, nz = data.shape

        # Create a dictionary with the data using the filename as key
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data_dict = {
            file_name: data
        }

        return data_dict

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    
#------------------------------------------------------------------------------------------------------------------------------------------

def load_UCLAnpy_files(directory_path):
    import os
    import numpy as np
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    data_dict = {}
    var = ['dissip.npy', 'Fcx.npy', 'Fcy.npy', 'Fcz.npy', 'uFcx.npy', 'vFcy.npy', 'wFcz.npy']

    for npy_file in npy_files:
        if npy_file in var:
            file_path = os.path.join(directory_path, npy_file)
            loaded_data = load_2d_UCLAdata(file_path)
        else:
            file_path = os.path.join(directory_path, npy_file)
            loaded_data = load_3d_UCLAdata(file_path)

        if loaded_data:
            data_dict.update(loaded_data)

    return data_dict






























