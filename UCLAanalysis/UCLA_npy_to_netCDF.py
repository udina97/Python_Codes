#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 07:19:18 2023

@author: benjamin
"""
def UCLA_npy_to_netCDF(path,Nx,Ny,Nz,data_flag):
    
    import numpy as np
    import xarray as xr
    import os
    
    if data_flag == 'true':
        
        txx = np.load(path + 'txx.npy')
        txy = np.load(path + 'txy.npy')
        txz = np.load(path + 'txz.npy')
        tyy = np.load(path + 'tyy.npy')
        tyz = np.load(path + 'tyz.npy')
        tzz = np.load(path + 'tzz.npy')
        u = np.load(path + 'u.npy')
        v = np.load(path + 'v.npy')
        w = np.load(path + 'w.npy')
        u2 = np.load(path + 'u2.npy')
        v2 = np.load(path + 'v2.npy')
        w2 = np.load(path + 'w2.npy')
        uv = np.load(path + 'uv.npy')
        uw = np.load(path + 'uw.npy')
        vw = np.load(path + 'vw.npy')
        
        data = xr.DataArray(np.ones(shape = (Nz,Ny,Nx,15),order='F'),\
                                    dims=('z','y','x','variable'), coords = {'variable':['txx','tyy',\
                                    'tzz','txy','txz','tyz','u','v','w','u2','v2','w2','uv','uw','vw']})
            
            
        data[:,:,:,0] = txx; data[:,:,:,1] = tyy; data[:,:,:,2] = tzz; data[:,:,:,3] = txy; 
        data[:,:,:,4] = txz; data[:,:,:,5] = tyz; data[:,:,:,6] = u; data[:,:,:,7] = v; 
        data[:,:,:,8] = w; data[:,:,:,9] = u2; data[:,:,:,10] = v2; data[:,:,:,11] = w2; 
        data[:,:,:,12] = uv; data[:,:,:,13] = uw; data[:,:,:,14] = vw; 
        
        os.chdir(path)
        
        data.to_netcdf('data.nc')
    else:
        data = xr.open_dataarray(path + 'data.nc')
    
    return data