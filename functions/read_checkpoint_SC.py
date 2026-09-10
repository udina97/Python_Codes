#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 02:29:38 2023

@author: u1450851
"""



def read_checkpoint(path, stepouts, nx, ny, nz):
    
    import numpy as np
    import os
    
    fs3D1 = ['u', 'v', 'w', 'RHSx', 'RHSy', 'RHSz']
    fs3D2 = ['Cs_opt2']
    fs3D2_uvp = ['tke_sgs']
    fs3D2_debug = ['trace_L_sgs', 'denom_tke_sgs']
    fs3D2_lag = ['F_LM', 'F_MM', 'F_QN', 'F_NN']
    fs3D3 = ['SC','RHSsc', 'Ds_opt2', 'F_KX', 'F_XX', 'F_PY', 'F_YY']
    
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_wsc_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz + 1)
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs3D1)):
            # print(fs3D1[kk])
            chpt[fs3D1[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
        
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D2)):
            chpt[fs3D2[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * nz
        for kk in range(len(fs3D2_uvp)):
            chpt[fs3D2_uvp[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D2_debug)):
            chpt[fs3D2_debug[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D2_lag)):
            chpt[fs3D2_lag[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D3)):
            chpt[fs3D3[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

    
    return chpt

def read_checkpoint_aniso(path, stepouts, nx, ny, nz):
    
    import numpy as np
    import os
    
    fs3D1 = ['u', 'v', 'w', 'RHSx', 'RHSy', 'RHSz']
    fs3D2 = ['Cs_opt2']
    fs3D2_uvp = ['tke_sgs']
    fs3D2_debug = ['trace_L_sgs', 'denom_tke_sgs']
    fs3D2_lag = ['F_LM', 'F_MM', 'F_QN', 'F_NN']
    fs3D3 = ['SC', 'sc_new','wsc_new','RHSsc', 'Ds_opt2', 'F_KX', 'F_XX', 'F_PY', 'F_YY']
    fs3D4 = ['u_new','v_new','w_new','uu_new','vv_new','ww_new','uv_new','uw_new','vw_new']
    fs3D5 = ['txx','tyy','tzz','txy','txz','tyz']
    fs3D6 = ['S_uvp']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_wsc_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz + 1)
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs3D1)):
            # print(fs3D1[kk])
            chpt[fs3D1[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
        
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D2)):
            chpt[fs3D2[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * nz
        for kk in range(len(fs3D2_uvp)):
            chpt[fs3D2_uvp[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D2_debug)):
            chpt[fs3D2_debug[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D2_lag)):
            chpt[fs3D2_lag[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
            
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D3)):
            chpt[fs3D3[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
            
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D4)):
            chpt[fs3D4[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
            
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D5)):
            chpt[fs3D5[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        # Optional trailing diagnostics appended after the original anisotropy
        # checkpoint fields. This keeps older checkpoints readable.
        for kk in range(len(fs3D6)):
            if displ + len3D <= tmp.size:
                chpt[fs3D6[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
                displ += len3D
    
    return chpt


def read_checkpoint_sfc_L(path, stepouts, nx, ny):
    
    import numpy as np
    import os
    
    fs2D1 = ['sfcFLUX', 'sfcVAL', 'ustar', 'phi_m', 'psi_m', 'phi_h', 'psi_h','L']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_wsc_{stepouts[jt]:09}_sfc')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len2D = nx * ny 
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs2D1)):
            data2D = np.reshape(tmp[displ:displ + len2D], (nx, ny),order = 'F')
            displ += len2D
            chpt[fs2D1[kk]] = data2D
        
    
    return chpt

def read_checkpoint_sfc(path, stepouts, nx, ny):
    
    import numpy as np
    import os
    
    fs2D1 = ['sfcFLUX', 'sfcVAL', 'ustar', 'phi_m', 'psi_m', 'phi_h', 'psi_h']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_wsc_{stepouts[jt]:09}_sfc')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len2D = nx * ny 
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs2D1)):
            data2D = np.reshape(tmp[displ:displ + len2D], (nx, ny),order = 'F')
            displ += len2D
            chpt[fs2D1[kk]] = data2D
        
    
    return chpt

def read_checkpoint_mom(path, stepouts, nx, ny, nz):
    
    import numpy as np
    import os
    
    fs3D1 = ['u', 'v', 'w', 'RHSx', 'RHSy', 'RHSz']
    fs3D2 = ['Cs_opt2', 'k_sgs', 'L','G','num','c_eps', 'F_LM', 'F_MM', 'F_QN', 'F_NN']
    fs3D3 = ['yB']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_mom_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz + 1)
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs3D1)):
            # print(fs3D1[kk])
            chpt[fs3D1[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
        
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D2)):
            chpt[fs3D2[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
            
        len3D = nx * ny * nz
        for kk in range(len(fs3D3)):
            chpt[fs3D3[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz),order = 'F')
            displ += len3D
    
    return chpt
