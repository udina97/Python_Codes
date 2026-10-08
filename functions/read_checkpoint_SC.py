#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 02:29:38 2023

@author: u1450851
"""



def read_checkpoint(path, stepouts, nx, ny, nz, read_lagran=None):
    
    import numpy as np
    import os
    
    fs3D1 = ['u', 'v', 'w', 'RHSx', 'RHSy', 'RHSz']
    fs3D2 = ['Cs_opt2']
    fs3D2_uvp = ['k_sgs']
    fs3D2_lag = ['F_LM', 'F_MM', 'F_QN', 'F_NN']
    fs3D3 = ['SC','RHSsc', 'Ds_opt2']
    fs3D3_lag = ['F_KX', 'F_XX', 'F_PY', 'F_YY']
    
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_wsc_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz + 1)
        n_required = (len(fs3D1) + len(fs3D2) + len(fs3D3) + len(fs3D2_uvp)) * len3D #+ (len(fs3D2_uvp))*(nx*ny*nz)
        n_with_lagran = n_required + (len(fs3D2_lag) + len(fs3D3_lag)) * len3D
        if read_lagran is None:
            has_lagran = tmp.size == n_with_lagran
        else:
            has_lagran = read_lagran

        expected_sizes = [n_required, n_with_lagran]
        if tmp.size not in expected_sizes:
            raise ValueError(
                f"{fn} contains {tmp.size} float64 values, but MOM checkpoints "
                f"with nx={nx}, ny={ny}, nz={nz} should contain either "
                f"{n_required} values without Lagran_init fields or "
                f"{n_with_lagran} values with them."
            )
        if has_lagran and tmp.size != n_with_lagran:
            raise ValueError(
                f"{fn} does not contain the four Lagran_init fields "
                f"({', '.join(fs3D2_lag)})."
            )
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

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D2_uvp)):
            chpt[fs3D2_uvp[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        if has_lagran:
            for kk in range(len(fs3D2_lag)):
                chpt[fs3D2_lag[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
                displ += len3D

        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D3)):
            chpt[fs3D3[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D
        
        len3D = nx * ny * (nz + 1)
        if has_lagran:
            for kk in range(len(fs3D3_lag)):
                chpt[fs3D3_lag[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
                displ += len3D

    
    return chpt

def read_checkpoint_aniso(path, stepouts, nx, ny, nz):
    
    import numpy as np
    import os
    
    fs3D1 = ['xB', 'yB']
    fs3D2 = ['u_new','v_new','w_new','uu_new','vv_new','ww_new','uv_new','uw_new','vw_new']
    fs3D3 = ['S_uvp']
    fs3D4 = ['txx_new','tyy_new','tzz_new','txy_new','txz_new','tyz_new','tke_sgs_new']
    fs3D5 = ['tke_sgs','L_trace','denom']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_aniso_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz)
        displ = 0
        # Reshape data and save to structure
        for kk in range(len(fs3D1)):
            # print(fs3D1[kk])
            chpt[fs3D1[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz),order = 'F')
            displ += len3D
        
        len3D = nx * ny * (nz + 1)
        # Reshape data and save to structure
        for kk in range(len(fs3D2)):
            chpt[fs3D2[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D3)):
            chpt[fs3D3[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D4)):
            chpt[fs3D4[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        len3D = nx * ny * (nz + 1)
        for kk in range(len(fs3D5)):
            chpt[fs3D5[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
            displ += len3D

        # # Optional trailing diagnostics appended after the original anisotropy
        # # checkpoint fields. This keeps older checkpoints readable.
        # for kk in range(len(fs3D6)):
        #     if displ + len3D <= tmp.size:
        #         chpt[fs3D6[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
        #         displ += len3D
    
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

def read_checkpoint_mom(path, stepouts, nx, ny, nz, read_lagran=None):
    
    import numpy as np
    import os
    
    fs3D1 = ['u', 'v', 'w', 'RHSx', 'RHSy', 'RHSz']
    fs3D2 = ['Cs_opt2']
    fs3D_lag = ['F_LM', 'F_MM', 'F_QN', 'F_NN']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_mom_{stepouts[jt]:09}')
        fid = open(fn, 'rb')
        tmp = np.fromfile(fid, dtype=np.float64)
        fid.close()
        
        len3D = nx * ny * (nz + 1)
        n_required = (len(fs3D1) + len(fs3D2)) * len3D
        n_with_lagran = n_required + len(fs3D_lag) * len3D
        if read_lagran is None:
            has_lagran = tmp.size == n_with_lagran
        else:
            has_lagran = read_lagran

        expected_sizes = [n_required, n_with_lagran]
        if tmp.size not in expected_sizes:
            raise ValueError(
                f"{fn} contains {tmp.size} float64 values, but MOM checkpoints "
                f"with nx={nx}, ny={ny}, nz={nz} should contain either "
                f"{n_required} values without Lagran_init fields or "
                f"{n_with_lagran} values with them."
            )
        if has_lagran and tmp.size != n_with_lagran:
            raise ValueError(
                f"{fn} does not contain the four Lagran_init fields "
                f"({', '.join(fs3D_lag)})."
            )

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

        if has_lagran:
            for kk in range(len(fs3D_lag)):
                chpt[fs3D_lag[kk]] = np.reshape(tmp[displ:displ + len3D], (nx, ny, nz + 1),order = 'F')
                displ += len3D
    
    return chpt


def read_checkpoint_tke(path, stepouts, nx, ny, nz):
    
    import numpy as np
    import os
    
    fs3D1 = ['k_sgs_uvp', 'rhs_k_sgs', 'rhs_k_sgs_f']
    
    chpt = {}
    
    for jt in range(len(stepouts)):
        # Read 3D data file
        fn = os.path.join(path, f'checkpoint_tke_{stepouts[jt]:09}')
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
    
    return chpt
