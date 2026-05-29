#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:20:48 2025

@author: u1450851
"""

# Bicheng functions to interpolate to nodes and compute derivatives -----------------------------------------------------------------------

def get_dphidx(phi, wn):
    import numpy as np
    phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
    dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
    dphidx_c[-1, :, :] = 0
    return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
    import numpy as np
    phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
    dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
    dphidy_c[:, -1, :] = 0
    return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
    dphidz[:,:,-1] = dphidz[:,:,-2]
    return dphidz

def uvpnode2wnode(phi_c):
    import numpy as np
    phi_h = np.zeros(phi_c.shape)
    phi_h[:, :, 0] = 0
    phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
    return phi_h

def wnode2uvpnode(phi_h):
    import numpy as np
    phi_c = np.zeros(phi_h.shape)
    phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
    phi_c[:, :, -1] = phi_h[:, :, -1]
    return phi_c

# Compute the displacement height ------------------------------------------------------------------------------------------------------

def compute_d_twr(data, coord, dist, height, dz, zi, u_scale, LAD):
    import numpy as np
    from scipy.integrate import trapezoid
    npoints = coord.shape[0]
    d_dim = np.zeros(npoints)

    for i in range(npoints):
        ix, iy = coord[i]
        z_nan = dist[ix, iy, :]

        # Find first valid vertical index
        valid_start = np.argmax(z_nan > 0) - 5
        
        # Extract u and v from data array
        u = data.data[ix, iy, valid_start:, 0]
        v = data.data[ix, iy, valid_start:, 1]

        U = np.sqrt(u**2 + v**2)
        
        Z = np.linspace(1, height - 5, 16) * dz * zi - 0.5 * dz * zi

        # Make sure we have enough points
        max_len = min(height - 5, len(U), len(LAD))
        VAR = U[:max_len] * u_scale
        Y = (VAR[:max_len]**2) * 0.4 * np.array(LAD[:max_len])
        Z_use = Z[:max_len]

        d_dim[i] = trapezoid(Y * Z_use, Z_use) / trapezoid(Y, Z_use)

    return d_dim

# Find the coordinates of the towrs on Ben topography cases (Peaks or Valleys) -----------------------------------------------------------

def find_coordinates(elevation_map, N_twrs, loc='max'):
    """
    Finds the coordinates of local peaks or valleys in a topography map.

    Parameters
    ----------
    elevation_map : 2D numpy array
        Topography height map.
    N_twrs : int
        Number of towers to select.
    loc : str, optional
        'max' for peaks, 'min' for valleys. Default is 'max'.

    Returns
    -------
    coords : ndarray of shape (N_twrs, 2)
        Selected (i, j) coordinates.
    """
    import numpy as np

    arr = np.array(elevation_map)
    rows, cols = arr.shape
    
    if loc == 'flat':
        # Uniform random sampling of N_twrs coordinates
        all_indices = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1).reshape(-1, 2)
        selected = all_indices[np.random.choice(all_indices.shape[0], N_twrs, replace=False)]
        return selected

    # Prepare slices for neighbors (4-connectivity)
    up    = arr[:-2, 1:-1]
    down  = arr[2:, 1:-1]
    left  = arr[1:-1, :-2]
    right = arr[1:-1, 2:]
    center = arr[1:-1, 1:-1]

    if loc == 'max':
        # Local peak condition (center > all neighbors)
        peak_mask = (center >= up) & (center >= down) & (center >= left) & (center >= right)
        # Threshold to keep only the top 10% elevation
        threshold = 0.9 * np.max(arr)
        high_mask = center > threshold
        valid_mask = peak_mask & high_mask
    elif loc == 'min':
        # Local valley condition (low elevation below a certain threshold)
        threshold = 2 * np.min(arr)
        valid_mask = center <= threshold        
    else:
        raise ValueError("loc must be 'max' or 'min'")

    # Get indices of valid points and shift to original indexing
    indices = np.argwhere(valid_mask) + 1  # shift due to 1:-1 slicing

    if len(indices) < N_twrs:
        raise ValueError(f"Not enough {'peaks' if loc == 'max' else 'valleys'} found to select {N_twrs} towers.")

    # Randomly sample N_twrs from candidates
    selected = indices[np.random.choice(len(indices), N_twrs, replace=False)]

    return selected

# Compute the average over tower profiles for Giulia's data -------------------------------------------------------------------------------

def average_over_selected_coords(data, selected_coords):
    """
    Compute the average over the (nx, NY) dimensions for selected (i, j) coordinates.
    
    Parameters:
    -----------
    data : np.ndarray
        A 3D array of shape (nx, NY, nz)
    selected_coords : np.ndarray
        Array of shape (n_points, 2), where each row is an (i, j) coordinate
    
    Returns:
    --------
    avg_profile : np.ndarray
        1D array of length nz, the average over selected (nx, NY) points
    """
    import numpy as np
    # Extract the profiles at selected (i, j) locations
    profiles = np.array([data[i, j, :] for i, j in selected_coords])

    # Average over the selected points (axis 0)
    avg_profile = np.mean(profiles, axis=0)
    
    return avg_profile

# Build the phi function for the topography cases using the IBM --------------------------------------------------------------------------

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