#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:33:27 2023

@author: u0851921
"""
import numpy as np
import lespy as lp
import os.path
import time
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.colors as colors
from scipy.fft import fftn, ifftn
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange
from scipy.interpolate import interpn

################### Averaging Functions  ####################################


def tavg_xy(var, intf, dz, lz):
    print('here')
    # Get the dimensions of the input variable
    nz, ny, nx = var.shape

    # Define levels based on the minimum and maximum values of intf
    levels = np.linspace(np.min(intf), lz - np.max(intf), int((lz - np.min(intf))/dz))
    out = np.zeros_like(levels)

    # Compute an array for w1 and w2 computation
    k_arr = np.arange(2, nz-1)[:, None, None] * dz - intf[None, :, :]
    
    # Prepare masks for conditions
    mask = (levels[:, None, None, None] - dz / 2 < k_arr) & (k_arr < levels[:, None, None, None] + dz / 2)
    ch = k_arr - levels[:, None, None, None]
    mask_w = ch < 0

    # Compute w1 and w2
    w1 = (dz - np.abs(ch)) / dz
    w2 = 1.0 - w1

    # Compute var_tmp
    var_tmp = np.where(mask_w, var[2:nz-1] * w1 + var[3:] * w2, var[2:nz-1] * w1 + var[1:nz-2] * w2)

    # Assign the calculated value to the temporary average array
    tmp_avgxy = np.where(mask.any(axis=1), var_tmp.mean(axis=1), 0)

    # Calculate the average value for the current level
    out = tmp_avgxy.sum(axis=(-1, -2)) / (nx * ny)

    return out




def tavg_xy2(var, intf, dz, lz):
   
    # Get the dimensions of the input variable
    nz, ny, nx = var.shape

    # Define levels based on the minimum and maximum values of intf
    levels = np.arange(np.min(intf), lz - np.max(intf), dz)
    out = np.zeros_like(levels)

    # Compute an array for w1 and w2 computation
    k_arr = np.arange(2, nz-1)[:, None, None] * dz - intf[None, :, :]
    
    # Loop over levels
    for ll in range(len(levels)):
        # Initialize temporary average array inside the loop
        tmp_avgxy = np.zeros((nx, ny))

        # Prepare masks for conditions
        mask = (levels[ll] - dz / 2 < k_arr) & (k_arr < levels[ll] + dz / 2)
        ch = k_arr - levels[ll]
        mask_w = ch < 0

        # Compute w1 and w2
        w1 = (dz - np.abs(ch)) / dz
        w2 = 1.0 - w1

        # Compute var_tmp
        var_tmp = np.where(mask_w, var[2:nz-1] * w1 + var[3:] * w2, var[2:nz-1] * w1 + var[1:nz-2] * w2)

        # Assign the calculated value to the temporary average array
        tmp_avgxy = np.where(mask.any(axis=0), var_tmp[mask].mean(axis=0), 0)

        # Calculate the average value for the current level
        out[ll] = np.sum(tmp_avgxy) / (nx * ny)

    return out

# def tavg_xy(var, dz, lz, intf):
#     # Get the dimensions of the input variable
#     nz, ny, nx = var.shape

#     # Initialize temporary average array
#     tmp_avgxy = np.zeros((nx, ny))

#     # Define levels based on the minimum and maximum values of intf
#     levels = np.arange(np.min(intf), lz - np.max(intf), dz)
#     out = np.zeros(len(levels))

#     # Loop over levels
#     for ll in range(len(levels)):
#         # Loop over x and y dimensions
#         for i in range(nx):
#             for j in range(ny):
#                 # Loop over z dimension, starting from 2 and ending at nz-2
#                 for k in range(2, nz-1):
#                     h = k * dz - intf[i, j]
#                     # Check if h is within the range of levels[ll] +/- dz/2
#                     if levels[ll] - dz / 2 < h < levels[ll] + dz / 2:
#                         ch = k * dz - (levels[ll] + intf[i, j])
#                         if ch < 0.0:
#                             w1 = (dz - abs(ch)) / dz
#                             w2 = 1.0 - w1
#                             var_tmp = var[k, j, i] * w1 + var[k + 1, j, i] * w2
#                         else:
    #                         w1 = (dz - abs(ch)) / dz
    #                         w2 = 1.0 - w1
    #                         var_tmp = var[k, j, i] * w1 + var[k - 1, j, i] * w2

    #                     # Assign the calculated value to the temporary average array
    #                     tmp_avgxy[i, j] = var_tmp
    #                     break

    #     # Calculate the average value for the current level
    #     out[ll] = np.sum(tmp_avgxy) / (nx * ny)

    #     # Reset the temporary average array for the next iteration
    #     tmp_avgxy = np.zeros((nx, ny))

    # return out

def average_isocontours(variable, surface, dz, lz):
    # Check if the input variable is 3D and the surface is 2D
    if len(variable.shape) != 3:
        raise ValueError("The input variable must be 3D")
    if len(surface.shape) != 2:
        raise ValueError("The input surface must be 2D")
        
    print('hre')
    # Get the dimensions
    nz, ny, nx = variable.shape

    # Create a 3D mesh grid
    z_grid = np.linspace(0, lz, nz)
    y_grid = np.arange(ny)
    x_grid = np.arange(nx)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    # Shift the Z levels based on the surface topography
    Z_shifted = Z - surface.T[np.newaxis, :, :]

    # Initialize an array to store the average values
    average_values = np.zeros((nz,))

    # Calculate the average values along the iso-contours
    for k in range(nz):
        contour_level = k * dz
        contour_mask = np.logical_and(Z_shifted >= contour_level - dz / 2, Z_shifted < contour_level + dz / 2)
        contour_points = np.column_stack((Z[contour_mask], Y[contour_mask], X[contour_mask]))
        contour_values = interpn((z_grid, y_grid, x_grid), variable, contour_points, bounds_error=False, fill_value=None)
        average_values[k] = np.nanmean(contour_values)

    return average_values

def average_isocontours2(variable, surface, dz, lz):
    # Check if the input variable is 3D and the surface is 2D
    if len(variable.shape) != 3:
        raise ValueError("The input variable must be 3D")
    if len(surface.shape) != 2:
        raise ValueError("The input surface must be 2D")

    # Get the dimensions
    nz, ny, nx = variable.shape

    # Create a 3D mesh grid
    z_grid = np.linspace(0, lz, nz)
    y_grid = np.arange(ny)
    x_grid = np.arange(nx)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    # Shift the Z levels based on the surface topography
    Z_shifted = Z - surface.T[np.newaxis, :, :]

    # Initialize an array to store the average values
    average_values = np.zeros((nz,))

    # Calculate the average values along the iso-contours
    for k in range(nz):
        contour_level = k * dz
        contour_mask = np.logical_and(Z_shifted >= contour_level - dz / 2, Z_shifted < contour_level + dz / 2)
        contour_points = np.column_stack((Z[contour_mask], Y[contour_mask], X[contour_mask]))
        contour_values = interpn((z_grid, y_grid, x_grid), variable, contour_points, bounds_error=False, fill_value=None)
        average_values[k] = np.nanmean(contour_values)

    return average_values

def average_isocontours(variable, surface, dz, lz):
    # Check if the input variable is 3D and the surface is 2D
    if len(variable.shape) != 3:
        raise ValueError("The input variable must be 3D")
    if len(surface.shape) != 2:
        raise ValueError("The input surface must be 2D")

    # Get the dimensions
    nz, ny, nx = variable.shape

    # Create a 3D mesh grid
    z_grid = np.linspace(0, lz, nz)
    y_grid = np.arange(ny)
    x_grid = np.arange(nx)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    # Shift the Z levels based on the surface topography
    Z_shifted = Z - surface.T[np.newaxis, :, :]

    # Create a RegularGridInterpolator for the input variable
    interpolator = RegularGridInterpolator((z_grid, y_grid, x_grid), variable, bounds_error=False, fill_value=None)

    # Initialize an array to store the average values
    average_values = np.zeros((nz,))

    # Calculate the average values along the iso-contours
    for k in range(nz):
        contour_level = k * dz
        contour_mask = np.logical_and(Z_shifted >= contour_level - dz / 2, Z_shifted < contour_level + dz / 2)
        contour_points = np.column_stack((Z[contour_mask], Y[contour_mask], X[contour_mask]))
        contour_values = interpolator(contour_points)
        average_values[k] = np.nanmean(contour_values)

    return average_values

def average_isocontours_y(variable, surface, dz, lz):
 # Check if the input variable is 3D and the surface is 2D
    if len(variable.shape) != 3:
        raise ValueError("The input variable must be 3D")
    if len(surface.shape) != 2:
        raise ValueError("The input surface must be 2D")

    # Get the dimensions
    nz, ny, nx = variable.shape

    # Create a 3D mesh grid
    z_grid = np.linspace(0, lz, nz)
    y_grid = np.arange(ny)
    x_grid = np.arange(nx)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    # Shift the Z levels based on the surface topography (corrected broadcasting)
    Z_shifted = Z - surface.T[np.newaxis, :, :]

    # Create a RegularGridInterpolator for the input variable
    interpolator = RegularGridInterpolator((z_grid, y_grid, x_grid), variable, bounds_error=False, fill_value=None)

    # Initialize an array to store the average values
    average_values = np.zeros((nz, nx))

    # Calculate the average values along the iso-contours
    for k in range(nz):
        for i in range(nx):
            contour_level = k * dz
            contour_mask = np.logical_and(Z_shifted[:, :, i] >= contour_level - dz / 2, Z_shifted[:, :, i] < contour_level + dz / 2)
            contour_points = np.column_stack((Z[:, :, i][contour_mask], Y[:, :, i][contour_mask], X[:, :, i][contour_mask]))
            contour_values = interpolator(contour_points)
            average_values[k, i] = np.nanmean(contour_values)


    return average_values
################################## Build Intf and IIntf ########################
def build_intf(phi,dz):
    nz,ny,nx = np.shape(phi)
    intf = np.zeros((nx,ny))
    iintf = np.zeros((nx,ny))
    init = 0
    for j in range(0,ny):
        for i in range(0,nx):
            for k in range(0,nz-1):
                if phi[k,j,i]*phi[k+1,j,i] <= 0.0 and init == 0:
                    intf[i,j] = (k-1)*dz-phi[k,j,i]
                    iintf[i,j] = k
                    init = 1
            init = 0
    return intf, iintf

############################# Calculate Derivatives #############################

def dvardx_f(array, dx):
    # Check if the input array is 3D
    if len(array.shape) != 3:
        raise ValueError("The input array must be 3D")

    # Get the dimensions
    nz, ny, nx = array.shape

    # Create an array with the wavenumbers in the x-direction
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

    # Take the Fourier Transform of the input array
    array_fft = fftn(array)

    # Multiply the Fourier Transform by 1j * kx
    array_fft_derivative = 1j * kx[np.newaxis, np.newaxis, :] * array_fft

    # Take the inverse Fourier Transform to get the derivative in real space
    array_derivative = np.real(ifftn(array_fft_derivative))

    return array_derivative

def dvardy_f(array, dy):
    # Check if the input array is 3D
    if len(array.shape) != 3:
        raise ValueError("The input array must be 3D")

    # Get the dimensions
    nz, ny, nx = array.shape

    # Create an array with the wavenumbers in the y-direction
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi

    # Take the Fourier Transform of the input array
    array_fft = fftn(array)

    # Multiply the Fourier Transform by 1j * ky
    array_fft_derivative = 1j * ky[np.newaxis, :, np.newaxis] * array_fft

    # Take the inverse Fourier Transform to get the derivative in real space
    array_derivative = np.real(ifftn(array_fft_derivative))

    return array_derivative

def dvardz(array, dz):
    # Check if the input array is 3D
    if len(array.shape) != 3:
        raise ValueError("The input array must be 3D")

    # Get the dimensions
    nz, ny, nx = array.shape

    # Initialize the derivative array with the same shape as the input array
    array_derivative = np.zeros_like(array)

    # Apply the forward difference for the first plane (z=0)
    array_derivative[0] = (array[1] - array[0]) / dz

    # Apply the central difference for the interior planes (1 <= z <= nz-2)
    for z in range(1, nz - 1):
        array_derivative[z] = (array[z + 1] - array[z - 1]) / (2 * dz)

    # Apply the backward difference for the last plane (z=nz-1)
    array_derivative[-1] = (array[-1] - array[-2]) / dz

    return array_derivative

def dvardx(array, dx):
    # Check if the input array is 3D
    if len(array.shape) != 3:
        raise ValueError("The input array must be 3D")

    # Get the dimensions
    nz, ny, nx = array.shape

    # Initialize the derivative array with the same shape as the input array
    array_derivative = np.zeros_like(array)

    # Apply the forward difference for the first column (x=0)
    array_derivative[:, :, 0] = (array[:, :, 1] - array[:, :, 0]) / dx

    # Apply the central difference for the interior columns (1 <= x <= nx-2)
    for x in range(1, nx - 1):
        array_derivative[:, :, x] = (array[:, :, x + 1] - array[:, :, x - 1]) / (2 * dx)

    # Apply the backward difference for the last column (x=nx-1)
    array_derivative[:, :, -1] = (array[:, :, -1] - array[:, :, -2]) / dx

    return array_derivative















