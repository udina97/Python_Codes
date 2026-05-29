#!/usr/bin/env python3

"""
Program: convert
Convert LES topography data.
"""

### Histories:
### 2019/05/11 -- Bicheng Chen (chabby@ucla.edu) -- First create



## Prerequisite Module
import numpy as np



## User-specified Variable
# File
path_in = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/amazon_canopy_hill05h'
path_out = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_hill05h_3D'
fmtfn = '{path:s}/input/topography.in'

# LES setup
nx = 320
ny = 160
nz = 265
nz_in = 265
nz_out = 264



## Initilize the program
nd_i = nx * ny * nz_in
nd_o = nx * ny * nz_out



## Convert the data
# Read the data
fn = fmtfn.format(path=path_in)
with open(fn, 'rb') as fid:
  fid.read(4)
  phitot_uv = np.fromfile(fid, dtype=np.float64, count=nd_i).reshape(
    (nz_in, ny, nx))
  norm_phitotuv = np.fromfile(fid, dtype=np.float64, count=3*nd_i).reshape(
    (nz_in, ny, nx, 3))
  phitot_w = np.fromfile(fid, dtype=np.float64, count=nd_i).reshape(
    (nz_in, ny, nx))
  norm_phitotw = np.fromfile(fid, dtype=np.float64, count=3*nd_i).reshape(
    (nz_in, ny, nx, 3))
  z_tpg = np.fromfile(fid, dtype=np.float64, count=nx*ny).reshape((ny, nx))

# Process data
phitot_uv = phitot_uv[:nz_out, :, :]
phitot_w = phitot_w[:nz_out, :, :]
norm_phitotuv = np.transpose(norm_phitotuv[:nz_out, :, :, :], (3, 0, 1, 2))
norm_phitotw = np.transpose(norm_phitotw[:nz_out, :, :, :], (3, 0, 1, 2))

# Write the data
fn = fmtfn.format(path=path_out)
with open(fn, 'wb') as fid:
  fid.write(phitot_uv)
  fid.write(norm_phitotuv.flatten())
  fid.write(phitot_w)
  fid.write(norm_phitotw.flatten())
  fid.write(z_tpg)
