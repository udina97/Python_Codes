#!/usr/bin/env python3

"""
Module: dmFun
The Function to generate the coordinates for the entire domain.

Class list
----------

Function list
-------------
"""
################################################################################
### Histories
### 2019/02/22 -- Bicheng Chen (chabby@ucla.edu) -- First create
### 2019/03/05 -- Bicheng Chen (chabby@ucla.edu) -- Improve the comments
################################################################################

### Import Modules ###
import numpy as np
import sys
from lespy.lesParam.lesClass import dmParam
### END Import Modules ###



### Functions ###
def coord1d(ndim, ldim, ctype='nonstaggered'):
  """
  Generate the 1D coordinate using coordinate dimension and length

  Parameters
  ----------
  ndim : integer
    Dimension of the coordinate.
  ldim : float
    Length of this dimension.
  ctype : string
    Coordinate type. Two options: staggered and nonstaggered.

  Returns
  -------
  coord : array
    The coordinate defined by its dimension and length.
  """

  step = ldim/ndim
  if ctype=='nonstaggered':
    coord = np.arange(step/2, ldim, step)
  elif ctype=='staggered':
    coord = np.arange(0, ldim, step)
  else:
    sys.exit("domain-coord1d:\
      The error in selection of coordinate type ctype. STOP!!!")
  return coord



def coord2d(nd1, l1, nd2, l2, ctype1='nonstaggered', ctype2='nonstaggered'):
  """
  Generate the 2D coordinates
  nd1, nd2 : dimension of the coordinate
  l1, l2 : length of this dimension
  ctype1, ctype2 : coordinate type. Two options: staggered and nonstaggered
  """

  step1 = l1/nd1
  step2 = l2/nd2

  if ctype1=='nonstaggered':
    start1 = step1/2
  elif ctype1=='staggered':
    start1 = 0
  else:
    sys.exit("domain-coord2d:\
      The error in selection of ctype1. STOP!!!")

  if ctype2=='nonstaggered':
    start2 = step2/2
  elif ctype2=='staggered':
    start2 = 0
  else:
    sys.exit("domain-coord2d:\
      The error in selection of ctype2. STOP!!!")

  coord2, coord1 = np.mgrid[start2:l2:step2, start1:l1:step1]

  return coord2, coord1



def xcoord(domain, norm=False, point=False):
  """Generate the 1D x coordinate"""
  ## domain - the domain parameter
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    return coord1d(domain.nx, domain.nx, ctype='staggered')
  else:
    if norm:
      return coord1d(domain.nx, domain.lx/domain.zi)
    else:
      return coord1d(domain.nx, domain.lx)



def ycoord(domain, norm=False, point=False):
  """Generate the 1D y coordinate"""
  ## domain - the domain parameter
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    return coord1d(domain.ny, domain.ny, ctype='staggered')
  else:
    if norm:
      return coord1d(domain.ny, domain.ly/domain.zi)
    else:
      return coord1d(domain.ny, domain.ly)



def zcoord(domain, ctype='nonstaggered', norm=False, point=False):
  """Generate the 1D y coordinate"""
  ## domain - the domain parameter
  ## ctype - coordinate type. Two options: staggered and nonstaggered
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    return coord1d(domain.nz, domain.nz, ctype='staggered')
  else:
    if norm:
      return coord1d(domain.nz, domain.lz/domain.zi, ctype)
    else:
      return coord1d(domain.nz, domain.lz, ctype)



def xycoord(domain, norm=False, point=False):
  """Generate the 2D xy coordinates"""
  ## domain - the domain parameter
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    y, x = coord2d(domain.nx, domain.nx, domain.ny, domain.ny,
      ctype1='staggered', ctype2='staggered')
  else:
    if norm:
      lx = domain.lx/domain.zi
      ly = domain.ly/domain.zi
    else:
      lx = domain.lx
      ly = domain.ly

    y, x = coord2d(domain.nx, lx, domain.ny, ly,
      ctype1='staggered', ctype2='staggered')

  return y, x



def xzcoord(domain, ztype='nonstaggered', norm=False, point=False):
  """Generate the 2D xz coordinates"""
  ## domain - the domain parameter
  ## ztype - coordinate type for z. Two options: staggered and nonstaggered
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    z, x = coord2d(domain.nx, domain.nx, domain.nz, domain.nz,
      ctype1='staggered', ctype2='staggered')
  else:
    if norm:
      lx = domain.lx/domain.zi
      lz = domain.lz/domain.zi
    else:
      lx = domain.lx
      lz = domain.lz

    z, x = coord2d(domain.nx, lx, domain.nz, lz, ctype1='staggered',
      ctype2=ztype)

  return z, x



def yzcoord(domain, ztype='nonstaggered', norm=False, point=False):
  """Generate the 2D xz coordinates"""
  ## domain - the domain parameter
  ## ztype - coordinate type for z. Two options: staggered and nonstaggered
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    z, y = coord2d(domain.nz, domain.nz, domain.ny, domain.ny,
      ctype1='staggered', ctype2='staggered')
  else:
    if norm:
      ly = domain.ly/domain.zi
      lz = domain.lz/domain.zi
    else:
      ly = domain.ly
      lz = domain.lz

    z, y = coord2d(domain.ny, ly, domain.nz, lz, ctype2=ztype)

  return z, y



def xyzcoord(domain, ztype='nonstaggered', norm=False, point=False):
  """Generate the 3D xyz coordinates"""
  ## domain - the domain parameter
  ## ztype - coordinate type for z. Two options: staggered and nonstaggered
  ## norm - normalized by LES length scale or not

  if type(domain) != dmParam:
    raise TypeError("Unsupported type. The type must be '{:s}'".format(
      dmParam.__name__))

  if point:
    lx = domain.nx
    ly = domain.ny
    lz = domain.nz
    startz = 0
    stepx = 1
    stepy = 1
    stepz = 1

  else:
    if norm:
      lx = domain.lx/domain.zi
      ly = domain.ly/domain.zi
      lz = domain.lz/domain.zi
    else:
      lx = domain.lx
      ly = domain.ly
      lz = domain.lz

    stepx = lx/domain.nx
    stepy = ly/domain.ny
    stepz = lz/domain.nz

    if ztype=='nonstaggered':
      startz=stepz/2
    elif ztype=='staggered':
      startz=0
    else:
      sys.exit("domain-xyzcoord:\
        The error in selection of ztype. STOP!!!")

  z, y, x = np.mgrid[startz:lz:stepz, 0:ly:stepy, 0:lx:stepx]

  return z, y, x



### END Functions ###
