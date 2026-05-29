#!/usr/bin/env python3

"""
Module: dmClass 
The domain classes.

Classes
-------
dmParam: Domain parameter.
immersedbdy: Immersed boundary.

"""

################################################################################
### Histories:
### 2019/03/14 -- Bicheng Chen (chabby@ucla.edu) -- First create
################################################################################



### Import Modules ###
import numpy as np
import sys
from ..lesParam.lesClass import dmParam
### END Import Modules ###



### Class Definition ###
class immersedbdy:
  """
  Class for immersed boundary.

  Attributes
  ----------
  phi_uv: 3D array
    The signed-distance function phi for uvp-node.
  phi_w: 3D array
    The signed-distance function phi for w-node.
  norm_phiuv: 4D array
    The gradient of signed-distance function phi for uvp-node.
  norm_phiw: 4D array
    The gradient of signed-distance function phi for w-node.
  z_phi0 : 2D array
    The vertical coordinate where signed-distance function phi is 0.

  Methods
  -------
  read(param, fmt_fn='input/topography.in', dtype=np.float64)
    Read the signed-distance function phi and the corresponding norm from
    the LES input.
  """

  def __init__(self, param=None):
    """ Initilize the immersedbdy class  """

    if param is None:
      self.phi_uv = None
      self.phi_w = None
      self.norm_phiuv = None
      self.norm_phiw = None
      self.z_phi0 = None
    else:
      self.read(param)


  def read(self, param, basename='topography.in', dtype=np.float64):
    """
    Read the signed-distance function phi and the corresponding norm from
    the LES input.

    Parameters
    ----------
    param: param(lespy)
      The parameters of the LES.
    basename: str, opt
      The base name of the source file.
    dtype: data-type, opt
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    ## Initialize the read environment
    fn = param.path + '/input/' + basename
    dm = param.domain
    nd_3d = dm.nz * dm.ny * dm.nx

    ## Read the data
    with open(fn, 'rb') as fp:
      self.phi_uv = np.fromfile(fp, dtype, count=nd_3d).reshape(
        dm.nz, dm.ny, dm.nx)
      self.norm_phiuv = np.fromfile(fp, dtype, count=3*nd_3d).reshape(
        3, dm.nz, dm.ny, dm.nx)
      self.phi_w = np.fromfile(fp, dtype, count=nd_3d).reshape(
        dm.nz, dm.ny, dm.nx)
      self.norm_phiw = np.fromfile(fp, dtype, count=3*nd_3d).reshape(
        3, dm.nz, dm.ny, dm.nx)
      self.z_phi0 = np.fromfile(fp, dtype, count=dm.ny*dm.nx).reshape(
        dm.ny, dm.nx)


  def get_slope(self, param):
    dm = param.domain



### END Class Definition ###
