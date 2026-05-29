#!/usr/bin/env python3

"""
Module: lesClass
The classes of the lesParam module.

Class List
----------
param
dmParam
timeParam
flowParam
conParam
"""

################################################################################
### Histories
### 2018/11/24 -- Bicheng Chen (chabby@ucla.edu) -- First create.
### 2019/07/16 -- Bicheng Chen (chabby@ucla.edu) -- Make the nz without ghost
### and add nz_ghost with ghost node.
################################################################################



### Import Modules ###
import f90nml
import sys
import os
import errno
### END Import Modules ###



### Class Definition ###
class param:
  """
  Class for all LES setup parameters.

  Attributes
  ----------
  raw: f90nml namelist
    The raw parameters of LES.
  path: str
    The path of the LES.
  domain: dmParam
    The domain parameters.
  time: timeParam
    The time parameters.
  flow: flowParam
    The flow parameters.
  con: conParam
    The concentration parameters.


  Methods
  -------
  get_param(lespath)
    Get the les param from lespath.
  """

  def __init__(self, lespath=None, ghost=False):
    if (os.path.isdir(lespath)):
      self.get_param(lespath, ghost)
    elif lespath is None:
      self.raw = None
    else:
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), lespath)


  def get_param(self, lespath, ghost):
    """Get the parameters of the LES"""
    fn_param = lespath+'/param.nml'
    if (os.path.isfile(fn_param)):
      self.raw = f90nml.read(fn_param)
      self.path = lespath
      self.domain = dmParam(self.raw, ghost)
      self.time = timeParam(self.raw)
      self.flow = flowParam(self.raw)
      self.con = conParam(self.raw)
    else:
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fn_param)



class dmParam:
  """
  class for domain parameters

  Attributes
  ----------
  nx : int
    The dimension in x-direction.
  ny : int
    The dimension in y-direction.
  nz : int
    The dimension in z-direction, not including toppest Ghost node.
  nz_ghost : int
    The dimension in z-direction, including toppest Ghost node.
  ldx : int
    The dimension in x-direction, including padding for using FFT.
  lx : float
    The domain size in x-direction.
  ly : float
    The domain size in y-direction.
  lz : float
    The domain size in z-direction.
  zi : float
    The length scale used to normalized all length in LES.
  dx : float
    The step size in x-direction.
  dy : float
    The step size in y-direction.
  dz : float
    The step size in z-direction.

  Methods
  -------
  show()
    Show the domain parameters.
  """

  def __init__(self, raw, ghost):
    if type(raw) is f90nml.namelist.Namelist:
      self.nx = raw['domain_param']['nx']
      self.ny = raw['domain_param']['ny']
      self.nz = raw['domain_param']['nz_tot']
      if ghost:
        self.nz -= 1
      self.nz_ghost = self.nz + 1
      self.lx = raw['domain_param']['lx']
      self.ly = raw['domain_param']['ly']
      self.lz = raw['domain_param']['lz_tot']
      self.zi = raw['domain_param']['z_i']

      self.ldx = self.nx+2
      self.dx = self.lx/self.nx
      self.dy = self.ly/self.ny
      self.dz = self.lz/self.nz
    else:
      sys.exit("lesParam-dmParam:\
        The input argument is not the Fortran namelist type. STOP!!!")

  def show(self):
    print('#'*29, ' Domain Parameters ', '#' * 30)
    print("The grid numbers in 3D are ", self.nx, self.ny, self.nz)
    print("The grid sizes in 3D are ", self.dx, self.dy, self.dz, " m")
    print("The domain sizes in 3D are ", self.lx, self.ly, self.lz, " m")
    print('#'*27, ' END Domain Parameters ', '#'*28)



class timeParam:
  """class for time parameters"""
  ## nt - total time steps
  ## dt - time interval
  ## tt_avg - time step to average the LES quantities for aver_*.out files
  def __init__(self, raw):
    if type(raw) is f90nml.namelist.Namelist:
      self.nt = raw['time_param']['nsteps']
      self.dt = raw['time_param']['dt']
      self.tt_avg = raw['time_param']['p_count']
    else:
      sys.exit("lesParam-timeParam:\
        The input argument is not the param type. STOP!!!")

      

class flowParam:
  """class for flow parameters"""
  ## uScal - the velocity scale
  def __init__(self, raw):
    if type(raw) is f90nml.namelist.Namelist:
      if 'u_scale' in raw['flow_param'].keys():
        self.uScal = raw['flow_param']['u_scale']
      else:
        self.uScal = raw['flow_param']['u_star']
    else:
      sys.exit("lesParam-flowParam:\
        The input argument is not the param type. STOP!!!")



class conParam:
  """
  class for concentration parameters

  Attributes
  ----------
  key : str
    The namelist name of the concentration parameters

  Methods
  -------
  """

  key = 'con_param'

  def __init__(self, raw):
    if type(raw) is f90nml.namelist.Namelist:
      if self.key in raw.keys():
        if 'n_con' in raw[self.key].keys():
          self.ncon = raw[self.key]['n_con']

        if 'pcon_scale' in raw[self.key].keys():
          self.cScal = raw[self.key]['pcon_scale']

    else:
      sys.exit("lesParam-conParam:\
        The input argument is not the param type. STOP!!!")



### END Class Definition ###
