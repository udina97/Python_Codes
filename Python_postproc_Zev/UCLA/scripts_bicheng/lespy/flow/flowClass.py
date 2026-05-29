#!/usr/bin/env python3

################################################################################
### Module: The classes of the LES flow data
### 2019/02/26 -- Bicheng Chen (chabby@ucla.edu) -- First create
################################################################################



### Import Modules ###
import numpy as np
import sys
import lespy.io.io_instFile as io_instFile
### END Import Modules ###



### Classes ###
class velocity:
  """
  Class of the vel variables.

  Attributes
  ----------
  u : float array
    The velocity component in x-direction.
  v : float array
    The velocity component in y-direction.
  w : float array
    The velocity component in z-direction.

  Methods
  -------
  read_vel(param, tt, fmt_fn='uvw_jt{tt:08d}.sbin', dtype=np.float64)
    Read the velocity components from the LES sbin file.
  write_vel(param, tt, fmt_fn='uvw_jt{tt:08d}.sbin.new', dtype=np.float64)
    Write the velocity components to the LES sbin file.
  """


  def __init__(self, param=None, tt=None):
    if param is None or tt is None:
      self.u = None
      self.v = None
      self.w = None
    else:
      self.read_vel(param, tt)


  def read_vel(self, param, tt, fmt_fn='uvw_jt{tt:08d}.sbin', dtype=np.float64):
    """
    Read the velocity from the LES sbin file.

    Parameters
    ----------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the data.
    fmt_fn : str, optional
      The format of the source file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    self.u, self.v, self.w = io_instFile.load_uvw(param, tt, fmt_fn=fmt_fn,
      dtype=dtype)


  def write_vel(self, param, tt, fmt_fn='uvw_jt{tt:08d}.sbin.new',
    dtype=np.float64):
    """
    Write the velocity to the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the file.
    fmt_fn : str, optional
      The format of the source file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    return io_instFile.save_uvw(self.u, self.v, self.w, param, ttt,
      fmt_fn=fmt_fn, dtype=dtype)



class pressure:
  """
  Class of the pressure variable.

  Attributes
  ----------
  p : float array
    The pressure field.

  Methods
  -------
  read_press(param, tt, fmt_fn='press_jt{tt:08d}.sbin', dtype=np.float64)
    Read the pressure field from the LES sbin file.
  write_press(param, tt, fmt_fn='press_jt{tt:08d}.sbin.new', dtype=np.float64)
    Write the pressure field to the LES sbin file.
  """


  def __init__(self, param=None, tt=None):
    if param is None or tt is None:
      self.p = None
    else:
      self.read_press(param, tt)


  def read_press(self, param, tt, fmt_fn='press_jt{tt:08d}.sbin', dtype=np.float64):
    """
    Read the velocity from the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    fmt_fn : str, optional
      The format of the source file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    self.p = io_instFile.load_press(param, tt, fmt_fn=fmt_fn, dtype=dtype)


  def write_press(self, param, tt, fmt_fn='press_jt{tt:08d}.sbin.new',
    dtype=np.float64):
    """
    Write the press to the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the file.
    fmt_fn : str, optional
      The format of the source file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    return io_instFile.save_press(self.press, param, tt, fmt_fn=fmt_fn,
      dtype=dtype)
### END Classes ###
