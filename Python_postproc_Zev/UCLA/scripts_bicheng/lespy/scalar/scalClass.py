#!/usr/bin/env python3

"""
Module: scalClass 
The classes of the LES scalar data. The scalar data includes both temperature
and concentrations.
"""

################################################################################
### Histories:
### 2019/02/28 -- Bicheng Chen (chabby@ucla.edu) -- Abandon the old one and
###   rewrite.
################################################################################



### Import Modules ###
import numpy as np
import sys
import lespy.io.io_instFile as io_instFile
### END Import Modules ###



### Classes ###
class theta:
  """
  Class of the temperature variable.

  Attributes
  ----------
  theta : float array
    The temperature field

  Methods
  -------
  read(param, tt, fmt_fn='theta_jt{tt:08d}.sbin', dtype=np.float64)
    Read the temperature field from the LES sbin file.
  write(param, tt, fmt_fn='theta_jt{tt:08d}.sbin.new', dtype=np.float64)
    Write the temperature field to the LES sbin file.
  """

  def __init__(self, param=None, tt=None):
    if param is None or tt is None:
      self.theta = None
    else:
      self.read(param, tt)


  def read(self, param, tt, fmt_fn='theta_jt{tt:08d}.sbin', dtype=np.float64):
    """
    Read the temperature from the LES sbin file.

    Parameters
    ---------
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

    self.theta = io_instFile.__load_one__(param, tt, fmt_fn, dtype=dtype)


  def write(self, param, tt, fmt_fn='theta_jt{tt:08d}.sbin.new',
    dtype=np.float64):
    """
    Write the temperature to the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the data.
    fmt_fn : str, optional
      The format of the destination file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    flag = io_instFile.__save_one__(self.theta, param, tt, fmt_fn, dtype=dtype)
    return flag



class concentration:
  """
  Class of the concentration variable.

  Attributes
  ----------
  con : float array
    The concentration field

  Methods
  -------
  read(param, tt, fmt_fn='pcon_jt{tt:08d}.sbin', dtype=np.float64)
    Read the temperature field from the LES sbin file.
  write(param, tt, fmt_fn='pcon_jt{tt:08d}.sbin.new', dtype=np.float64)
    Write the temperature field to the LES sbin file.
  """

  def __init__(self, param=None, tt=None):
    if param is None or tt is None:
      self.con = None
    else:
      self.read(param, tt)


  def read(self, param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin',
    dtype=np.float64):
    """
    read(param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin', dtype=np.float64)

    Read the concentration from the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the data.
    icon : int tuple, optional
      The indices of concentration bins which are going to be read.
      Start from zero.
    fmt_fn : str, optional
      The format of the source file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    self.con = io_instFile.load_con(param, tt, icon, fmt_fn, dtype=dtype)


  def write(self, param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin.new',
    dtype=np.float64):
    """
    Write the temperature to the LES sbin file.

    Parameters
    ---------
    param : param(lespy)
      The parameters of the LES.
    tt : integer
      The time step of the data.
    fmt_fn : str, optional
      The format of the destination file name.
    dtype : data-type, optional 
      The data type of the LES file. Two options: float32 and float64.
      The default is the float64.
    """

    flag = io_instFile.save_con(self.con, param, tt, icon, fmt_fn, dtype=dtype)

    return flag



### END Classes ###
