#!/usr/bin/env python3

"""
Module: instFile.py
The functions to read and write LES sbin files of instantaneous
fields. These quantities include velocity, pressure, temperature and
concentration. These files are written in a binary format.

Function List
-------------
load_uvw
save_uvw
load_theta
save_theta
load_press
save_press
load_con
save_con
read_3dFile
load_cross
__load_one__
__save_one__
"""

################################################################################
### 2019/02/25 -- Bicheng Chen (chabby@ucla.edu) -- First create
### 2019/03/04 -- Bicheng Chen (chabby@ucla.edu) -- Add the IO for concentration
################################################################################



### Import Modules ###
import numpy as np
import lespy as lp
import sys
from os.path import isfile
### END Import Modules ###



### Functions ###
def load_uvw(param, tt, mode='uvw', fmt_fn='uvw_jt{tt:08d}.sbin',
  dtype=np.float64):
  """
  Load the 3D velocity data.

  Parameters
  ----------
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the extracted data.
  mode : string
    The velocity components need to be loaded. The mode includes 'uvw', 'uv',
    'uw', 'vw', 'u', 'v', 'w'
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  u, v, w : array
    The velocity components read from file.
  """

  ## Initialize the read environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain

  ## Load the data
  if mode is 'uvw':
    data = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=3, dtype=dtype)
    u = data[0]
    v = data[1]
    w = data[2]
    return u, v, w

  elif mode is 'uv':
    data = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=2, dtype=dtype)
    u = data[0]
    v = data[1]
    return u, v

  elif mode is 'uw':
    u = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=1, dtype=dtype)[0]
    w = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=1, nq_skip=2,
      dtype=dtype)[0]
    return u, w

  elif mode is 'vw':
    data = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=2, nq_skip=1,
      dtype=dtype)
    v = data[0]
    w = data[1]
    return v, w

  elif mode is 'u':
    u = read_3dFile(fn, dm.nx, dm.ny, dm.nz, dtype=dtype)[0]
    return u

  elif mode is 'v':
    v = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_skip=1, dtype=dtype)[0]
    return v

  elif mode is 'w':
    w = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_skip=2, dtype=dtype)[0]
    return w

  else:
    sys.exit("io-io_instFile-load_uvw:\
      Error in mode ('all', 'uv', 'uw', 'vw', 'u', 'v' or 'w'). STOP!!!")



def save_uvw(u, v, w, param, tt, fmt_fn='uvw_jt{tt:08d}.sbin.new',
  dtype=np.float64):
  """
  Save the 3D velocity data as the LES sbin file.

  Parameters
  ----------
  u, v, w : float array
    The velocity components read from file.
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the data.
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  flag : bool
    Flag to indicate if the data has been writen successfully.
  """

  ## Initialize the write environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain

  ## Check the data shape and format
  if (u.shape is not [dm.nz, dm.ny, dm.nx]) or\
    (v.shape is not [dm.nz, dm.ny, dm.nx]) or\
    (w.shape is not [dm.nz, dm.ny, dm.nx]):
    sys.exit("io-io_instFile-__save_uvw__:\
      Error in velocity shape. STOP!!!")

  if type(u[0, 0, 0]) is not dtype or\
    type(v[0, 0, 0]) is not dtype or\
    type(w[0, 0, 0]) is not dtype:
    sys.exit("io-io_instFile-__save_uvw__:\
      Error in data type. STOP!!!")

  ## Save the data
  with open(fn, 'wb') as fp:
    u.tofile(fp)
    v.tofile(fp)
    w.tofile(fp)
  
  return True



def load_theta(param, tt, fmt_fn='theta_jt{tt:08d}.sbin', dtype=np.float64):
  """
  Load potential temperature.

  Parameters
  ----------
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the extracted data.
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  theta : array
    The potential temperature field loaded from the LES file.
  """

  theta = __load_one__(param, tt, fmt_fn, dtype=np.float64)

  return theta



def save_theta(theta, param, tt, fmt_fn='theta_jt{tt:08d}.sbin.new',
  dtype=np.float64):
  """
  Save the potential temperature.

  Parameters
  ----------
  theta : array
    The 3D data need to be saved.
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the data.
  fmt_fn : string
    The format of the destination file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  flag : Bool
    Flag to indicate if the data has been writen successfully.
  """

  flag = __save_one__(theta, param, tt, fmt_fn, dtype)

  return flag


def load_press(param, tt, fmt_fn='press_jt{tt:08d}.sbin', dtype=np.float64):
  """
  Load pressure field.

  Parameters
  ----------
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the extracted data.
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  press : array
    The pressure field loaded from the LES file.
  """

  press = __load_one__(param, tt, fmt_fn, dtype=dtype)

  return press



def save_press(press, param, tt, press_fn='press_jt{tt:08d}.sbin.new',
  dtype=np.float64):
  """
  Save the pressure.

  Parameters
  ----------
  press : array
    The 3D data need to be saved.
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the data.
  fmt_fn : string
    The format of the destination file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  flag : Bool
    Flag to indicate if the data has been writen successfully.
  """

  flag = __save_one__(press, param, tt, fmt_fn, dtype)

  return flag



def load_con(param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin',
  dtype=np.float64):
  """
  load_con(param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin',
    dtype=np.float64)

  Load the concentration data

  Parameters
  ----------
  param : param(lespy)
    The parameters of the LES.
  tt : int
    The time step for the extracted data.
  icon : int tuple, optional
    The indices of concentration bins which are going to be loaded.
    Start from zero.
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  con : array
    The concentration loaded from file.
  """

  ## Initialize the read environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain
  ncon = param.con.ncon

  ## Load the data
  if icon is None :
    con = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_read=ncon,
      dtype=dtype)[0]
  elif type(icon) is int :
    con = read_3dFile(fn, dm.nx, dm.ny, dm.nz, nq_skip=icon,
      dtype=dtype)[0]
  elif type(icon) is tuple or list :
    n_icon = len(icon)
    con = np.zeros(n_icon, dm.nz, dm.ny, dm.nx)
    for ind in n_icon:
      con[ind, :, :, :] = read_3dFile(fn, dm.nx, dm.ny, dm.nz,
        nq_skip=icon[ind], dtype=dtype)
  else :
    sys.exit("io-io_instFile-load_con:\
      Error in type of icon. STOP!!!")

  return con



def save_con(con, param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin.new',
  dtype=np.float64):
  """
  save_con(con, param, tt, icon=None, fmt_fn='pcon_jt{tt:08d}.sbin.new',
    dtype=np.float64)

  Save the concentration data as the LES sbin file.

  Parameters
  ----------
  con : array
    The concentration is going to save
  param : param(lespy)
    The parameters of the LES.
  tt : int
    The time step for the data.
  icon : int tuple, optional
    The indices of concentration bins which are going to be saved.
    Start from zero.
  fmt_fn : string
    The format of the destination file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  flag : bool
    Flag to indicate if the data has been writen successfully.
  """

  ## Initialize the read environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain
  ncon = param.con.ncon
  nd = nz * ny * nx
  if dtype==np.float64:
    nbytes = 8
  elif dtype==np.float32:
    nbytes = 4
  else:
    sys.exit("io-io_instFile-save_con:\
      Error in dtype (np.float64 or np.float32). STOP!!!")

  ## Load the data
  if icon is None :
    with open(fn, 'wb') as fp:
      con.tofile(fp)

  elif type(icon) is int :
    if isfile(fn) :
      mode = 'r+b'
    else :
      mode = 'wb'

    with open(fn, mode) as fp :
      fp.seek(nbytes * icon*nd)
      con.tofile(fp)

  elif type(icon) is tuple or list :
    if isfile(fn) :
      mode = 'r+b'
    else :
      mode = 'wb'

    n_icon = len(icon)
    with open(fn, mode) as fp :
      for ind in range(n_icon):
        fp.seek(nbytes * icon[ind]*nd)
        con[ind, :, :, :].tofile(fp)

  else :
    sys.exit("io-io_instFile-save_con:\
      Error in type of icon. STOP!!!")

  return True



def read_3dFile(fpath, nx, ny, nz, nq_read=1, nq_skip=0, dtype=np.float64):
  """
  read_3dFile(fpath, nx, ny, nz, nq_read=1, nq_skip=0, dtype=np.float64)

  Read the 3D instantaneous quantities from LES sbin files.

  Parameters
  ----------
  fpath : string
    The path of the data file.
  nx : integer
    The grid number in x direction.
  ny : integer
    The grid number in y direction.
  nz : integer
    The grid number in z direction. Typically, it should be the nz in
    domain class.
  nq_read : integer, optional
    The number of the quantities need to read in.
  nq_skip : integer, optional
    The number of the quantities need to skip at the beginning of the file.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  data : list and array
    The data read from the file. The structure is data[nq_read][nz, ny, nx].
  """

  ## Initialize the read environment
  nd = nz * ny * nx
  if dtype==np.float64:
    nbytes = 8
  elif dtype==np.float32:
    nbytes = 4
  else:
    raise Exception("io-io_instFile-read_3dFile:\
      Error in dtype (np.float64 or np.float32). STOP!!!")
  data = [[] for iq in range(nq_read)]
    
  ## Read the data
  with open(fpath, 'rb') as fp:
    fp.seek(nbytes * nq_skip*nd)
    for iq in range(nq_read):
      data[iq] = np.fromfile(fp, dtype=dtype, count=nd).reshape(
        (nz, ny, nx))

  if nq_read is 1:
    data = data[0]

  return data



def load_cross(fpath, ic, nx, ny, nz, mode, nfield_skip=0, dtype=np.float64):
  """
  Load the values at the cross-section from the LES sbin file.

  Format
  ------
  load_cross(fpath, ic, nx, ny, nz, mode, nfield_skip=0, dtype=np.float64):

  Parameters
  ----------
  fpath : string
    The path of the data file.
  ic : int
    The index of the cross-section. Start from 0.
  nx : int
    The grid number in x direction.
  ny : int
    The grid number in y direction.
  nz : int
    The grid number in z direction. Typically, it should be the nz in
    domain class.
  mode : str
    Determine which plane is going to be crop. The mode is selected from
    'xy', 'xz' and 'yz'.
  nfield_skip : int, opt
    The number of 3D variable fields are going to be skipped.
    The default value is 0.
  dtype : data-type, opt
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  cross : array
    The data at specific cross-section
  """

  ## Initialize the read environment
  nd = nz * ny * nx
  if dtype==np.float64 :
    nbytes = 8
  elif dtype==np.float32 :
    nbytes = 4
  else :
    sys.exit("io-io_instFile-load_cross:\
      Error in dtype (np.float64 or np.float32). STOP!!!")

  ## Read the data
  nbytes_skip = nbytes * nfield_skip*nd

  with open(fpath, 'rb') as fp :
    fp.seek(nbytes_skip)
    if mode is 'xy' :
      nbytes_skip = ic * nbytes * ny * nx
      fp.seek(nbytes_skip, 1)
      cross = np.fromfile(fp, np.float64, ny*nx).reshape(ny, nx)
    elif mode is 'xz' :
      cross = np.fromfile(fp, np.float64, nd).reshape(nz, ny, nx)[:, ic, :]
    elif mode is 'yz' :
      cross = np.fromfile(fp, np.float64, nd).reshape(nz, ny, nx)[:, :, ic]
    else :
      sys.exit("io-io_instFile-load_cross:\
        Error in mode ('xy', 'xz', and 'yz'). STOP!!!")
      data[iq] = np.fromfile(fp, dtype=dtype, count=nd).reshape(
        (nz, ny, nx))

  return cross



def __load_one__(param, tt, fmt_fn, dtype=np.float64):
  """
  Load one 3D field stored as the first field of the LES file.

  Parameters
  ----------
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the extracted data.
  fmt_fn : string
    The format of the source file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  data : array
    The quantity read from the LES file.
  """

  ## Initialize the read environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain

  ## Load the data
  data = read_3dFile(fn, dm.nx, dm.ny, dm.nz, dtype=dtype)
  
  return data



def __save_one__(data, param, tt, fmt_fn, dtype=np.float64):
  """
  Save one 3D field as a LES sbin file.

  Parameters
  ----------
  data : array
    The 3D data need to be saved.
  param : param(lespy)
    The parameters of the LES.
  tt : integer
    The time step for the data.
  fmt_fn : string
    The format of the destination file name.
  dtype : data-type, optional
    The data type of the LES file. Two options: float32 and float64.
    The default is the float64.

  Returns
  -------
  flag : Bool
    Flag to indicate if the data has been writen successfully.
  """

  ## Initialize the write environment
  fn = param.path + '/output/' + fmt_fn.format(tt=tt)
  dm = param.domain

  ## Check the data shape and format
  if data.shape is not [dm.nz, dm.ny, dm.nx]:
    sys.exit("io-io_instFile-__save_one__:\
      Error in data.shape. STOP!!!")

  if type(data[0, 0, 0]) is not dtype:
    sys.exit("io-io_instFile-__save_one__:\
      Error in data type. STOP!!!")

  ## Save the data
  with open(fn, 'wb') as fp:
    data.tofile(fp)
  
  return True



### END Functions ###
