#!/usr/bin/env python3

"""
Module: io_averFile
The functions used to read and write LES aver_*.out. The aver_*.out files have
two formats:
1) The 3D simulated fields are averaged only in y direction then the data is
a function of x, z, and t;
2) The 3D simulated fields are averaged in both x and y directions then the
data is a function of z and t only. No matter which format, the data is in
ASCII format and the each line is for one averaged time period.
The first element of each line is the nomalized time at the end of the
average.
"""

################################################################################
### Histories:
### 2019/02/20 -- Bicheng Chen (chabby@ucla.edu) -- First create
################################################################################

### Import Modules ###
import numpy as np
import sys
import lespy as lp
import os.path
from glob import glob
### END Import Modules ###



### Functions ###
def read_averFile(path, nz, nx=-1, mode='xy', ts=None, te=None):
  """The basic read function for aver_*.out files"""
  ## path - The path of the aver_*.out file
  ## nz, nx - The dimensions in x and z direction
  ## mode - The averaged method. 'xy' is refered planar average; 'y' is refered
  ## >>to averaging in y direction only.
  ## ts - The start time (non-dimensional)
  ## te - The end time (non-dimensional)

  ## Load all the data in the file
  data = np.loadtxt(path, dtype=np.float64)

  ## Pick up the selected range
  time = data[:, 0]
  data = data[:, 1:]

  if not (ts==None and te==None):
    if ts==None:
      ls=0
    else:
      ls=np.squeeze(np.where(ts<=time))[0]

    if te==None:
      le=len(time)
    else:
      le=np.squeeze(np.where(te>=time))[-1]+1

    if le<=ls:
      sys.exit("io-io_averFile-read_averFile:\
        Error in selection of the time range {ts:f}-{te:f}. STOP!!!".
        format(ts=ts, te=te))

    if mode=='xy':
      time=time[ls: le]
    elif mode=='y':
      time=time[ls: le: nz]
    data = data[ls: le, :]

  else:
    if mode=='y':
      time=time[:: nz]


  ## Reshape the data
  nt = len(time)

  if mode=='y':
    data = data.reshape((nt, nz, nx))
  else:
    if mode != 'xy':
      sys.exit("io-io_averFile-read_averFile: Error in average mode. STOP!!!")

  return data, time



def loadLES_averFile(param, qtype, tss=None, tes=None, ftype='new'):
  """Load data from LES aver_*.out files"""
  ## param - The parameter variables of LES
  ## qtype - The quantity of LES, used to reconstruct the file name
  ## tss - the start time (unit:s)
  ## tes - the end time (unit:s)

  if type(param) is lp.lesParam.lesClass.param:

    if ftype == 'new':
      fn = param.path + '/output/avg_{:s}.out'.format(qtype)
    elif ftype == 'old':
      fn = param.path + '/output/aver_{:s}.out'.format(qtype)

    if os.path.isfile(fn):
      #-- Determine the read mode
      if 'average_dim_num' in param.raw['output_control']:
        nd_avg = param.raw['output_control']['average_dim_num']
      else:
        nd_avg = 2 # default average dimension in LES

      if nd_avg == 1:
        mode = 'y'
      elif nd_avg == 2:
        mode = 'xy'
      else:
        sys.exit("io-io_averFile-loadLES_averFile:\
            The LES average_dim_num has abnormal value. STOP!!!")

      #-- Determine the line range
      tscal = param.domain.zi / param.flow.uScal

      if tss==None:
        ts = None
      else:
        ts = tss / tscal

      if tes==None:
        te=None
      else:
        te=tes / tscal

      #-- Read the data from the file
      data, time = read_averFile(fn, param.domain.nz, param.domain.nx, mode,
        ts, te)

      return data, time

    else:
      sys.exit("io-io_averFile-loadLES_averFile:\
        The file {fn:s} doesn't exist. STOP!!!".format(fn=fn))

  else:
    sys.exit("io-io_averFile-loadLES_averFile:\
      The input argument is not the param type. STOP!!!")



def loadallLES_averFile(param, tss=None, tes=None):
  """Load all data from LES aver_*.out files"""
  ## param - The parameter variables of LES
  ## tss - the start time (unit:s)
  ## tes - the end time (unit:s)

  if type(param) is lp.lesParam.lesClass.param:
    #- get list of files
    fmt_fn = param.path + '/output/aver_*.out'
    list_fn = sorted(glob(fmt_fn))

    #- get the quantity name of LES
    data_all = dict()
    for fn in list_fn:
      qtype = os.path.basename(fn).split('aver_')[-1].split('.')[0]
      data_all.update({qtype: []})

    #- load each quantity from LED aver_*.out file
    for qtype in data_all.keys():
      data_all[qtype], time = loadLES_averFile(param, qtype, tss=tss, tes=tes)
    data_all.update(time=time)

    return data_all



### END Functions ###
