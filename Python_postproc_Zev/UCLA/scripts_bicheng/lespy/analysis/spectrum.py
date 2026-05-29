#!/usr/bin/env python3

"""
Module: Spectrum 
The spectrum analysis for flow variables. The spectrum of a quantity is based on
Fast Fourier Transform (FFT).
The spectrum module key features include:
1. The high order (>=2) term has aliasing problem, so A padding and unpadding
method is applied to do the dealiasing.
2. Because all the LES quantities are real numbers, the functions of rfft and
irfft from numpy.fft are applied.
"""

################################################################################
### Histories:
### 2019/02/26 -- Bicheng Chen (chabby@ucla.edu) -- First create
################################################################################



### Import Modules ###
import numpy as np
import lespy as lp
import sys
### END Import Modules ###



### Functions ###
def getSpec_1d(data, faxis=-1, mean=False, maxis=-2):
  """
  Get the energy spectrum from 1d forward fft.

  Parameters
  ----------
  data : array
    The input data
  faxis : integer, optional
    Axis over which to compute the energy spectrum.
    The default axis is the last one
  mean : bool, optional
    If this is set to True, the energy spectrum is going to take average along
    maxis.
  maxis : integer or tuple of integers, optional
    The axes along which the means are computed.

  Returns
  -------
  espec : array
    The energy spectral density of the data.
  """

  ## Calcuate the spectrum
  fdata = np.fft.rfft(data, axis=faxis, norm='ortho')
  espec = (fdata*fdata.conjugate()).real
  
  ## Averaging
  if mean:
    espec = np.mean(espec, axis=maxis)

  return espec



def getWnum_1d(num, ds):
  """
  Get wave number of the 1D FFT.

  Parameters
  ----------
  num : integer
    Windown length.
  ds : float
    Sample spacing. It should be either dx, dy or dz for LES.

  Returns
  -------
  wn : float array
    Wave number.
  """

  wn = np.fft.rfftfreq(num, ds)
  
  return wn



def padding_1d(data, axis=-1, mode='3/2'):
  """
  Do the padding for the input data. The only mode right now is 3/2 rule.
  The goal of the padding/unpadding process is to de-alias the high-order (>1)
  term.

  Parameters
  ----------
  data : array
    The data need padding.
  axis : integer
    The axis need padding.
  mode : string
    The only padding/unpadding method is following 3/2 rule (or 2/3 rule).

  Returns
  -------
  data_pad : array
    The padding data. The dimension of the padding axis is 3/2 times of
    the original axis (floor(3/2*N)).
  """

  ## Initialize the padding parameters
  nd = data.shape[axis]
  nd_pad = int(np.floor(3/2*nd))
  ndf = int(np.ceil((nd+1)/2))
  ndf_pad = int(np.ceil((nd_pad+1)/2))
  width_pad = [[0, 0] for ind in range(len(data.shape))]
  width_pad[axis][-1] = ndf_pad - ndf

  ## Pad the data from nd to nd_pad
  fdata = np.fft.rfft(data, axis=axis, norm='ortho')
  fdata_pad = np.pad(fdata, width_pad, mode='constant')\
    * np.sqrt(nd_pad/nd)
  data_pad = np.fft.irfft(fdata_pad, n=nd_pad, axis=axis, norm='ortho')

  return data_pad



def unpadding_1d(data, axis=-1, mode='3/2'):
  """
  Do the unpadding for the input data. The only mode right now is 3/2 rule.
  The goal of the padding/unpadding process is to de-alias the high-order (>1)
  term.

  Parameters
  ----------
  data : array
    The data need unpadding.
  axis : integer
    The axis need unpadding.
  mode : string
    The only padding/unpadding method is following 3/2 rule (or 2/3 rule).

  Returns
  -------
  data_upd : array
    The unpadding data. The dimension of the unpadding axis is 2/3 times of
    the original axis (ceil(2/3*N)).
  """

  ## Initialize the uppadding parameters
  nd = data.shape[axis]
  nd_upd = int(np.ceil(2/3*nd))
  ndf = int(np.ceil((nd+1)/2))
  ndf_upd = int(np.ceil((nd_upd+1)/2))

  ## Unpad the data from nd to nd_upd
  fdata = np.fft.rfft(data, axis=axis, norm='ortho')
  fdata_upd = np.delete(fdata, np.s_[ndf_upd:], axis=axis) * np.sqrt(nd_upd/nd)
  data_upd = np.fft.irfft(fdata_upd, n=nd_upd, axis=axis, norm='ortho')

  return data_upd



def dealisProd_1d(data1, data2, axis=-1, mode='3/2'):
  """
  Do the production between two quantities with the de-aliasing.
  The de-alising only applied in one direction.

  Parameters:
  -----------
  data1 : array
    The first input data.
  data2 : array
    The second input data.
  axis : integer
    The axis to do the de-aliasing
  mode : string
    The only padding/unpadding method is following 3/2 rule (or 2/3 rule).

  Returns:
  --------
  prod : array
    The production of two data with dealising.
  """

  data1_pad = padding_1d(data1, axis=axis, mode=mode)
  data2_pad = padding_1d(data2, axis=axis, mode=mode)
  prod_pad = data1_pad * data2_pad
  prod = unpadding_1d(prod_pad, axis=axis, mode=mode)

  return prod



### END Functions ###
