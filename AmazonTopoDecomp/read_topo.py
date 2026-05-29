import numpy as np
import matplotlib.pyplot as plt
import dted as dd
from pathlib import Path
from heapq import nlargest
import scipy as sp
from scipy.io import loadmat
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.io import savemat

def Lorentz_Curve(ftopo):
  spec_topo = np.abs(ftopo) ** 2
  top = np.max(spec_topo)
  mask = spec_topo==top
  spec_topo[mask] = 0.0
  total_variance = np.sum(spec_topo) / 2
  sz = np.shape(ftopo)
  nx = sz[0];
  ny = sz[1];
  total_modes = int(np.floor(nx * ny / 2))
  variance_acc = np.zeros(total_modes)
  variance_individual = np.zeros(total_modes)
  percent_modes = np.zeros(total_modes)
  it = 1
  valmax  = nlargest(np.size(spec_topo),spec_topo.flatten())
  for i in range(1, total_modes):
    #valmax = nlargest(2 * i, spec_topo.flatten())
    valmaxtmp = valmax[:2*i]
    valmaxit = valmaxtmp[::2]
    variance_acc[i] = np.sum(valmaxit)
    variance_individual[i] = np.sum(valmax[2 * i - 1])
    percent_modes[i] = it / total_modes * 100.0
    # ith_mode, sum_modes = generate_mode(valmax,spec_topo,i)
    it += 1

  variance_acc /= total_variance
  variance_acc *= 100.0

  dvar = np.gradient(variance_acc)
  ddvar = np.gradient(dvar)

  variance_curvature = abs(ddvar) / (1 + abs(dvar) ** 2) ** (3 / 2)
  var_curve_grad = np.gradient(variance_curvature)

  # Choose curvature based off of continuous derivative of the curvature
  var_curve_adj = np.array([])
  max_curve_it = 0
  max_curve_val = 0.0
  box = 4
  for i in range(0, int(np.size(var_curve_grad) / box) - box, box):
    values = var_curve_grad[i:i + box]
    sup = np.max(values)
    infm = np.min(values)
    if sup - infm < .00002:  # .000075 for normal domain # this value may need to be adjusted
      # Maybe choose this number based off the variance of the signal
      var_curve_adj = np.append(var_curve_adj, values)
      if np.max(values) > max_curve_val:
        max_curve_it = np.argmax(values) + i
        max_curve_val = values[np.argmax(values)]

  curve_max = nlargest(2, variance_curvature)

  optimal_index = np.argmax(variance_curvature)
  optimal_variance = variance_acc[max_curve_it]
  optimal_mode = percent_modes[max_curve_it]

  # f, ax = plt.subplots()
  # ax.plot(variance_acc, percent_modes)
  # ax.plot(optimal_variance, optimal_mode, 'ro')
  # ax.set_ylabel('Percent Coefficients Represented')
  # ax.set_xlabel('Topographic Variance')

  Lor_plt = dict()
  Lor_plt['var_acc'] = variance_acc
  Lor_plt['perc_modes'] = percent_modes
  Lor_plt['opt_var'] = optimal_variance
  Lor_plt['optimal_mode'] = optimal_mode

  plt.figure()
  #plt.semilogy(variance_individual)
  plt.semilogy(variance_individual/total_variance, 'ro')
  plt.semilogy(max_curve_it, variance_individual[max_curve_it]/total_variance, color = 'k', marker = '*', markersize = 10 )
  plt.semilogy
  plt.xlabel('Mode Number')
  plt.ylabel('percent variance contributed')

  return max_curve_it,Lor_plt

def generate_mode(valmax, nmax, spec_topo, ftopo):
  ftopo_acc = np.zeros(spec_topo.shape, dtype=np.complex64)
  ftopo_sin = np.zeros(spec_topo.shape, dtype=np.complex64)
  for inmax in range(np.size(valmax)):
    ind = list(np.where(spec_topo == valmax[inmax]))
    # print(ind)
    if (len(ind[0]) > 1):
      ind[0] = np.array([ind[0][np.mod(inmax, 2)], ])
      ind[1] = np.array([ind[1][np.mod(inmax, 2)], ])

    ftopo_acc[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
    if inmax >= 2 * nmax - 2:
      ftopo_sin[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]

  # topo_acc = np.fft.ifft2(ftopo_acc, norm="ortho")
  # topo_acc = topo_acc.real
  # topo_acc -= topo_acc.min()
  # topo_sin = np.fft.ifft2(ftopo_sin, norm="ortho")
  # topo_sin = topo_sin.real
  # topo_sin -= topo_sin.min()
  return ftopo_sin, ftopo_acc
#--------------------------------------------------------------------
#Start of User Defined section
#Load OG Perdigao and rotate and apply lorentz curve to it for matlab
#matlab will pre for LES
z = loadmat('/uufs/chpc.utah.edu/common/home/u0851921/Codes/matlab_processing_code/perd.mat')
#
# Key to correct roation is to roate the full domain and then narrow down to matirx of interest
zrot = rotate(z['zinterp'],6.5, reshape= False) # 5-6 degrees

plt.figure()
plt.pcolormesh(np.transpose(zrot[374:629,376:503])) #386:513

#Choose desired domain
ele_org = zrot[374:630,391:519] #374:630,376:504
sz = np.shape(ele_org)
nx = sz[0]-1
ny = sz[1]-1

npd = 10

# Smooth and make it period at the boundaries
ele = np.zeros(ele_org.shape)
ele_s = gaussian_filter(ele_org, sigma=npd/3, mode='wrap')
for ind in range(npd-1):
  ele[ind: nx-ind+1, ind] =\
    ele_org[ind: nx-ind+1, ind] * (ind+1)/npd\
    + ele_s[ind: nx-ind+1, ind] * (npd-ind-1)/npd
  ele[ind: nx-ind+1, ny-ind] =\
    ele_org[ind: nx-ind+1, ny-ind] * (ind+1)/npd\
    + ele_s[ind: nx-ind+1, ny-ind] * (npd-ind-1)/npd
  ele[ind, ind+1:ny-ind] =\
    ele_org[ind, ind+1:ny-ind] * (ind+1)/npd\
    + ele_s[ind, ind+1:ny-ind] * (npd-ind-1)/npd
  ele[nx-ind, ind+1:ny-ind] =\
    ele_org[nx-ind, ind+1:ny-ind] * (ind+1)/npd\
    + ele_s[nx-ind, ind+1:ny-ind] * (npd-ind-1)/npd
ele[npd-1: nx-npd+2, npd-1: ny-npd+2] =\
  ele_org[npd-1: nx-npd+2, npd-1: ny-npd+2]

ftopo = np.fft.fft2(ele, norm="ortho")
spec_topo = np.abs(ftopo)**2
optimal_mode, Lor_plt = Lorentz_Curve(ftopo)
nmax = optimal_mode+150
valmax = nlargest(2 * nmax, spec_topo.flatten())
#Generate ftopo Sin and ftopo Acc based off currecnt desired modes
ftopo_sin, ftopo_acc = generate_mode(valmax, nmax, spec_topo, ftopo)

topo_acc = np.fft.ifft2(ftopo_acc, norm="ortho")
topo_acc = topo_acc.real
topo_acc -= topo_acc.min()

topodict = {'topo_lorentz': topo_acc}
topodict['OG'] = ele
savemat('/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_simulation_results/validation/val_topo.mat', topodict)



pop =1