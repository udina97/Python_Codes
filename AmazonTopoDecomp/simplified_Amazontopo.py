#!/usr/bin/ipython3

################################################################################
### Program: Reconstruct the topography using the Fast Fourier Transformation for
### topography.
### 2018/01/19 -- Bicheng Chen (chabby@ucla.edu) -- First created
################################################################################

### Import Modules ###
import sys
sys.path.append("../")
#import lespy as lp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import detrend
from heapq import nlargest
from matplotlib import patches as pat
from matplotlib.collections import PatchCollection
import os
from scipy.io import savemat
### END Import Modules ###

#%%

### Functions ###
def compute_spectra(topo, num_bins=67, group_waves = []):
  S = fold_topo(topo)
  N = np.shape(topo)[0]
  nf = int(N/2)
  bin_width = np.round(N / num_bins);
  binterval = np.zeros([num_bins, 2]);

  for k in range(3,num_bins):
    binterval[k, 0] =  k * bin_width-bin_width #2 + (k - 1) * bin_width;
    binterval[k, 1] =  (k * bin_width+bin_width)-bin_width #k * bin_width+1;

  binterval[0,:] = np.array([.75, 1.25])
  binterval[1, :] = np.array([1.25, 2.25])
  binterval[2, :] = np.array([2.25, 4.0])

  sumvals = 0
  its = 0
  S_mean = np.zeros([num_bins, 1])
  for q in range(num_bins):
    for i in range(nf):
      for j in range(nf):
          if np.sqrt(i ** 2 + j ** 2 ) >= binterval[q, 0] and np.sqrt(i ** 2 + j ** 2 ) < binterval[q, 1]:
            sumvals += S[i, j]
            its += 1
    if its > 0.0:
      S_mean[q] = sumvals / its
      sumvals = 0
    its = 0
    sumvals = 0

  E_mag = S_mean[1:]
  k_mean = np.mean(binterval, axis=1);
  k_mean = k_mean[1:]

  # E_mag = S_mean;
  # k_mean = np.mean(binterval, axis=1);
  # k_mean = k_mean;

  E_9_2 = lambda x: x ** (-9 / 2)

  color_list = ['green', 'blue', 'red', 'orange', 'black', 'purple','cyan', 'magenta','yellow']

  fig, ax = plt.subplots()
  plt.loglog(k_mean[:-5],E_mag[:-5], 'ko',  linewidth = 3, label = 'Spectra')
  plt.loglog(k_mean,E_9_2(k_mean), 'b--', linewidth = 3, label = '$-k^{9/2}$')
  plt.xlabel('$|(k_x,k_y)|$')
  plt.ylabel('$|(E_x,E_y)|$')
  if len(group_waves) > 0:
    for i in range(len(group_waves)):
      x_min = [group_waves[i][0],group_waves[i][0]]
      x_max = [group_waves[i][1],group_waves[i][1]]
      ys = [0,10]
      plt.plot(x_min,ys, linewidth = 1.25, color = color_list[i], label= 'Group '+ str(i+1) )
      plt.plot(x_max,ys, linewidth = 1.25, color = color_list[i] )
      anchor = (x_min[0],0.0)
      height = 10.0
      width = x_max[0]-x_min[0]
      K_fill = pat.Rectangle(anchor, width,height, color= color_list[i],alpha = .2, fill=True,linestyle= '-', linewidth= 2.25 )
      ax.add_patch(K_fill)

  if len(group_waves) > 0:
   # plt.legend(('Spectra','$-k^{9/2}$', 'Group 1','Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6'))
    plt.legend()
  else:
    plt.legend(('Spectra','$-k^{9/2}$'))

  ax.set_xlim([np.min(k_mean)-.25, np.max(k_mean)]) #-60
  ax.set_ylim([10**-9, np.max(E_mag)+.15])

  plt.show()

def Lorentz_Curve(ftopo):
  spec_topo = np.abs(ftopo) ** 2
  total_variance = np.sum(spec_topo) / 2
  sz = np.shape(ftopo)
  nx = sz[0];
  ny = sz[1];
  total_modes = int(np.floor(nx * ny / 2))
  variance_acc = np.zeros(total_modes)
  variance_individual = np.zeros(total_modes)
  percent_modes = np.zeros(total_modes)
  it = 1
  for i in range(1, total_modes):
    valmax = nlargest(2 * i, spec_topo.flatten())
    variance_acc[i] = np.sum(valmax[::2])
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
    if sup - infm < .000075:  # .000075 for normal domain # this value may need to be adjusted
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


def phase_similarity(ftopo):
  mode1 = np.zeros(ftopo.shape, dtype=np.complex64)
  mode2 = np.zeros(ftopo.shape, dtype=np.complex64)

  phases = np.imag(ftopo)
  phase_flat = phases.flatten()
  phases_sort = np.sort(phase_flat)

  mask1 = phase_flat > 210.0
  phase_masked = phase_flat[mask1]
  mask2 = phase_masked < 320.0
  phase_masked2 = phase_masked[mask2]

  point1 = list(np.where(phases == phase_masked2[0]))
  point2 = list(np.where(phases == phase_masked2[1]))

  mode1[point1[0], point1[-1]] = ftopo[point1[0], point1[-1]]
  mode2[point2[0], point2[-1]] = ftopo[point2[0], point2[-1]]

  topo_mode1 = np.fft.ifft2(mode1, norm="ortho")
  topo_mode1 = topo_mode1.real
  topo_mode1 -= topo_mode1.min()

  topo_mode2 = np.fft.ifft2(mode2, norm="ortho")
  topo_mode2 = topo_mode2.real
  topo_mode2 -= topo_mode2.min()

  plt.figure()
  plt.pcolormesh(topo_mode1)
  plt.figure()
  plt.pcolormesh(topo_mode2)


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


def generate_mode_w_list(list_of_modes, spec_topo, ftopo, valmax= [], loc = []):
  gen = np.zeros(spec_topo.shape, dtype=np.complex64)
  # if len(loc)== 0:

  for i in range(len(list_of_modes)):
    ind = list(np.where(spec_topo == list_of_modes[i]))
    if (len(ind[0]) > 1):
      ind[0] = np.array([ind[0][np.mod(i, 2)], ])
      ind[1] = np.array([ind[1][np.mod(i, 2)], ])
    gen[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
  # else:
  #   for i in range(len(list_of_modes)):
  #     ind = list(np.where(spec_topo == list_of_modes[i]))
  #     if (len(ind[0]) > 1):
  #       ind[0] = np.array([ind[0][np.mod(i, 2)], ])
  #       ind[1] = np.array([ind[1][np.mod(i, 2)], ])
  #     gen[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
  #
  #     ind = list(np.where(spec_topo == list_of_modes[i]))
  #     if (len(ind[0]) > 1):
  #       ind[0] = np.array([ind[0][np.mod(i, 2)], ])
  #       ind[1] = np.array([ind[1][np.mod(i, 2)], ])
  #     gen[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]

  return gen

def multiples(num):
  repeat = True
  mult1 = 2
  mults = []
  while repeat:
    if num%mult1 == 0:
      mults.append((mult1, int(num/mult1)))
      if len(mults) > 1:
        if mults[-2][0] == num/mult1:
          repeat = False
    mult1 += 1
    if mult1 > 50:
      return [int(np.ceil(num/2)),2]
  return mults[-1]


def gen_mode_plots(mode_list, spec_topo, ftopo, total_var, valmax, gr_dict = [], box=10, total_box=240, \
                   grouping = 'linear', group_stag = None, output_topo = False, topo_name = None ):

  ifig = 0
  fig = plt.figure(ifig, figsize=[10, 10])
  if grouping == 'staggered':
    box = 1
    total_box = len(group_stag)-1
  assert(len(mode_list) >= box*total_box)
  pl = int(total_box / box)
  pl1, pl2 = multiples(pl)
  gs = gridspec.GridSpec(pl1, pl2)
  gs.update(**sz)
  cmap_topo = cm.get_cmap("rainbow")
  isub = 0
  levels = np.linspace(0, 60, 30)
  max_h = True
  if output_topo:
    if os.path.exists('data/' + topo_name):
      raise ValueError('Folder Already Exists :: May want to Save Files')
    else:
      os.makedirs('data/' + topo_name)
  if grouping == 'staggered':
    kxky_group = []
  group_it = 1
  sum_topo = np.zeros(np.shape(ftopo))
  val_list_check = []
  for i in range(0, total_box, box): #should be range(0,total_box/box,box)
    if grouping == 'staggered':
      group = mode_list[int(np.sum(group_stag[:i+1])) : int(np.sum(group_stag[:i+2]))]
      val_list_check.append(np.arange(int(np.sum(group_stag[:i+1])) , int(np.sum(group_stag[:i+2]))))
    else:
      group =  mode_list[i:i + box]
    fgroup_modes = generate_mode_w_list(group, spec_topo, ftopo) #valmax= valmax, loc= np.arange(int(np.sum(group_stag[:i+1])),int(np.sum(group_stag[:i+2]))))
    group_modes = np.fft.ifft2(fgroup_modes, norm="ortho")
    group_modes = group_modes.real
    group_modes -= group_modes.min()
    sum_topo += group_modes
    if output_topo:
      topodict = {'topo': group_modes}
      savemat('./data/'+topo_name +'/' + topo_name+ str(group_it)+'.mat', topodict)
    if max_h:
      levels = np.linspace(0, np.max(group_modes)+5, 40)
      max_h = False
    ax = plt.subplot(gs[isub])
    ax.set_aspect('equal', adjustable='box')
    cax = ax.contourf(x, y, group_modes, levels, cmap=cmap_topo, extend='both')
    if i < int(total_box/box-pl2):
      ax.set_xticklabels([])
      ax.set_yticklabels([])
    if i >= int(total_box/box-pl2):
      ax.set_ylabel('Ly')
      ax.set_xlabel('Lx')
    kx_list = []
    ky_list = []
    k_mag_list = []
    for iter in range(0,len(group)):
      k = return_xy(group[iter], spec_topo)
      if k[0] > 50:
        k[0] = np.shape(ftopo)[0]-k[0]
      if k[1] > 50:
        k[1] = np.shape(ftopo)[0]-k[1]
      kx_list.append(int(k[0])+1)
      ky_list.append(int(k[1])+1)
      k_mag_list.append(np.sqrt((k[0])**2+(k[1])**2)) # +1 here because python ind starts at 0 but wavenumber 1
    if len(gr_dict)>0:
      gr_dict[str(group_it)]
    kx_max = max(kx_list)
    ky_max = max(ky_list)
    # kx_min = min(kx_list)
    # ky_min = min(ky_list)
    k_mag_min = min(k_mag_list)
    k_mag_max = max(k_mag_list)

    if grouping == 'staggered':
      kxky_group.append([ k_mag_min, k_mag_max])
    if len(gr_dict) > 0:
      k,A = zip(*gr_dict[str(group_it)])
      kxky_group.pop()
      kxky_group.append([min(k), max(k)])
    if len(group) == 1:
      k = return_xy(group, spec_topo)
      kx = k[0]; ky = k[1]
      #ax.set_title('kx,ky '+ str(kx) + ','+str(ky) +' $|\phi^2_i|/|\sigma^2|=$' +'{:4.2f}%'.format(group[0] / var_topo * 100))
      ax.set_title('Group ' + str(group_it) + ' : $|\phi^2_i|/|\sigma^2|=$' + '{:4.2f}%'.format( \
        np.sum(group) / (2.0*var_topo) * 100) + ' : ' + str(int(np.sum(group_stag[:i + 1]))+1) + \
                   ' to ' + str(int(np.sum(group_stag[:i + 2]) )) + ' : $|k|_{max} = $ ' + '{:4.2f}'.format( \
                        k_mag_max))
      group_it += 1
    else:
      if grouping == 'linear':
        ax.set_title('$|\phi^2_i|/|\sigma^2|=$'+'{:4.2f}%'.format(np.sum(group) / var_topo * 100) )
      else:
        ax.set_title('Group '+ str(group_it) + ' : $|\phi^2_i|/|\sigma^2|=$' + '{:4.2f}%'.format(np.sum(group) / (2.0*var_topo) * 100) + ' : ' + str(int(np.sum(group_stag[:i+1]))+1) +\
                     ' to ' + str(int(np.sum(group_stag[:i+2]))) + ' : $|k|_{max} = $ '  + '{:4.2f}'.format(k_mag_max))
        group_it += 1

    isub += 1

  cbar_ax = fig.add_axes([0.05, 0.45, 0.02, 0.3])
  cbar = plt.colorbar(cax,
                      ticks=np.arange(levels[0], round(levels[-1]), round(levels[-1]/5)), cax=cbar_ax)
  cbar.ax.set_xlabel(r'hgt (m)')
  cbar.ax.xaxis.set_label_position('top')
  plt.show()
  if grouping == 'staggered':
    return kxky_group, sum_topo
  else:
    return None

def return_xy(mode,spec_topo):
  kxky = np.array([])
  for i in range(np.shape(spec_topo)[0]):
    for j in range(np.shape(spec_topo)[1]):
      if spec_topo[i,j] == mode:
        kxky= np.append(kxky,[i,j])

  if len(kxky)> 1:
    return kxky
  else:
    return kxky[0]

def fold_topo(topo):

  ftopo = np.fft.fft2(topo/np.max(topo), norm= 'forward')
  N = np.shape(ftopo)[0];
  if N%2 == 0:
    nf = int(N/2)
  else:
    nf = int((N-1)/2)
  S = np.zeros([nf, nf])
  Fa = (ftopo*np.conj(ftopo)) #/np.real(N)**2


  # if N%2 == 0:
  #   S = np.zeros([nf, nf])
  #   for i in range(nf):
  #     for j in range(nf):
  #       if i == nf or j == nf:
  #         S[i, j] = Fa[i, j]
  #       else:
  #         S[i, j]= 2.0 * Fa[i, j]
  # elif N%2 == 1:
  #   S = np.zeros([int(N/2-1/2), int(N/2-1/2)])
  #   for i in range(int((N-1)/2)):
  #     for j in range(int(N/2-1/2)):
  #       if i == nf or j == nf:
  #         S[i, j] = Fa[i, j]
  #       else:
  #         S[i, j]= 2.0 * Fa[i, j]
  #
  for i in range(nf):
    for j in range(nf):
      if i == nf or j == nf:
        S[i, j] = Fa[i, j]
      else:
        S[i, j]= 2.0 * Fa[i, j]

  # This accounts for diagonal wavenumbers
  for i in range(N-4,N):
    for j in range(1,nf):
      if abs((Fa[i,j] - Fa[abs(i-N), abs(j-N)])/Fa[i,j]) < 10**-13:
        S[abs(i-N), j] += Fa[i,j]

  return S

def organize_by_wave_mag(ftopo, nmax):
  # topo_mean = np.mean(topo)
  # topo = topo-topo_mean
  # folded_topo = fold_topo(topo)
  # ftopo = np.fft.fft2(topo, norm= 'ortho')
  N = np.shape(ftopo)[0];
  nf = int(N / 2);
  spec_topo = np.abs(ftopo)**2
  var_topo = np.sum(spec_topo) / 2
  valmax = nlargest(2 * nmax, spec_topo.flatten())

# So here because I want to organize computational mode groups by wavenumber mag, but I want to sort by wavenumbers less than the nyquist frequency
  # so when a kmag is added to list it needs to be associated with the index/amplitude of the negative frequency. The code below is attempting to do this
  # but Im unsure if its dpoing ti correctly because I domt know how negative frequency is stored in np.fft

# Because f tje uncertainties I had above this section is porgrammed very poorly---THe if statment isnt needed if you ignore the negative frequency
  # Sorry to any future reader including myself
  pos_neg_index = []
  K_mag = []
  #K_mag = np.array([])
  for i in range(0,len(valmax),2):
    pos = return_xy(valmax[i],spec_topo)+1
    if (pos[0] > nf and pos[1] > nf):
      pos = return_xy(valmax[i+1], spec_topo) + 1
    elif np.max(pos[:2]) > nf:
      pos[np.argmax(pos[:2])] = N-pos[np.where(np.max(pos[:2]) == pos)][0]
    K_mag.append([np.sqrt(pos[0]**2+pos[1]**2), valmax[i]])
    #K_mag = np.append(K_mag,[np.sqrt(pos[0]**2+pos[1]**2), valmax[i]] )
  K_mag = np.asarray(K_mag)
  sort = np.argsort(K_mag[:,0])
  K_mag_sorted = np.zeros([len(sort),2])
  for i in range(len(sort)):
    K_mag_sorted[i,:] = K_mag[sort[i],:]
  groups = organize_by_group_variance(K_mag_sorted, np.sum(valmax[::2]),var_topo, div = 'equal')

  return groups, K_mag_sorted

def organize_by_group_variance(k,L_var, var, div = None):
  if div == 'equal':
    frac = L_var/var/6 # 4-6 is a good number as its divides the wavenumber space but also leaves enough variance leftover in the final group
    sum_ = 0
    groups = {}
    gr = 1
    for i in range(len(k)):
        if str(gr) in groups:
          groups[str(gr)].append(k[i,:])
        else:
          groups[str(gr)] = []
          groups[str(gr)].append(k[i, :])
        sum_ += k[i,1]/var
        if sum_ > frac:
          print(len(groups[str(gr)]))
          gr += 1
          sum_ = 0

  return groups

#%%

### END Functions ###
#________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________
### User-specified Variables ###
## File

os.chdir('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/DOE-ARM/Amazon_topography')

fn_npz = './data/K34.npz'
fmt_fig = './figures/simp_topography_n{n:d}.png'
fmtfn_out = './data/amazon_topo_acc{:d}.npz'
lx = 2880 #5760 ATTO_large #3000 ATTO_3000 #2880 ATTO,K34
ly = 2880 #5760 ATTO_large #3000 ATTO_3000 #2880 ATTO,K34

# Calculate the Lorentz Curve and optimal mode
Lorentz = True
# Analyze the phases (useless function)
phase_test = False
# Organize by wave magnitude(useless function)
organize_by_wave_mag_flag = False
# Plot series of mode combinations
mode_plot = True
# Calculate Spectra
Spectra_flag = True
## Simplified topography parameters
# nmax read from commmand line
nmax = 199 # K34 number of fourier modes
# nmax = 171 # ATTO
#nmax = 188 # K34-like
# nmax = 551 # ATTO-large
# nmax = 206 # ATTO_3000
'''
Where do the values of nmax come from?
nmax = number of modes considered to break down the topography?
'''
## Figure
sz = dict(left=0.08, right=0.98, bottom=0.06, top=0.92,
  wspace=0.05, hspace=0.2)
level_ele = (-30, 30)
#level_ele = (-30, 40)
level_spec = (4, 6)
lw = 2
xspacing = 2
yspacing = 2

## Data parameters
nbyte = 8

    ### END User-specified Variables ###

#%%

### Main Body ###
## Read data
print('#'*80)
print('  Reading the topography from {fn:s}...'.format(fn=fn_npz))
topo = np.load(fn_npz)['topo']
mean_topo = topo.mean()
print(mean_topo)
topo -= mean_topo #remove the mean for the subsequent fourier analysis
ny = topo.shape[0]
nx = topo.shape[1]
dx = lx/nx
dy = ly/ny
y, x = np.mgrid[0:ly:dy, 0:lx:dx]
#nmax = int(sys.argv[1])

## Process data
print('#'*80)
# Forward FFT
print('  Fourier transforming the topography...')
kx = np.fft.fftfreq(nx, d=1/nx)
ky = np.fft.fftfreq(ny, d=1/ny)
#kx_r = np.roll(kx, nx//2-1)
#ky_r = np.roll(ky, nx//2-1)
ftopo = np.fft.fft2(topo, norm="ortho")
topo += mean_topo

print('  Get the most energetic modes...')
ftopo_acc = np.zeros(ftopo.shape, dtype=np.complex64)
ftopo_sin = np.zeros(ftopo.shape, dtype=np.complex64)
spec_topo = np.abs(ftopo)**2
#spec_topo = np.roll(spec_topo, ny//2-1, axis=0)
#spec_topo = np.roll(spec_topo, nx//2-1, axis=1)
valmax = nlargest(2*nmax, spec_topo.flatten())
var_topo = np.sum(spec_topo)/2
print('  Calculate Lorenze Curve...')

#%%

# Call function that calculates optimal mode based on Lorentz Curve
if Lorentz:
  optimal_mode, Lor_plt = Lorentz_Curve(ftopo)
  nmax = optimal_mode
  valmax = nlargest(2 * nmax, spec_topo.flatten())
  
#%%

# This is relatively useless
if phase_test:
  phase_similarity(ftopo)

#Generate ftopo Sin and ftopo Acc based off currecnt desired modes
ftopo_sin, ftopo_acc = generate_mode(valmax, nmax, spec_topo, ftopo)

# Make plot and organize by wave number
if organize_by_wave_mag_flag:
  groups,k_ = organize_by_wave_mag(ftopo, nmax)

# Make plot of every ith mode for certain N iterations
if mode_plot:
  #fig1 = gen_mode_plots(valmax, spec_topo, ftopo)
  #----------------------
  # Top 60 modes
  # largest_amps = valmax[:60]
  # fig1 = gen_mode_plots(largest_amps, spec_topo, ftopo, var_topo, box = 1, total_box= len(largest_amps))
  #
  # # Top 300 modes grouped
  largest_amps = valmax
  group_stag = [0, 1, 5, 9, 17, 40, nmax-(0 + 1 + 5 + 9 + 17 + 40)]
  # group_stag = [0, 2, 4, 12, 34, 80, nmax-(0 + 2 + 4 + 12 + 34 + 80)] # do not choose prime number

  # largest_amps = valmax[:nmax*2:2]
  # THIS ALL NEED TO BE MULTIPLIED BY 2 TO ACCOUNT FOR NEGATIVE FREQUENCIES OTHERWISE THEY ARE NOT ACCOUNTED FOR WHEN BUILDING THE TOPO
  # Currently only done for ATTO above
  # group_stag = [0, 1, 2, 6, 17, 40, nmax-(0+1+ 2+ 6+17+ 40)] # do not choose prime number
  # ATTO [0, 1, 2, 6, 17, 40, nmax-(0+1+ 2+ 6+17+ 40)]
  # K34 [0, 1, 5, 9, 17, 40, nmax-(0+1+ 5+ 9+17+ 40)]
  # K34-like [0, 3, 7, 11, 17, 40, nmax-(0+3+ 7+ 11+17+ 40)]
  # ATTO3k [0, 2, 3, 8, 17, 40, nmax-(0+2+ 3+ 8+17+ 40)]
  kxky_group, topo_check = gen_mode_plots(largest_amps, spec_topo, ftopo, var_topo, valmax, \
              box = 1 , total_box = len(largest_amps), grouping= 'staggered', group_stag=group_stag,output_topo = False, topo_name = 'ATTO_Ben')



  # Organized by Wavenumber groups outputted from organize_by_wave_mag_flag
  #group_stag = [len(groups[str(i+1)])  for i in range(len(groups))]
  #group_stag.insert(0, 0)
  #
  # #group_stag = [0, 3, 5, 10, 20, 55, nmax-(2+5+10+20+55)] # do not choose prime number
  #kxky_group = gen_mode_plots(k_[:,1], spec_topo, ftopo, var_topo, groups, \
  #             box = 5 , total_box = len(k_[:,1]), grouping= 'staggered', group_stag=group_stag)

if Spectra_flag:
  if mode_plot:
    compute_spectra(topo, group_waves=kxky_group)  # k_sort = k+sort
  else:
    compute_spectra(topo)


# Inversed FFT
print('  Inversed Fourier transforming the topography...')
topo_acc = np.fft.ifft2(ftopo_acc, norm="ortho")
topo_acc = topo_acc.real
topo_acc -= topo_acc.min()
print(topo_acc.max())
topo_sin = np.fft.ifft2(ftopo_sin, norm="ortho")
topo_sin = topo_sin.real
topo_sin -= topo_sin.min()

# # Save the simplified topography (accumulated version)
# fn_out = fmtfn_out.format(nmax)
# np.savez(fn_out, topo=topo_acc-topo_acc.mean()+mean_topo)

#%%

## Plot the data
print('#'*80)
print('  Plotting the data...')
levels_topo = np.arange(level_ele[0], level_ele[1]+0.1, 5)
cmap_topo = cm.get_cmap("rainbow")
xmaxLocator = MultipleLocator(xspacing)
ymaxLocator = MultipleLocator(yspacing)

print('  Initializaing the plot environment...')
ifig = 1
fig = plt.figure(ifig, figsize=[8, 8])
gs = gridspec.GridSpec(2, 2)
gs.update(**sz)

# Full topography
print('  Plotting the isolines of full topography...')
isub=0
ax = plt.subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')
cax = ax.contourf(x, y, topo-topo.mean(), levels_topo, cmap=cmap_topo, extend='both')
# ax.contour(x, y, topo, [40,],
#   colors='k', linewidths=1.5, linestyles='--')
#ax.set_xlabel('x (m)')
ax.set_xticklabels([])
ax.set_ylabel('y (m)')
ax.set_title('Full topography')

if not(Lorentz):
  # Single-mode topography
  print('  Plotting the isolines of single-mode topography...')
  isub += 1
  ax=plt.subplot(gs[isub])
  ax.set_aspect('equal', adjustable='box')
  cax = ax.contourf(x, y, topo_sin-topo_sin.mean(), levels_topo, cmap=cmap_topo, extend='both')
  ax.set_xlabel('x (m)')
  #ax.set_ylabel('y (m)')
  ax.set_yticklabels([])
  ax.set_title('Single-mode topography:\n'+'$i_{mode}=$'+'{n:d}, '.format(n=nmax)
    + '$|\phi^2_i|/|\sigma^2|=$' + '{:4.2f}%'.format(
    valmax[2*nmax-1]/var_topo*100))
else:
  isub += 1
  ax = plt.subplot(gs[isub])
  ax.set_aspect('equal', adjustable='box')
  ax.plot(Lor_plt['var_acc'], Lor_plt['perc_modes'], linewidth= 2.0)
  ax.plot(Lor_plt['opt_var'], Lor_plt['optimal_mode'], 'ro', linewidth= 2.0)
  ax.set_ylabel('Percent Coefficients Represented')
  ax.set_xlabel('Topographic Variance')
  ax.set_title('Lorentz Curve')

# Simplified topography
print('  Plotting the isolines of simplified topography...')
isub += 1
ax=plt.subplot(gs[isub])
ax.set_aspect('equal', adjustable='box')
cax = ax.contourf(x, y, topo_acc-topo_acc.mean(), levels_topo, cmap=cmap_topo, extend='both')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Simplified topography with '+'$n_{modes}=$'+'{n:d}:\n'.format(n=nmax)
  + '$\sum^n{|\phi^2_i|}/|\sigma^2|=$' + '{:4.2f}%'.format(
  np.sum(valmax[::2])/var_topo*100)) #np.sum(valmax[::2])/var_topo*100)

# Colorbar
print('  Generating the colorbar labels...')
cbar_ax = fig.add_axes([0.7, 0.1, 0.02, 0.3])
cbar = plt.colorbar(cax,
  ticks = np.arange(level_ele[0], level_ele[1]+10, 10), cax=cbar_ax)
cbar.ax.set_xlabel(r'hgt (m)')
cbar.ax.xaxis.set_label_position('top')

plt.show()

# Save file
#print('  Saving the figure...')
#plt.savefig(fmt_fig.format(n=nmax), dpi=600)

print('#'*80)
### END Main Body ###

#%%
#
# K_mag = np.array([])
# Pow = np.array([])
# indexes = []
# spec_topo = np.abs(ftopo) ** 2
#
# for i in range(np.shape(ftopo)[0]):
#   for j in range(np.shape(ftopo)[0]):
#     if ftopo[i, j] * np.conj(ftopo[i, j]) != 0.0:
#       K_mag = np.append(K_mag, np.sqrt(i ** 2 + j ** 2))
#       Pow = np.append(Pow, spec_topo[i, j])  # ftopo[i,j]*np.conj(ftopo[i,j]))
#       indexes.append((i, j))
#
# plt.figure()
# plt.plot(K_mag, Pow, 'bo')
# plt.xlabel('Wave Number Magnitude')
# plt.ylabel('Spectra Inner Product')
# plt.title('minimum required |k| Organization')
#
# # Scatter of waveumber vs mode number with color of amplitude
# # So the thing with the plot Marc Suggested is the y-axis is mode number
# # but the mode number is a choose by how we want to orgqnize the modes.
# # So you loose the unbiased perspective by organizing the modes
# K_mag_sorted = []
# modes_ = []
# for i in range(0, len(top_modes), 2):
#   xp = return_xy(top_modes[i], spec_topo_OG)
#   # print(i)
#
#   K_mag_sorted.append(np.sqrt(xp[0] ** 2 + xp[1] ** 2))
#   modes_.append(i)
#
# plt.figure()
# plt.scatter(K_mag_sorted, modes_, c=top_modes[::2], norm=matplotlib.colors.LogNorm())
# plt.colorbar()
# plt.xlabel('|k|')
# plt.ylabel('Mode #')
#
# # fig = plt.figure()
# # ax = plt.gca()
# # ax.scatter(x,y, c=Pow)
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# # plt.tight_layout()
#
# # ---------- Could be its own function ------------
# # Bin the wavenumbers and associated information by wavenumber magnitude
# bins = np.arange(0.0, np.max(K_mag), 3.0)
# inds = np.digitize(K_mag, bins)
# wave_bins = dict()
# for i in range(len(K_mag)):
#   if str(inds[i]) in wave_bins.keys():
#     wave_bins[str(inds[i])].append((K_mag[i], indexes[i], Pow[i]))
#   else:
#     wave_bins[str(inds[i])] = [(K_mag[i], indexes[i], Pow[i])]
# # ---------- Could be its own function ------------
# # Generate and plot the resulting bin as a plot
# bin1_mode_list = []
# for i in range(len(wave_bins['1'])):
#   bin1_mode_list.append(wave_bins['2'][i][2])
#
# mode_bin1 = generate_mode_w_list(bin1_mode_list, spec_topo, ftopo)
# mode_bin1 = np.fft.ifft2(mode_bin1, norm="ortho")
# mode_bin1 = mode_bin1.real
# mode_bin1 -= mode_bin1.min()
# plt.figure()
# plt.pcolormesh(mode_bin1)
# plt.colorbar()
# # I need to have some critea for if a there isnt enough variance in a bin such that I ccombine neighboring bins
# # Make Color scatterplot of the data in wavenumber space with color as amplitude
# x, y = zip(*indexes)
#
# ax = plt.scatter(x, y, c=Pow)
# plt.colorbar()
#
# plt.tight_layout()
#
# ax1 = plt.scatter(x, y, c=Pow)
# plt.colorbar()
# plt.xlim(0, 3)
# plt.ylim(0, 3)
# plt.tight_layout()
#
# # return kx,ky with mode amplitude
# # inds = []
# # for i in range(len(top_modes)):
# #   inds.append(return_xy(top_modes[i], spec_topo_OG))

# var_topo = np.sum(folded_topo)
# num_sims = 6.0
# var_div = var_topo / num_sims
#
# sorted_k = []
# for i in range(nf):
#   for j in range(nf):
#     K_mag = np.sqrt(i ** 2 + j ** 2)
#     for q in range(sorted_k):
#       if len(sorted_k > 0):
#         if sorted_k[q] <= K_mag or sorted_k[q + 1] > K_mag:
#           sorted_k.insert([K_mag, spec_topo[i, j]], q + 1)
#       else:
#         sorted_k.append([K_mag, spec_topo[i, j]])

# if len(pos) > 2:
#   k_temp = np.sqrt(pos[0]**2+pos[1]**2)
#   if ((pos[0] > nf or pos[1] > nf) and (pos[2] > nf or pos[3] > nf)):
#     klarge = N-pos[np.where(np.max(pos[:2]) == pos[:2])][0] # BY Def of DFT this is true for wavenumber magnitude purposes
#     pos[np.argmax(pos[:2])] = klarge
#     k_temp =  np.sqrt(pos[0]**2+pos[1]**2)
#   K_mag.append(k_temp)
#   pos_neg_index.append(pos) #plus 1 here because python index starts at 0 but wavenumber at 1
# else:
#   # if np.abs(valmax[i] - valmax[i + 1])/valmax[i] > 10.0 ** -14:
#   #   pop = 1
#   k_temp = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
#   neg = return_xy(valmax[i+1], spec_topo)
#   if ((pos[0] > nf or pos[1] > nf) and (neg[0] > nf or neg[1] > nf)):
#     klarge = N-pos[np.where(np.max(pos[:2]) == pos[:2])][0] # BY Def of DFT this is true for wavenumber magnitude purposes
#     pos[np.argmax(pos[:2])] = klarge
#     k_temp =  np.sqrt(pos[0]**2+pos[1]**2)
#   pos_neg = np.array([pos,neg]).flatten()
#   pos_neg_index.append(pos_neg)