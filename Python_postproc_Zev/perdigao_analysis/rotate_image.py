import PIL
import pandas as pd
from scipy.io import loadmat,savemat
from scipy.ndimage import rotate
import plot_data as plot
import numpy as np
import matplotlib as plt
# ***********Uncomment to rotate matrix
# # mat1 = pd.read_csv('/uufs/chpc.utah.edu/common/home/u0851921/Codes/UtahSolarCodeLES/misc/surf_Tmat.csv')
# mat1 = pd.read_csv('/uufs/chpc.utah.edu/common/home/u0851921/Codes/UtahSolarCodeLES/misc/M.csv')
# #mat = loadmat('/uufs/chpc.utah.edu/common/home/u0851921/Codes/UtahSolarCodeLES/misc/perdigao_TBR.mat')
# print(np.shape(mat1))
# mat1 = rotate(mat1,-5.5,reshape = False)
# #plot.plot_matrix(mat1)
# print(np.shape(mat1))
#
# np.savetxt('/uufs/chpc.utah.edu/common/home/u0851921/Codes/UtahSolarCodeLES/misc/perdigao_rot.csv',mat1,delimiter=',')
#
# #savemat('/uufs/chpc.utah.edu/common/home/u0851921/Codes/UtahSolarCodeLES/misc/perdigao_rot.csv',mat1)
#***********

data = pd.read_csv('/scratch/general/lustre/u0851921/simtest/profile.txt', header = None)
d1 = data.iloc[:,3];
y = []; x= []
for i in range(0,len(d1)):
    sp  = d1[i].split('   ')
    if i%4 == 0:
        y.append(float(sp[5]))
        x.append(float(sp[7]))

plot.plot_profile(x,y)

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



z = loadmat('/uufs/chpc.utah.edu/common/home/u0851921/Codes/matlab_processing_code/perd.mat')
#
# Key to correct roation is to roate the full domain and then narrow down to matirx of interest
zrot = rotate(z['zinterp'],5, reshape= False)
# theta = 5
# rot_mat = np.array([np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)])
# for i in range(0,np.shape(z['zint'])[0]):
#     for j in range(0, np.shape(z['zint'])[1]):
#         vrot = np.matmul(np.array([i,j]),rot_mat)
plt.figure()
plt.pcolormesh(np.transpose(zrot[374:629,376:503]))


ele_org = zrot[374:630,376:504]
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


plt.figure()
plt.plot(ele[:,1])
#plt.plot(ele[:,2])
plt.plot(ele[:,126])
#plt.plot(ele[:,125])

plt.figure()
plt.plot(ele[1,:])
#plt.plot(ele[2,:])
plt.plot(ele[254,:])
#plt.plot(ele[253,:])

topodict = {'topo': ele}
savemat('/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_simulation_results/validation/val_topo.mat', topodict)




pop = 1
