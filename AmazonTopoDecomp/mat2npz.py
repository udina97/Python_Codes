from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

topo_m = loadmat('/uufs/chpc.utah.edu/common/home/u0851921/Codes/matlab_processing_code/ATTO1.mat')
val2S = {'topo' : topo_m ['z']}
np.savez('topography.npz', **val2S)
topo = np.load('topography.npz')

# plt.figure()
# plt.pcolormesh(topo['topo'])
# plt.show()

check = np.load('amazon_topo.npz')
pop = 1