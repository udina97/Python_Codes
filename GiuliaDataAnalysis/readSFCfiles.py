# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 06:37:47 2025

@author: udina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from matplotlib.colors import ListedColormap

#%%

# Change directory to input folder
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/input_txt_files/'
case = 'input_p4/'
os.chdir(path+case)
gap = 400  # meters

# Load forest geometry
forest = np.loadtxt('forest.txt', dtype=int)
forest_area = np.loadtxt('forest_area.txt')
tree_index_array = np.loadtxt('tree_index_array.txt', dtype=int)

num_of_trees = tree_index_array[-1, 0]
tree_num_pts = tree_index_array[-1, 1]

# Load colormap
os.chdir('..')
mat = loadmat('my_summer3.mat')
my_summer3_array = mat['my_summer3']  # shape should be (N, 3) or (N, 4)
my_summer3 = ListedColormap(my_summer3_array)

# Grid setup
zi = 1000
nx = ny = nz = 256
lx = ly = 2 * np.pi * zi
lz = zi
dx, dy, dz = lx / nx, ly / ny, lz / nz

# Allocate arrays
LAD = np.zeros((nx, ny))
xo = np.zeros((nx, ny))
x = np.zeros(nx * ny)
y = np.zeros(nx * ny)

for l in range(1, num_of_trees + 1):
    for k in range(1, tree_num_pts + 1):
        displac = (l - 1) * tree_num_pts + (k - 1)

        tree_index = forest[displac, 0]
        i_here = forest[displac, 1] - 1
        j_here = forest[displac, 2] - 1
        k_here = forest[displac, 3] - 1

        LAD[i_here, j_here] = forest_area[displac, 3]  # y
        x[l - 1] = i_here
        y[l - 1] = j_here

# Occupancy matrix
xo = LAD / LAD.max()

# Reshape for plotting
xx = x.reshape((nx, ny))
yy = y.reshape((nx, ny))

# Plot LAD
plt.figure(figsize=(6, 6))
plt.pcolormesh(xx.T, yy.T, LAD.T, shading='auto', cmap=my_summer3)
plt.gca().set_aspect('equal')
plt.colorbar(label='LAD')
plt.title("LAD with Summer Colormap")

# Compute gap centers
r = int(np.floor(0.5 * gap / dx))
print(f"Gap radius (in grid cells): {r}")
C = []

a = 1  # 0 for gap, 1 for patch

for i in range(r, nx - r + 1):
    for j in range(r, ny - r + 1):
        if (xo[i, j] == a and
            xo[i, j - r + 1] == a and
            xo[i, j + r - 1] == a and
            xo[i - r + 1, j] == a and
            xo[i + r - 1, j] == a):
            C.append((i, j))

C = np.array(C)
xc = C[:, 0] if C.size > 0 else np.array([])
yc = C[:, 1] if C.size > 0 else np.array([])

# Plot with centers
plt.figure(figsize=(6, 6))
plt.pcolormesh(xx.T, yy.T, LAD.T, shading='auto', cmap=my_summer3)
plt.plot(xc, yc, 'k.', label='Gap Centers')
for i, j in zip(xc, yc):
    circ = plt.Circle((i, j), r, color='k', fill=False, linewidth=1.5)
    plt.gca().add_patch(circ)
plt.gca().set_aspect('equal')
plt.title("Gaps with Detected Centers")
plt.legend()

# Save centers
S = {'xc': xc, 'yc': yc, 'r': r}

#%%save the array
os.chdir(path+case)
np.save('sfc.npy',LAD)