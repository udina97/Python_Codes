#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:32:02 2023

@author: u1450851

build .dat file and write to file for LES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

outpath = '/uufs/chpc.utah.edu/common/home/u1450851/LES_code/UtahIBMParticle/misc'

#domain parameters

zi = 1000
lx = 2000/zi
ly = 100/zi
lz = 540/zi
nx = 320
ny = 16

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

H = 50/zi #topography height
L = 250/zi #topography half length

z = np.zeros((ny,nx))

for j in range(0,ny):
    for i in range(0,nx):
        z[j,i] = (H/2)*np.cos((np.pi/(2*L))*x[i] + np.pi) + H/2
        


#%%

X, Y = np.meshgrid(x, y)

# Create the first subplot
fig = plt.figure(figsize=(10, 13))
ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax1.plot_surface(X, Y, z, cmap=cm.plasma)
ax1.set_zlim(0, lz)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Create the second subplot
ax2 = fig.add_subplot(2, 1, 2, projection='3d')
ax2.plot_surface(X, Y, z, cmap=cm.plasma)
ax2.set_zlim(0, lz)
ax2.set_box_aspect([lx, ly, lz])
ax2.view_init(elev=0, azim=90)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Show the plots
plt.show()

# Plot the 2D plot
plt.figure()
plt.plot(x, z[0, :])
plt.xlabel('x')
plt.ylabel('z')
plt.show()

#%%

# Reshape x, y, and z

x2 = X.reshape(nx * ny, 1)
y2 = Y.reshape(nx * ny, 1)
z2 = z.reshape(nx * ny, 1)

# Create Delaunay triangulation
points = np.hstack((x2, y2))
tri = Delaunay(points)

nElems = len(tri.simplices)
nNodes = len(points)

nodeID = np.arange(1, nNodes + 1)
elemID = np.arange(1, nElems + 1)

# Write to file
with open(outpath+'/bicheng_hill_Ben_python.dat', 'w') as fileID:
    fileID.write(f'nNodes {nNodes}\n')
    fileID.write(f'nElems {nElems}\n')
    fileID.write('\nnodeID xCoord yCoord zCoord\n')
    for i in range(nNodes):
        fileID.write(f'{nodeID[i]:14.6f} {points[i][0]:14.6f} {points[i][1]:14.6f} {z2[i][0]:14.6f}\n')

    fileID.write('\nelemID NodeID1 NodeID2 NodeID3\n')
    for i in range(nElems):
        fileID.write(f'{elemID[i]:14.6f} {tri.simplices[i][0]:14.6f} {tri.simplices[i][1]:14.6f} {tri.simplices[i][2]:14.6f}\n')








































