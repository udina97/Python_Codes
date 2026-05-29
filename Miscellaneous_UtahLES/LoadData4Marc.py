#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:39:37 2024

@author: u1450851
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

#%%Load Giulia and Ben data

path = '/uufs/chpc.utah.edu/common/home/calaf-group3/CanopyData4Marc/'

with open(path+'empty.pickle', 'rb') as file:
    empty = pickle.load(file)
    
with open(path+'canopy.pickle', 'rb') as file:
    canopy = pickle.load(file)
    
with open(path+'canopy_aniso_full.pickle', 'rb') as file:
    canopy_aniso_full = pickle.load(file)
    
with open(path+'empty_aniso_full.pickle', 'rb') as file:
    empty_aniso_full = pickle.load(file)
    
#%%

iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]

#Transform the HEX colors to RGB.
from PIL import ImageColor

iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')

for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")

iva_colors_RGB = iva_colors_RGB[:,:]/(256)

#Transform the array of colors to a list of values. 
colors = iva_colors_RGB.tolist()
#----------------------------------------------------

#The next few lines create a new colormap using IVA's colors:
from matplotlib.colors import LinearSegmentedColormap,ListedColormap

inbetween_color_amount = 10

# the 10 is from the original 10 colors, the 4 is for R, G, B, A
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))

# add first one already
newcolvals[0] = colors[0]

for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
    
newcolvals

cmap = ListedColormap(newcolvals, name='from_list', N=None)

#%% Analyse data

'''
Some info on Giulia's data:
    Nx = Ny = Nz = 256
    Lx = Ly = 2*pi
    Lz = 1000m
    zi = 1000m
    Canopy Height = 39m
    
Some info on Ben's data:
    Nx = Ny = 256
    Nz = 384
    Lx = Ly = 2880m
    Lz = 960m
    zi = 1000
    Canopy height = 39m
    First 5 grid nodes in the z direction should be neglected because they are within the IBM
    
ALL DATA IS ON UVP NODES FOR BOTH DATASETS (keep this in mind if you compute the Reynolds stresses with Marc's function which interpolates
                                            data from w nodes to uvp nodes, this is not needed)
'''

Nx_B = 256
Ny_B = 256
Nz_B = 384
Lx_B = 2880
Ly_B = 2880
Lz_B = 960
zi = 1000
dz_B = (Lz_B/Nz_B)/zi

nz_ground = 5 #First point above the ground in pyhton (dz/2).
nz_canopy_top = int(np.ceil((39/zi)/dz_B)) + nz_ground

#Define the height matrix:
#z_ax_g = np.arange(10,Nz_SLayer+10)*dz_g #To be equivalent with previous analysis, we take 12 points (Nz_SLayer) above the canopy height.
#height_g = np.ones((Nx_G,Ny_G,Nz_SLayer))

z_ax_B = np.arange(0,Nz_B)*dz_B #To be equivalent with previous analysis, we take 12 points (Nz_SLayer) above the canopy height.
height_B = np.zeros((Nx_B,Ny_B,Nz_B))

for k in range(nz_ground,Nz_B):
    height_B[:,:,k] = z_ax_B[k]

ztop = 360

#-------------------------------------
#Load the data from the file: 

step = 32

#xB_flat_1D = np.ravel(ani_dict['xB'][0:Nx:step,0:Ny:step,5:ztop+5])
#yB_flat_1D = np.ravel(ani_dict['yB'][0:Nx:step,0:Ny:step,5:ztop+5])

xB_flat_1D = (canopy_aniso_full['xB'][0:Nx_B:step,0:Ny_B:step,nz_canopy_top:ztop]).flatten()
yB_flat_1D = (canopy_aniso_full['yB'][0:Nx_B:step,0:Ny_B:step,nz_canopy_top:ztop]).flatten()
height_B_1D = (height_B[0:Nx_B:step,0:Ny_B:step,nz_canopy_top:ztop]).flatten()

#Graphical representation of the Baricentric Map for the Flat homogeneous empty neutral simulation:
fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(5, 4))
# axs.plot(x_bottom,y_bottom,'-k',x_left,y_left,'-k', x_right,y_right,'-k')
# axs.scatter(x_edges,y_edges,s=25,marker='o',color='k')
# im = axs.scatter(xB_flat_1D,yB_flat_1D,s = 15,marker='o',color = 'gray',alpha=0.5)
im = axs.scatter(xB_flat_1D,yB_flat_1D,s = 15,c = height_B_1D, marker='o',cmap = cmap)

plt.colorbar(im)
axs.set_xlim(0,1)
axs.set_ylim(0,np.sqrt(3)/2)
axs.set_xlabel(f"xB")
axs.set_ylabel(f"yB")
axs.set_title(f'IBM Neutral Flat with Canopy')

#plt.savefig(f'{path_out}BaricentricMap_Neutral_Flat.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()