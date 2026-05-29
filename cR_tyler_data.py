#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:06:47 2024

@author: u1450851
"""

import pickle

# Replace 'filename.pkl' with the path to your pickle file
file_path = '/uufs/chpc.utah.edu/common/home/u1450851/Tyler_yB/data4ben.pkl'
pathOUT = '/uufs/chpc.utah.edu/common/home/u1450851/Tyler_yB/Figures/'

# Open the pickle file in binary read mode ('rb') and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now you can use the 'data' variable to access the contents of the pickle file
print(data)
        
        
#%%
import matplotlib.pyplot as plt
import numpy as np

data['site_std_dsm'] = np.nan_to_num(data['site_std_dsm'], nan=0)
sorted_indices = np.argsort(data['site_std_dsm'])
data['site_std_dsm'] = data['site_std_dsm'][sorted_indices]
data['sites'] = np.array(data['sites'], dtype=str)
data['sites'] = data['sites'][sorted_indices]

for i in range(0,data['ybd_site'].shape[1]):
    data['ybd_site'][:,i] = data['ybd_site'][:,i][sorted_indices]
    
cR = np.zeros_like(data['ybd_site'])

for i in range(0,data['ybd_site'].shape[0]):
    for j in range(0,data['ybd_site'].shape[1]):
        cR[i,j] = 1/(2-(4/np.sqrt(3))*data['ybd_site'][i,j])

site_names = data['sites']
stability_values = data['zL'][1:-1]

X, Y = np.meshgrid(np.arange(len(site_names)), stability_values)

fig, ax = plt.subplots(figsize=(10, 6))

sc = ax.scatter(X, Y, c=cR[:,1:-1].T, cmap='jet')

plt.colorbar(sc, ax=ax, label='cR values')

ax.set_xticks(np.arange(len(site_names)))
ax.set_xticklabels(site_names, rotation=90)

ax.set_xlabel('Sites')
ax.set_ylabel('Stability (z/L)')

ax.set_ylim(-1,1)

plt.tight_layout()

# plt.savefig(pathOUT+'cR_allsites_ordered.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()