#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:28:09 2024

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import get_var_ts

#%% Set path

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Figures/TimeSeries_Anisotropy/'

# path = '/scratch/general/nfs1/u1450851/LES_Sims/simflat_256x256x384_ts_120min/output/time_series/'
# path = '/scratch/general/nfs1/u1450851/LES_Sims/simbicheng_valley_256x256x384_ts_120min/output/time_series/'
# path = '/scratch/general/nfs1/u1450851/LES_Sims/simbicheng_hill_256x256x384_ts_120min/output/time_series/'
# path = '/scratch/general/nfs1/u1450851/LES_Sims/simATTO_valley_256x256x384_ts_120min/output/time_series/'
path = '/scratch/general/nfs1/u1450851/LES_Sims/simATTO_256x256x384_ts_120min/output/time_series/'
os.chdir(path)

#%% Open Time series files

# nx_coords = [50,128,200,50,200,50,128,200,128] #flat coords
# ny_coords = [50,128,200,200,50,128,50,128,200]
# index = [5,5,5,5,5,5,5,5,5]

# nx_coords = [84,84,84,173,173,173] #bicheng valley coords
# ny_coords = [50,128,200,200,50,128]
# index = [5,5,5,5,5,5,5,5,5]

# nx_coords = [40,129,218,40,218,40,129,218,129] #bicheng ridge coords
# ny_coords = [50,128,200,200,50,128,50,128,200]
# index = [24,24,24,24,24,24,24,24,24]

# nx_coords = [54,29,43,194,180,211,220,223,222] #ATTO valley coords
# ny_coords = [171,132,153,60,58,59,107,43,85]
# index = [9,8,7,6,8,7,7,7,7]

nx_coords = [56,78,94,102,131,146,159,181,155] #ATTO ridge coords
ny_coords = [31,70,94,110,132,182,201,222,157]
index = [36,37,38,38,38,38,38,38,39]


nz = 384
ztop = 960
dz = ztop/nz
canopyH = 39
nt = 72000
avg_T = 72000
mpiProc = 32
dt = 0.1
zi = 1000
uscale = 0.4

data_ts = dict()
data_ts_resample = dict()

for i in range(len(nx_coords)):
    fname_u = 'TimeSeries_'+str(nx_coords[i])+'_'+str(ny_coords[i])+'_u'
    fname_v = 'TimeSeries_'+str(nx_coords[i])+'_'+str(ny_coords[i])+'_v'
    fname_w = 'TimeSeries_'+str(nx_coords[i])+'_'+str(ny_coords[i])+'_w'
    
    data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = get_var_ts(path,fname_u,nz,nt,mpiProc,avg_T)
    data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = get_var_ts(path,fname_v,nz,nt,mpiProc,avg_T)
    data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = get_var_ts(path,fname_w,nz,nt,mpiProc,avg_T)
    
    data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])][index[i]:,:]
    data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])][index[i]:,:]
    data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])] = data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])][index[i]:,:]
    
for j in range(0,len(nx_coords)):
    for i in range(0,len(data_ts['u_'+str(nx_coords[j])+'_'+str(ny_coords[j])][:,0])-1):
        data_ts['w_'+str(nx_coords[j])+'_'+str(ny_coords[j])][i,:] = 0.5*(data_ts['w_'+str(nx_coords[j])+'_'+str(ny_coords[j])][i,:]+\
                                                                           data_ts['w_'+str(nx_coords[j])+'_'+str(ny_coords[j])][i+1,:])

#%% 

lvl = 22

time = np.arange(0, nt, 1)*dt/60

fig, axs = plt.subplots(3,1,figsize=(10,8),tight_layout=True)
c = ['k','r','g','b','y','brown','orange','purple','pink']
for i in range(len(nx_coords)):
    axs[0].plot(time,data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])][lvl,:],c=c[i],label=f'({nx_coords[i]};{ny_coords[i]})')
    axs[1].plot(time,data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])][lvl,:],c=c[i],label=f'({nx_coords[i]};{ny_coords[i]})')
    axs[2].plot(time,data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])][lvl,:],c=c[i],label=f'({nx_coords[i]};{ny_coords[i]})')

axs[2].set_xlabel('Time [min]')
axs[0].set_ylabel('u [-]');axs[1].set_ylabel('v [-]');axs[2].set_ylabel('w [-]')
# axs.set_xlim(0,60)
# axs.set_ylim(0,10)
axs[0].legend(loc='upper right')
axs[0].set_title(f'height: {(lvl)*dz}m')

plt.show()

#%%

tstep = 2343

z_ax = np.arange(0,nz-5)*dz/canopyH

fig, axs = plt.subplots(1,1,figsize=(5,9),tight_layout=True)
axs.plot(np.mean(data_ts['u_75_200'],axis=(1))[5:],z_ax,c='k')
axs.axhline(1,ls='--',c='k')
# axs.axhline(28/39,ls='--',c='k')

axs.set_xlabel('u/u_s',fontsize=14)
axs.set_ylabel('z/H',fontsize=14)
axs.set_ylim(0,z_ax[-1])

plt.show()


#%% MRD

def MRD_Marc(u1,u2,level,plot=False):
    #Loading the Data to be analyzed: ------------------------------------------------
    u1 = u1[level,:]
    u2 = u2[level,:]
    
    #Make the Signal a power of 2.
    M = np.int64(np.floor(np.log2(len(u1)))) #Maximum Power of 2 within the length of the signal measure at 20Hz.
    u1_short = u1[0:int(2**M)]
    u2_short = u2[0:int(2**M)]
    
    #-----------------------------------------------------------------------------------
    var1 = u1_short
    var2 = u2_short
    
    a = np.array(var1)
    b = np.array(var2)
    
    D = np.zeros(M+1)
    Mx = 0
    for ims in range(0,M-Mx+1):
        ms = M-ims  # Scale
        l = 2**ms    # Number of points (width) of the averaging segments "nw" at a given scale "m".
        nw = np.int64((2**M)/l)  # Number of segments, each with "l" number of points.
        
        sumab = 0
        
        for i in range(1,nw+1):  #Loop through the different averaging segments "nw" 
            k = (i-1)*l
            za = a[k]
            zb = b[k]
        
            for j in range(k+1,k+l):  #Loop within the datapoints inside one specific [i] segment (ot of the total "nw").
                za = za + a[j]  #Cumulative sum of subsegment "i" in time series "a"
                zb = zb + b[j]  #Cumulative sum of subsegment "i" in time series "b"
            
            za = za/l
            zb = zb/l
            sumab = sumab + (za*zb)
                
            for j in range(k,k+l): # i*l #Subtract the mean from the time series to form the residual to be reused in next iteration. 
                tmpa = a[j] - za
                tmpb = b[j] - zb
                a[j] = tmpa
                b[j] = tmpb
            
        if nw>1: #Computing the MR spectra at a given scale[m]. For scale ms = M is the largest scale.
            D[ms] = (sumab/nw)
    
    #-----------------------------------------------------------------------------------
    # Comparing the Variance with the sum of the Spectra Power:
    
    var = np.var(u1_short)
    MRDvar = np.sum(D)
    # print(var,MRDvar)  
    
    ogive = np.cumsum(D)
    
    #% Graphical Representation of the MRD energy spectr in semilog and loglog form.
    
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.patches as mpatches
    
    # dt = 1/10  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t
    
    E = D
    
    if plot:
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.semilogx(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)
        ax2.loglog(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)
        
        ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
        ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')
        
        ax2.set_ylim((1e-5, 1e0))
        
        # ylim_ax1 = 1.2; ax1.set_ylim(0,ylim_ax1)
        ymin = 1e-3
        ylim_ax2 = 2; ax2.set_ylim(ymin,ylim_ax2)
        
        day = 1/(24*3600)
        twelveHr = 1/(12*3600)
        sixhour = 1/(6*3600)
        halfhour = 1/(1800)
        fivemin = 1/(5*60)
        
        ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
        ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
        ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
        ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
        ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
        
        angle = 90
        ycorner = 1
        l1 = np.array((1e-5, 1e-2))
        th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
                   rotation=angle, rotation_mode='anchor')
        l2 = np.array((2e-5, 1e-2))
        th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
                   rotation=angle, rotation_mode='anchor')
        l3 = np.array((8e-5, 1e-2))
        th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
                   rotation=angle, rotation_mode='anchor')
        l4 = np.array((4e-4, 1e-2))
        th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
                   rotation=angle, rotation_mode='anchor')
        l5 = np.array((2e-3, 1e-2))
        th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
                   rotation=angle, rotation_mode='anchor')
        
        plt.tight_layout()
        plt.show()
    
        fig = plt.figure()
        plt.loglog(f,abs(ogive))
        plt.xlabel('$f\,\,[Hz]$')
        plt.ylabel('Cumulative cospectral sum (Ogive)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return ogive,f,D

lvl = 22
i=5
x_coord = nx_coords[i]
y_coord = ny_coords[i]

o_uu,f,D_uu = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['u_'+str(x_coord)+'_'+str(y_coord)],lvl,True)
o_uv,f,D_uv = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],lvl,False)
o_uw,f,D_uw = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],lvl,False)
o_vv,f,D_vv = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],lvl,False)
o_vw,f,D_vw = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],lvl,False)
o_ww,f,D_ww = MRD_Marc(data_ts['w_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],lvl,False)

# fig = plt.figure()
# plt.loglog(f,abs(o_uu),c='k',label='uu')
# plt.loglog(f,abs(o_uv),c='g',label='uv')
# plt.loglog(f,abs(o_uw),c='y',label='uw')
# plt.loglog(f,abs(o_vv),c='r',label='vv')
# plt.loglog(f,abs(o_vw),c='orange',label='vw')
# plt.loglog(f,abs(o_ww),c='b',label='ww')
# plt.xlabel('$f\,\,[Hz]$')
# plt.ylabel('Cumulative cospectral sum (Ogive)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


#%% Computing the mean large scale Reynolds stress components

M = 16

uu_avg = np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))
vv_avg = np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))
ww_avg = np.mean(data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))
uv_avg = np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))
uw_avg = np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['u_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))
vw_avg = np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)]*data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1)) \
    - np.mean(data_ts['v_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))*np.mean(data_ts['w_'+str(x_coord)+'_'+str(y_coord)][:,0:int(2**M)],axis=(1))

#%%Computing anisotropy invariants from MRD

def computing_invariants(R11,R22,R33,R12,R13,R23):
    # Calculate the TKE
    e = R11 + R22 + R33

    # Define identity matrix
    Id = np.eye(3,3) 

    R = np.matrix([[R11, R12, R13],
                   [R12, R22, R23],
                   [R13, R23, R33]])
   
    # .. calculate he anisotropy tensor
    B = R/e -1/3*Id

    # .. calculate the eigenvalues values
    [eigenVal,eigenVec] = np.linalg.eig(B)
   
    #Sort Eigenvalues in decreasing order
    SortedeigenVal = np.sort(eigenVal)[::-1] 
   
    # .. compute the C coefficients for Barycentric map
    C1c = SortedeigenVal[0] - SortedeigenVal[1]
    C2c = 2*(SortedeigenVal[1] - SortedeigenVal[2])
    C3c = 3*SortedeigenVal[2] + 1
   
    # .. compute the barycentric invariants
    xB = C1c + C3c*1/2
    yB = C3c*np.sqrt(3)/2
       
    return xB,yB #,SortedeigenVal

yB = np.zeros((len(D_uu)-2))
xB = np.zeros((len(D_uu)-2))
SortedeigenVal = dict()

xB_mean, yB_mean = computing_invariants(uu_avg[lvl], vv_avg[lvl], ww_avg[lvl], uv_avg[lvl], uw_avg[lvl], vw_avg[lvl])

for i in range(1,len(D_uu)-1):
    xB[i-1],yB[i-1] = computing_invariants(o_uu[i], o_vv[i], o_ww[i], o_uv[i], o_uw[i], o_vw[i])

#%%Plotting Lumley Triangle trajectories

from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

# Plotting the actual triangle----------------------------------------------------------------------------------------------------------------------

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
# plot a triangle
xc = np.array([0, 1, 0.5])
yc = np.array([0, 0, np.sqrt(3)*0.5])
for i in np.arange(3):
    ip1 = (i+1)%3
    axs.plot([xc[i], xc[ip1]], [yc[i], yc[ip1]], 'k', linewidth=2)
# add grid
nsp = 5
lc = np.abs(xc[1]-xc[0])
dc = lc/nsp
cl = np.zeros([3*(nsp-1), 3])
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        cl[k,i] = dc * (j+1)
        cl[k,ip1] = dc * (nsp-j-1)
xl = np.dot(xc, cl.transpose())
yl = np.dot(yc, cl.transpose())
nl = xl.size
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        kp = (ip1 * (nsp-1) + nsp - j - 2) % nl
        axs.plot([xl[k], xl[kp]], [yl[k], yl[kp]], '--k', linewidth=0.75)
# plain strain limit
c1ps = np.array([2/3, 1/3, 0])
x1ps = np.dot(xc, c1ps.transpose())
y1ps = np.dot(yc, c1ps.transpose())
c2ps = np.array([0, 0, 1])
x2ps = np.dot(xc, c2ps.transpose())
y2ps = np.dot(yc, c2ps.transpose())
axs.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
# add labels
labels = ['2-comp axi', '1-comp', 'Isotropic']
lbpx = xc
dshift = 0.05
lbpy = [yc[0]-dshift*lc, yc[1]-dshift*lc, yc[2]+dshift*lc]
for i in np.arange(3):
    axs.text(lbpx[i], lbpy[i], labels[i], ha='center', fontsize=12)
label_side1 = 'Prolate'
label_side2 = 'Oblate'
label_side3 = 'Two-component'
axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')

#-----------------------------------------------------------------------------------------------------------------------------------------
# sc = axs.scatter(xB[:],yB[:],c=f[1:-1],marker='o',cmap='jet',norm='log')
# axs.plot(xB[:],yB[:],c='k')
# axs.scatter(xB_mean,yB_mean,s=100,c='r',marker='o')
# cbar = plt.colorbar(sc,label='f [Hz]')

lvl = 20

for i in range(len(nx_coords)):
    o_uu,f,D_uu = MRD_Marc(data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    o_uv,f,D_uv = MRD_Marc(data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    o_uw,f,D_uw = MRD_Marc(data_ts['u_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    o_vv,f,D_vv = MRD_Marc(data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    o_vw,f,D_vw = MRD_Marc(data_ts['v_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    o_ww,f,D_ww = MRD_Marc(data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])], data_ts['w_'+str(nx_coords[i])+'_'+str(ny_coords[i])],lvl,False)
    
    yB = np.zeros((len(D_uu)-2))
    xB = np.zeros((len(D_uu)-2))

    for i in range(1,len(D_uu)-1):
        xB[i-1],yB[i-1] = computing_invariants(o_uu[i], o_vv[i], o_ww[i], o_uv[i], o_uw[i], o_vw[i])
    
    sc = axs.scatter(xB[:],yB[:],c=f[1:-1],marker='o',cmap='jet',norm='log',alpha=0.4,edgecolors='none')
    axs.plot(xB[:],yB[:],c='k')
cbar = plt.colorbar(sc,label='f [Hz]')

axs.set_title(f'Height: {(lvl)*dz}m')
axs.axhline(0.38,0,2,c='k',ls='-.')
axs.axhline(0.36,0,2,c='k',ls='-.')
axs.set_xlim(-0.2,1.2)
axs.set_ylim(-0.1,1)
axs.set_xlabel(r'$x_B$',fontsize=15)
axs.set_ylabel(r'$y_B$',fontsize=15)
xticks = [0,0.5,1]
yticks = [0,0.38,np.sqrt(3)/2]
ylabels = ['0','0.38',r'$\sqrt{3}/2$']
axs.set_xticks(xticks)
axs.set_yticks(yticks,labels=ylabels)
# cbar = plt.colorbar(sc,label=r'$z/h_c$')

plt.show()
        

#%%Creating array of xB and yB for all points and for a specific number of heights

lvl_i = 16

xB_dict = np.zeros((15,5*len(nx_coords)))
yB_dict = np.zeros((15,5*len(nx_coords)))

for k in range(0,len(nx_coords)):
    x_coord = nx_coords[k]
    y_coord = ny_coords[k]
    for j in range(lvl_i,lvl_i+5):
        o_uu,f,D_uu = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['u_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_uv,f,D_uv = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_uw,f,D_uw = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_vv,f,D_vv = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_vw,f,D_vw = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_ww,f,D_ww = MRD_Marc(data_ts['w_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        
        yB = np.zeros((len(D_uu)-2))
        xB = np.zeros((len(D_uu)-2))

        for i in range(1,len(D_uu)-1):
            xB[i-1],yB[i-1] = computing_invariants(o_uu[i], o_vv[i], o_ww[i], o_uv[i], o_uw[i], o_vw[i])
            
        xB_dict[:,(j-lvl_i)+k*5] = xB
        yB_dict[:,(j-lvl_i)+k*5] = yB

        #-----------------------------------------------------------------------------------------------------------------------------------------
        sc = axs.scatter(xB[:],yB[:],c=f[1:-1],marker='o',cmap='jet',norm='log',alpha=0.4,edgecolors='none')
        # cbar = plt.colorbar(sc,label='f [Hz]')
        # cbar = plt.colorbar(sc,label=r'$z/h_c$')

        # plt.show()
        
        print(k,j)


#%% Plotting mediand and std of all trajectories for all points

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from operator import add
from operator import sub

# Plotting the actual triangle----------------------------------------------------------------------------------------------------------------------

# plot a triangle
xc = np.array([0, 1, 0.5])
yc = np.array([0, 0, np.sqrt(3)*0.5])
for i in np.arange(3):
    ip1 = (i+1)%3
    axs.plot([xc[i], xc[ip1]], [yc[i], yc[ip1]], 'k', linewidth=2)
# add grid
nsp = 5
lc = np.abs(xc[1]-xc[0])
dc = lc/nsp
cl = np.zeros([3*(nsp-1), 3])
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        cl[k,i] = dc * (j+1)
        cl[k,ip1] = dc * (nsp-j-1)
xl = np.dot(xc, cl.transpose())
yl = np.dot(yc, cl.transpose())
nl = xl.size
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        kp = (ip1 * (nsp-1) + nsp - j - 2) % nl
        axs.plot([xl[k], xl[kp]], [yl[k], yl[kp]], '--k', linewidth=0.75)
# plain strain limit
c1ps = np.array([2/3, 1/3, 0])
x1ps = np.dot(xc, c1ps.transpose())
y1ps = np.dot(yc, c1ps.transpose())
c2ps = np.array([0, 0, 1])
x2ps = np.dot(xc, c2ps.transpose())
y2ps = np.dot(yc, c2ps.transpose())
axs.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
# add labels
labels = ['2-comp axi', '1-comp', 'Isotropic']
lbpx = xc
dshift = 0.05
lbpy = [yc[0]-dshift*lc, yc[1]-dshift*lc, yc[2]+dshift*lc]
for i in np.arange(3):
    axs.text(lbpx[i], lbpy[i], labels[i], ha='center', fontsize=12)
label_side1 = 'Prolate'
label_side2 = 'Oblate'
label_side3 = 'Two-component'
axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')
axs.axhline(0.38,0,2,c='k',ls='-.')
axs.axhline(0.36,0,2,c='k',ls='-.')
axs.set_xlim(-0.2,1.2)
axs.set_ylim(-0.1,1)
axs.set_xlabel(r'$x_B$',fontsize=15)
axs.set_ylabel(r'$y_B$',fontsize=15)
xticks = [0,0.5,1]
yticks = [0,0.38,np.sqrt(3)/2]
ylabels = ['0','0.38',r'$\sqrt{3}/2$']
axs.set_xticks(xticks)
axs.set_yticks(yticks,labels=ylabels)

lvl_i = 16

xB_dict = np.zeros((15,40*len(nx_coords)))
yB_dict = np.zeros((15,40*len(nx_coords)))

for k in range(0,len(nx_coords)):
    x_coord = nx_coords[k]
    y_coord = ny_coords[k]
    for j in range(lvl_i,lvl_i+40):
        o_uu,f,D_uu = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['u_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_uv,f,D_uv = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_uw,f,D_uw = MRD_Marc(data_ts['u_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_vv,f,D_vv = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['v_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_vw,f,D_vw = MRD_Marc(data_ts['v_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        o_ww,f,D_ww = MRD_Marc(data_ts['w_'+str(x_coord)+'_'+str(y_coord)], data_ts['w_'+str(x_coord)+'_'+str(y_coord)],j,False)
        
        yB = np.zeros((len(D_uu)-2))
        xB = np.zeros((len(D_uu)-2))

        for i in range(1,len(D_uu)-1):
            xB[i-1],yB[i-1] = computing_invariants(o_uu[i], o_vv[i], o_ww[i], o_uv[i], o_uw[i], o_vw[i])
            
        xB_dict[:,(j-lvl_i)+k*40] = xB
        yB_dict[:,(j-lvl_i)+k*40] = yB

        #-----------------------------------------------------------------------------------------------------------------------------------------
        sc = axs.scatter(xB[:],yB[:],c=f[1:-1],marker='o',cmap='jet',norm='log',alpha=0.4,edgecolors='none')
        # cbar = plt.colorbar(sc,label='f [Hz]')
        # cbar = plt.colorbar(sc,label=r'$z/h_c$')

        # plt.show()
        
        print(k,j)

axs.plot(np.median(xB_dict,axis=(1)),np.median(yB_dict,axis=(1)),c='k',marker='o')
# axs.fill_between(np.median(xB_dict,axis=(1)), np.median(yB_dict,axis=(1)) - np.std(yB_dict,axis=(1)), np.median(yB_dict,axis=(1)) + np.std(yB_dict,axis=(1)), color='b', alpha=0.5)
axs.set_title(f'All towers, points between: {lvl_i*dz} and {(lvl_i+40)*dz}')

plt.savefig(pathOUT + 'LT_ATTO_r_all_16_56.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()


#%% Autocorrelation, integral length scale and taylor microscale

lvl = 30
time_series = data_ts['u_200_128'][lvl,:]
N = len(time_series)
var = np.var(time_series)
autocorr = np.zeros((int(len(time_series)/2)))
mean = np.mean(time_series)
for j in range(0,N//2):
    if j==0:
        autocorr[j] = np.mean(((time_series-mean)*(time_series-mean)))/var
    else:
        autocorr[j] = np.mean(((time_series[:-j]-mean)*(time_series[j:]-mean)))/var



fig,ax = plt.subplots(1,1,figsize=(6,6),sharex=False,sharey=True)
ax.plot(np.arange(0,len(time_series)//2,1)*dt,autocorr)
ax.axhline(0,0,5)
ax.set_title(f'lvl = {(lvl-5)*dz}m')
ax.set_xlabel(r'$\tau$ [s]')
ax.set_ylabel(r"$\rho$($\tau$)")
# ax.set_xlim(-0.1,3)

plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

def pos_to_neg(timeseries):
    timeseries = np.array(timeseries)
    for i in range(1, len(timeseries)):
        if timeseries[i-1] > 0 and timeseries[i] < 0:
            return i
    return None

indx = pos_to_neg(autocorr)

from scipy.integrate import trapz

int_time = trapz(autocorr[:indx],np.arange(0,indx,1)*dt)
print(f"Integral time scale at height {(lvl-5)*dz}m is {int_time}s.")

#--------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import curve_fit

def parabola(tau, tms):
    return 1 - (tau / tms)**2

def compute_taylor_microscale(autocorr_tau, autocorr_rho):
    popt, pcov = curve_fit(parabola, autocorr_tau, autocorr_rho)
    taylor_microscale = popt[0]
    return taylor_microscale

tms = []

rho = autocorr[:5]
tau = np.arange(0,0.5,0.1)
taylor_microscale = compute_taylor_microscale(tau, rho)
tms.append(taylor_microscale)
print(f"Taylor Microscale at height {(lvl-5)*dz}m:", abs(taylor_microscale))

#%% Second order structure function

def second_order_structure_function(data):

    sf_values = []

    for i in range(1, int(len(data))-10):
        diff = data[i:] - data[:-i]
        sf_values.append(np.mean(diff ** 2))

    return np.array(sf_values)

lvl = 22
time_series = data_ts['u_200_128'][lvl,:]

sf = second_order_structure_function(time_series-np.mean(time_series))
    
    
dt_shift = dict()

dt_shift = np.arange(1,int(len(time_series))-10,1)*dt

plt.rcParams.update({'font.size': 14})
fig,axs = plt.subplots(1,1,figsize=(8,6))

axs.loglog(dt_shift,sf,color='k')
axs.loglog(dt_shift,(dt_shift)**(2/3)*10)
axs.set_xlabel(f'$log(\Delta t)$')
axs.set_title(f'{(lvl-5)*dz}m')
#axs[i].set_xlim(0,0.)
#axs[i].set_ylim(0,2.5)
axs.axhline(2*np.var(time_series),0,len(time_series)*dt,ls='--',color='red',label=f'2$u^{2}$')
axs.legend()
    
axs.set_ylabel(f'<$[\Delta v]^{2}$>')

plt.show()

#%% Fourier Spectra

lvl = 10
time_series = data_ts['u_200_128'][lvl,:]

#Remove the mean from the signal
u_fluc = time_series-np.mean(time_series)
x_ax_1hr = np.arange(0,np.size(time_series))*(1/10)

fig,ax = plt.subplots(1,1,figsize=(12,5),tight_layout=True)
ax.plot(x_ax_1hr,u_fluc)
ax.set_ylabel('u [m/s]')
ax.set_xlabel('Time [s]')
ax.set_xlim(x_ax_1hr[0],x_ax_1hr[-1])
plt.show()


#Modify the signal to make its length equal to the closest power of 2
import math

def Log2(x):
    return (math.log10(x) /
            math.log10(2))
def isPowerTwo(num):
    return (math.ceil(Log2(num)) == math.floor(Log2(num)))

if not isPowerTwo(len(time_series)):
    print('Original signal is not a power of 2.')

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

def padpad(data, iterations = 1):
    narray = data
    for i in range(iterations):
        length = len(narray)
        diff = shift_bit_length(length + 1) - length
        if length % 2 == 0:
            pad_width = diff / 2
        else:
            # need an uneven padding for odd-number lengths
            left_pad = diff / 2
            right_pad = diff - left_pad
            pad_width = (left_pad, right_pad)
        narray = np.pad(narray, int(pad_width), 'constant')
    return narray

#u_pad = padpad(u1hr_fluc,1)

M = np.floor(np.log2(len(u_fluc)))
u_pad = u_fluc[0:int(2**M)]

x_ax_pad = np.arange(0,np.size(u_pad))*dt

fig,ax = plt.subplots(1,1,figsize=(12,5),tight_layout=True)
ax.plot(x_ax_pad,u_pad)
ax.set_ylabel('u [m/s]')
ax.set_xlabel('Time [s]')
ax.set_xlim(x_ax_pad[0],x_ax_pad[-1])
plt.show()


#Window the signal to enforce periodicity
import scipy.signal as sci
window = sci.windows.hann(len(u_pad))
data_win = u_pad*window

fig,ax = plt.subplots(1,1,figsize=(12,5),tight_layout=True)
ax.plot(x_ax_pad,u_pad*window)
ax.set_ylabel('u [m/s]')
ax.set_xlabel('Time [s]')
ax.set_xlim(x_ax_pad[0],x_ax_pad[-1])
plt.show()

#compute fourier coefficients and energy spectra
print('*'*100)
Finv = np.fft.fft(data_win,len(data_win),norm='forward')
E = 2*(abs(Finv)**2)
f = np.fft.fftfreq(len(u_pad),d=dt)
print(f'Variance computed doing the sum of the energy spectra is: {np.sum(E[1:len(E)//2])}')

fig,ax = plt.subplots(1,2,figsize=(14,5),tight_layout=True)
ax[0].loglog(f[:len(f)//2],E[:len(E)//2])
ax[0].set_xlabel('f [Hz]')
ax[0].set_ylabel('E(f)')
ax[0].set_title('Normal Spectra')
ax[1].loglog(f[:len(f)//2],f[:len(f)//2]*E[:len(E)//2])
ax[1].set_xlabel('f [Hz]')
ax[1].set_ylabel('f*E(f)')
ax[1].set_title('Pre-multiplied Spectra')

plt.show()





















