#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:13:39 2026

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L

#%%1D interpolation from w node to uvp node

def wnode2uvpnode1D(var):
    import numpy as np
    out_var = np.zeros(var.shape)
    out_var[:-1] = 0.5*(var[:-1]+var[1:])
    out_var[-1] = var[-1]
    return out_var
    

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz
dt = 0.1

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

nt_start = 99000
steps = 72000
sfc = 'Homog' #Patch
most = 'classic'
Ug = 1
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sfc+'/'+str(nz)+'/twr_ts/'+'homo'+str(nz)+'_'+most+'_'+str(Ug)+'/'

#%%Load time series for the virtual towers

x_coord = [nx//4,int(3*nx/4)]
y_coord = [ny//4,int(3*ny/4)]
var = ['u','v','w','sc']

twr_prof = dict()

for i in range(len(x_coord)):
    for j in range(len(y_coord)):
        for k in range(len(var)):

            # filename = path+'output_twr/twr_ts_0'+str(x_coord[i])+'_0'+str(y_coord[i])+'_'+var[k]+'.out'
            filename = f"{path}output_twr/twr_ts_{x_coord[i]:03d}_{y_coord[j]:03d}_{var[k]}.out"
            
            iterations = []
            profiles = []
            seen_iterations = set()
            
            with open(filename, "rb") as f:
                while True:
                    # Read iteration number (int32)
                    iter_bytes = f.read(4)
                    if not iter_bytes:
                        break
                    
                    jj = np.frombuffer(iter_bytes, dtype=np.int32)[0]
                    
                    # Read profile (float64)
                    prof = np.fromfile(f, dtype=np.float64, count=nz)
                    
                    if prof.size < nz:
                        break
                    
                    if jj not in seen_iterations:
                        seen_iterations.add(jj)
                        iterations.append(jj)
                        profiles.append(prof)
            
            iterations = np.array(iterations)
            profiles = np.array(profiles)
            
            twr_prof[str(x_coord[i])+'_'+str(y_coord[j])+'_'+var[k]] = profiles
            
            print("Read", len(iterations), "time steps")
            print("Profiles shape:", profiles.shape)
            
#%%Interpolate w to uvp node

for n in range(steps):
    for i in range(len(x_coord)):
        for j in range(len(y_coord)):
            twr_prof[str(x_coord[i])+'_'+str(y_coord[j])+'_w'][n,:] = wnode2uvpnode1D(twr_prof[str(x_coord[i])+'_'+str(y_coord[j])+'_w'][n,:])

#%%Plot surface temperature and tower coordinates

sfcT = read_checkpoint_sfc_L(path+'output_checkpoint/',[10000],nx,ny)['sfcVAL']*Tscale

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,sfcT.T,cmap='hot_r',vmin=285,vmax=295)
for i in range(len(x_coord)):
    for j in range(len(y_coord)):
        axs.scatter(x_coord[i]*dx,y_coord[j]*dy,s=10,c='k')
cbar = plt.colorbar(p)
axs.set_xlabel(r"$x/x_i$",fontsize=14)
axs.set_ylabel(r"$y/y_i$",fontsize=14)
axs.set_title(r"Surface T",fontsize=12)

plt.show()

#%%Plot median and iq range of time series profile for a coordinate

x_c = x_coord[0]
y_c = y_coord[0]

fig,axs = plt.subplots(1,4,tight_layout=True,sharey=True,figsize=(8,4))

for k in range(len(axs)):
    axs[k].plot(np.median(twr_prof[str(x_c)+'_'+str(y_c)+'_'+var[k]],axis=(0)),z_uvp)
    axs[k].fill_betweenx(z_uvp,np.quantile(twr_prof[str(x_c)+'_'+str(y_c)+'_'+var[k]],0.25,axis=(0)),np.quantile(twr_prof[str(x_c)+'_'+str(y_c)+'_'+var[k]],0.75,axis=(0)),
                      alpha=0.3)
    axs[k].set_xlabel(f"{var[k]}",fontsize = 14)

axs[0].set_ylabel(r"$z/z_i$",fontsize = 14)
axs[0].set_ylim(0,z_uvp[-1])
fig.suptitle(f"x: {x_c} - y: {y_c}",fontsize=12)

plt.show()

#%%Plot time series at a specific height for the 4 coordinates

z_lvl = 3
v = var[0]
c = ['k','g','r','b']
time = np.arange(0,steps)*dt/60

fig,axs = plt.subplots(1,1,figsize=(7,3),tight_layout=True)

k = 0
for i in range(len(x_coord)):
    for j in range(len(y_coord)):
        axs.plot(time,twr_prof[str(x_coord[i])+'_'+str(y_coord[j])+'_'+v][:,z_lvl],c=c[k],label=f"{x_coord[i]}_{y_coord[j]}")
        k+=1
        
axs.set_xlabel(r"Time [min]", fontsize = 14)
axs.set_ylabel(f"{v}",fontsize = 14)
axs.set_xlim(0,time[-1])
axs.set_title(f"Height: {z_lvl*dz*zi + dz*zi/2}m", fontsize=14)
axs.legend()


plt.show()

#%%MRD analysis

def MRD_decomp(v1,v2,c1,c2,lvl):
    
    # We begin by importing the Libraries that we will need to 

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import xarray as xr


    #Loading the Data to be analyzed: ------------------------------------------------

    u1 = twr_prof[str(c1)+'_'+str(c2)+'_'+v1][:,lvl]
    u2 = twr_prof[str(c1)+'_'+str(c2)+'_'+v2][:,lvl]

    # At the same time, and for the sake of simplifying the analysis, we Rensure there are no NaNs in the Signal.
    # If we encounter any, we set them to zero. Note, that when you analyze your data you might/could decide using 
    # more advnaced methods to fill in the gaps. Clearly, depending on what you do, it might affect the outcome
    # of your analysis.

    testNan = np.isnan(u1)
    ind = np.where(testNan == True)
    u1[ind] = 0
    testNan = np.isnan(u2)
    ind = np.where(testNan == True)
    u2[ind] = 0

    Number_of_Nans = np.size(ind) #Determines how many NaNs have been found.
    # Next, we define a couple parameters related to the signal.print(f"The total number of NaNs found is of {Number_of_Nans} points")

    #Make the Signal a power of 2.
    M = np.int64(np.floor(np.log2(len(u1)))) #Maximum Power of 2 within the length of the signal measure at 20Hz.
    u1_short = u1[0:int(2**M)]
    u2_short = u2[0:int(2**M)]

    #-----------------------------------------------------------------------------------
    # var1 = u1_short
    # var2 = u2_short
    
    u1_short = u1_short - np.mean(u1_short)
    u2_short = u2_short - np.mean(u2_short)
    
    var1 = u1_short.copy()
    var2 = u2_short.copy()
    
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
                
            for j in range(k,i*l): #Subtract the mean from the time series to form the residual to be reused in next iteration. 
                tmpa = a[j] - za
                tmpb = b[j] - zb
                a[j] = tmpa
                b[j] = tmpb
            
        
        if nw>1: #Computing the MR spectra at a given scale[m]. For scale ms = M is the largest scale.
            D[ms] = (sumab/nw)

    #-----------------------------------------------------------------------------------
    # Comparing the Variance with the sum of the Spectra Power:

    covar = np.mean(u1_short*u2_short)
    MRDvar = np.sum(D)
    print(covar,MRDvar)  

    
    #% Graphical Representation of the MRD energy spectr in semilog and loglog form.

    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.patches as mpatches


    dt = 1/10  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t

    E = D

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.semilogx(f[1:-1],abs(E[1:-1]),color='black',linewidth=8,alpha=0.4)
    ax2.loglog(f[1:-1],abs(E[1:-1]),color='black',linewidth=8,alpha=0.4)

    ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
    ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

    ax2.set_ylim((1e-5, 1e0))

    # ylim_ax1 = 1.2e-3; ax1.set_ylim(0,ylim_ax1)
    ymin = 1e-5
    ylim_ax2 = 2e-3; ax2.set_ylim(ymin,ylim_ax2)

    day = 1/(24*3600)
    twelveHr = 1/(12*3600)
    sixhour = 1/(6*3600)
    halfhour = 1/(1800)
    fivemin = 1/(5*60)
    
    # ax1.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    # ax1.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax1.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax1.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax1.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

    # ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    # ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
    ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

    angle = 90
    ycorner = 1
    # l1 = np.array((1e-5, 1e-2))
    # th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
    #            rotation=angle, rotation_mode='anchor')
    # l2 = np.array((2e-5, 1e-2))
    # th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
    #            rotation=angle, rotation_mode='anchor')
    l3 = np.array((8e-5, 1e-2))
    th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    h2 = ax1.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    l4 = np.array((4e-4, 1e-2))
    th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    th2 = ax1.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    l5 = np.array((2e-3, 1e-2))
    th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    th2 = ax1.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
               rotation=angle, rotation_mode='anchor')
    fig.suptitle(f"{v1} - {v2}", fontsize=12)

    plt.tight_layout()
    plt.show()

for i in range(len(x_coord)):
    for j in range(len(y_coord)):
        mrd = MRD_decomp('w','sc',x_coord[i],y_coord[j],1)




































