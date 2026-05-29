#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:30:31 2024

@author: u1450851
"""

def MRD_decomp():
    
    # We begin by importing the Libraries that we will need to 

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import xarray as xr


    #Loading the Data to be analyzed: ------------------------------------------------

    # Include the path to where you have locally stored the data file.
    file_path = "/Users/mcalaf/Documents/Utah/Courses/Summer_school_Norway_2022/Lecture1/"

    os.chdir(file_path)
    
    # Next, we load the information in the data file onto the variable Data. This variable is a dataarray, 
    # with 2 dimensions (.dims) and 4 coordinates (.coords, u,v,w,T)

    Data = xr.open_dataarray("SonicData.nc")

    # To simplify the analysis, we will focus for now only on the streamwise velocity, u.
    u = Data.data[:,0]

    # At the same time, and for the sake of simplifying the analysis, we Rensure there are no NaNs in the Signal.
    # If we encounter any, we set them to zero. Note, that when you analyze your data you might/could decide using 
    # more advnaced methods to fill in the gaps. Clearly, depending on what you do, it might affect the outcome
    # of your analysis.

    testNan = np.isnan(u)
    ind = np.where(testNan == True)
    u[ind] = 0

    Number_of_Nans = np.size(ind) #Determines how many NaNs have been found.
    # Next, we define a couple parameters related to the signal.print(f"The total number of NaNs found is of {Number_of_Nans} points")

    #Make the Signal a power of 2.
    M = np.int64(np.floor(np.log2(len(u)))) #Maximum Power of 2 within the length of the signal measure at 20Hz.
    u_short = u[0:int(2**M)]

    #-----------------------------------------------------------------------------------
    var1 = u_short
    var2 = u_short
    
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

    var = np.var(u_short)
    MRDvar = np.sum(D)
    print(var,MRDvar)  

    
    #% Graphical Representation of the MRD energy spectr in semilog and loglog form.

    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.patches as mpatches


    dt = 1/20  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t

    E = D

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.semilogx(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)
    ax2.loglog(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)

    ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
    ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

    ax2.set_ylim((1e-5, 1e0))

    ylim_ax1 = 1.2; ax1.set_ylim(0,ylim_ax1)
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
    