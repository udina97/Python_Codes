#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:12:26 2024

@author: u1450851
"""

def Twr_Anis_Multi(dataNAN,dist,xB_1D,yB_1D,AnisType_1D,Nx,Ny,Nz,Nz_SLayer,coord,d_dim,z_uvp,zi,canopyH,path,name,name2,saveFig=False):
    
    #Libraries and Functions
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/')
    
    from functions import load_3d_UCLAdata, load_UCLAnpy_files, load_2d_UCLAdata
    
    os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
    
    from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc
    
    kappa = 0.4
    phi_m_2D = np.zeros((len(coord),Nz_SLayer))
    z2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    xB_2D = np.zeros((len(coord),Nz),'d',order='F')
    yB_2D = np.zeros((len(coord),Nz),'d',order='F')
    AnisType_2D = np.zeros((len(coord),Nz),'d',order='F')
    z0hi = np.zeros((len(coord)),'d',order='F')
    ustar = np.zeros((len(coord)),'d',order='F')
    z_over_d = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        loc = coord[i]
    
        z = z_uvp
        z_d = (z - ((d_dim[i])/zi))
        z2D[i,:] = z_d[0:Nz_SLayer]
        z_over_d[i,:] = ((z_uvp[0:Nz_SLayer]*zi)-d_dim[i])/39 
        
        from scipy.optimize import curve_fit     
        
        """
        Define a function that will fit the mean velocity with a logarithmic fit
        """
        def log_fit(z, a, b):
            return a*np.log(b*z)


        """
        Define a function that will compute z0hi, and u_star using a logarithmic fit on the mean velocity profile.
        """
        def compute_ustar(Nz_SLayer,z_d,u,v,twr=False):    

            U = np.sqrt(u**2 + v**2)
            U_mean = U[0:Nz_SLayer]

            #Data used for the Inertial logarithmic fit above the RSL. --------------
            #------------------------------------------------------------------------
            level1 = 20 #70
            level2 = 30 #95
            level3 = 70
            level4 = 90

            u1 = U_mean[level1] #This the velocity @ z-d/zi = 0.253
            u2 = U_mean[level2] #This the velocity @ z-d/zi = 0.350
            u3 = U_mean[level3]
            u4 = U_mean[level4]
            
            z_data = np.array([z_d[level1], z_d[level2], z_d[level3], z_d[level4]])#
            U_data = np.array([u1, u2, u3, u4])


            coefs, pcov = curve_fit(log_fit, z_data, U_data)
            u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

            #------------------------------------------------------------------------
            #------------------------------------------------------------------------

            z0hi = (1/coefs[1]) #Normalized values of 'z0hi', hence z0hi/zi.
            ustar = U_mean[level1]/((1/kappa)*np.log(z_d[level1]/z0hi)) #Normalized values of 'ustar', hence u*/uscale.


            return(z0hi,ustar,U_mean,U_data,z_data,u_fit)
        
        [z0hi[i],ustar[i],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                                                                dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],True)
        
        phi_m_2D[i,:] = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                              dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                                  dataNAN['dudz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                                      dataNAN['dvdz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],ustar[i])
        
        xB_2D[i,:] = xB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
        yB_2D[i,:] = yB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
        AnisType_2D[i,:] = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
    
    xB_SLayer = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    yB_SLayer = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    AnisType_SLayer = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        xB_SLayer[i,:] = xB_2D[i,int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        yB_SLayer[i,:] = yB_2D[i,int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        AnisType_SLayer[i,:] = AnisType_2D[i,int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
    z2D_1D = np.ndarray.flatten(z2D)
    yB_2D_1D = np.ndarray.flatten(yB_SLayer)
    AnisType_2D_1D = np.ndarray.flatten(AnisType_SLayer)
    phi_M_1D = np.ndarray.flatten(phi_m_2D)
    
    # phi_m_1D = np.nanmedian(phi_m_2D,axis=(0))
    ## Plotting the Mean wind speed and the corresponding velocity gradient.
    from scipy import stats

    cmap = ColorAnisotropy() 
    
    Nclusters = 9 #Number of clusters used to group the anisotorpy.
    
    fig, axs = plt.subplots(nrows=3,ncols=3)
    plt.ion()
    
    #Loop through all the clusters (1 to 9) organized in a 3x3 subplot..
    
    #Overlay the median on the denisty plot:
    phiM_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); phiM_median.fill(np.NaN)
    yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)
    
    y_ax = z_uvp[0:Nz_SLayer]/(39/zi)
    
    cluster = 0
    for m in range(0,3):
        for n in range(0,3):
            
            cluster = cluster + 1
            
            print(f'cluster = {cluster}')
    
            tmp_z3D = z2D_1D[(AnisType_2D_1D == cluster)]
            tmp_phi_u = phi_M_1D[(AnisType_2D_1D == cluster)]
            tmp_yB = yB_2D_1D[(AnisType_2D_1D == cluster)]
            
            NumPoints = np.size(tmp_yB) #Total number of points in a given cluster. This should be larger than 100 to do statistics.
            
            print(f'Total number of points in cluster = {NumPoints}')
    
            if (cluster <= 9 and NumPoints > 100):
                x = tmp_phi_u
                y = tmp_z3D/(39/zi)
        
    
                #Developing a "2D Density plot" to better visualize where there are more points.
                tmp = np.array([x,y])
                nbins = 40
    
                # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
                k = stats.gaussian_kde(tmp)
                xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
                density = k(np.vstack([xi.flatten(), yi.flatten()]))
     
                # plot the density with shading
                #axs.set_title('2D Density with shading')
                axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')
    
        
            for i in range(0,Nz_SLayer-1):
                phiM_median[cluster-1,i] = np.median(tmp_phi_u[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])
                yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])     
        
    
            print(f'Done with cluster {cluster}')
            
            if (np.size(tmp_phi_u) > 0):
                sc = axs[m,n].scatter(phiM_median[cluster-1,0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[cluster-1,0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    
    #------------------------------------------------------------------------------
    
    
            axs[m,n].set_xlim(-0.5, 2.5)
            axs[m,n].set_ylim(y_ax[0], y_ax[-1])
            axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
            axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
            axs[m,n].axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
            
         
            axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
            axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
            axs[m,n].minorticks_on()
        
    
            axs[m,n].set_xscale('linear')
            axs[m,n].set_title(f'cluster = {cluster}')
        
        
    cbar = plt.colorbar(sc)
        
    axs[0,0].set_ylabel(r'$z/h$')
    axs[1,0].set_ylabel(r'$z/h$')
    axs[2,0].set_ylabel(r'$z/h$')
    
    axs[2,0].set_xlabel(r'$\phi_M$')
    axs[2,1].set_xlabel(r'$\phi_M$')
    axs[2,2].set_xlabel(r'$\phi_M$')
    
    
    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name +'.png',dpi=300,facecolor='white', edgecolor='white')
    
    fig2, axs = plt.subplots(nrows=1,ncols=1)
    plt.ion()

    #Median of the gradient profiles as a function of cluster and height. 
    #For a fixed cluster (e.g. cluster = 4), we average all values of the gradient
    #at a specific height.
     
    for cluster in range(1,Nclusters):
        #print(f'{np.count_nonzero(np.isnan(yB_median[cluster,0:-1]))}')
        if (np.count_nonzero(np.isnan(yB_median[cluster,0:-1])) < Nz_SLayer-1):
            sc = axs.scatter(phiM_median[cluster,0:-1],(y_ax[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

    axs.plot(np.nanmean(phi_m_2D[:,:],axis=(0)),(y_ax[0:Nz_SLayer]),linestyle='-',color='k')

    axs.set_xlim(-0.5, 2.5)
    axs.set_ylim(y_ax[0], y_ax[Nz_SLayer-1])
    #axs.axhline(y = (canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle=':')
    #axs.axhline(y = 3*(canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')
    axs.axhline(y = (canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle=':')
    axs.axhline(y = 3*(canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle='-.')

    axs.set_xscale('linear')
    cbar = plt.colorbar(sc)
    #axs.set_ylabel(r'$(z-d)/h$')
    axs.set_ylabel(r'$z/h$')
    axs.set_xlabel(r'$\phi_M$')
    # axs.set_title(f'{cases[num]}')

    axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
    axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')

    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name2 +'.png',dpi=300,facecolor='white', edgecolor='white')
    
    h1 = np.min(z_over_d)
    z_over_d_m = []
    phi_m_v2 = []
    for i in range(0,int(np.ceil(np.max(z_over_d)/0.5))):
        z_over_d_m.append(np.median(z_over_d.flatten()[(z_over_d.flatten()>h1) & (z_over_d.flatten()<h1+0.5)]))
        phi_m_v2.append(np.median(phi_m_2D.flatten()[(z_over_d.flatten()>h1) & (z_over_d.flatten()<h1+0.5)]))
        h1 = h1+0.5
    # z_over_d_m = 
        
    # fig3, axs = plt.subplots(1,1,tight_layout=True)
    
    # axs.plot
    
    return fig, fig2, z0hi, ustar, z_over_d, z_over_d_m, phi_m_v2

#--------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_topo_twr(zi,Nx,Ny,dx,dy,intf,coord,textsize):
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0, Nx) * dx
    y = np.arange(0, Ny) * dy
    fig, ax = plt.subplots()
    contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
    for i in range(len(coord)):
        ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
    # contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
    plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
    plt.xlabel('$x$', usetex=True)
    plt.ylabel('$y$', usetex=True)
    plt.colorbar(contour)
    ax.set_aspect('auto')
    plt.gca().set_facecolor('white')
    # plt.savefig(figPath+'topo.png',dpi=300,facecolor='white', edgecolor='white')
    
    return fig

#-------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_d_twr(dataNAN,coord,dist,height,dz,zi,u_scale,LAD,topo='flat'):
    import numpy as np
    from scipy.integrate import trapz
    
    d_dim = np.zeros(len(coord))
    for i in range(len(coord)):
        loc = coord[i]
        z_nan = dist[loc[0],loc[1],:]
        
        u = dataNAN['u'][loc[0],loc[1],int(np.where(z_nan>0)[0][0]):]
        v = dataNAN['v'][loc[0],loc[1],int(np.where(z_nan>0)[0][0]):]
        U = np.sqrt(u**2+v**2)
        
        prof2 = {}
        prof2['U'] = U #mean_U
        if topo=='flat':
            Z = np.linspace(1,height-5,10)*dz*zi - (dz*zi)/2
        else:
            Z = np.linspace(1,height-5,16)*dz*zi - (dz*zi)/2
        VAR = prof2['U'][0:height-5]*u_scale
        Y = (VAR**2)*0.4*LAD
        d_dim[i] = trapz(Y*Z,Z)/trapz(Y,Z)
        
    return d_dim

#--------------------------------------------------------------------------------------------------------------------------------------------------

def Twr_TKE_Multi(Nx,Ny,Nz,Nz_SLayer,z_on_h,coord,xB_1D,yB_1D,AnisType_1D,TKE_term,dist,Nclusters,labels,p,path,name,name2,z_uvp,zi,d_dim,saveFig=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy import stats
    
    os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
    
    from Anisotropy_Functions import ColorAnisotropy
    
    z2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    TKE_term_2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    yB_2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    AnisType_2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    z_over_d = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        loc = coord[i]
        z2D[i,:] = z_on_h[0:Nz_SLayer]
        z_over_d[i,:] = (z_uvp[0:Nz_SLayer]*zi)/d_dim[i] 
        TKE_term_2D[i,:] = TKE_term[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        yB_2D[i,:] = yB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        AnisType_2D[i,:] = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]

    z2D_1D = np.ndarray.flatten(z2D)
    yB_2D_1D = np.ndarray.flatten(yB_2D)
    AnisType_2D_1D = np.ndarray.flatten(AnisType_2D)
    TKE_term_2D_1D = np.ndarray.flatten(TKE_term_2D)
    #----------------------------------------

    #Overlay the median on the denisty plot:
    TKE_term_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); TKE_term_median.fill(np.NaN)
    yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)

    fig, axs = plt.subplots(nrows=1,ncols=1)
    plt.ion()

    cmap = ColorAnisotropy()  

    cluster = 0
    for cluster in range(1,10):
            
        tmp_TKE_term_1D = TKE_term_2D_1D[(AnisType_2D_1D == cluster)]
        tmp_z3D = z2D_1D[(AnisType_2D_1D == cluster)]
        tmp_yB = yB_2D_1D[(AnisType_2D_1D == cluster)]
        
        for i in range(0,Nz_SLayer-1):
            TKE_term_median[cluster-1,i] = np.median(tmp_TKE_term_1D[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])
            yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])     
        
        print(f'Done with cluster {cluster}')
            
        if (np.size(tmp_TKE_term_1D) > 0):
            sc = axs.scatter(TKE_term_median[cluster-1,:],z_on_h[0:Nz_SLayer],s=10,marker='o',c = yB_median[cluster-1,:],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

    # axs.plot(np.nanmean(TKE_term[:,:,0:Nz_SLayer],axis=(0,1)),z_on_h[0:Nz_SLayer],'-k')

    #------------------------------------------------------------------------------

    x_min = -100
    x_max = 100

    axs.set_xlim(x_min, x_max)
    axs.set_ylim(z_on_h[0], z_on_h[Nz_SLayer])
    axs.axhline(y = 1, xmin=x_min, xmax=x_max,color='gray',linestyle='--')
    #axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
    axs.axvline(x =0, ymin = 0, ymax=13,color='gray',linestyle='--')
            
    axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
    axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
    axs.minorticks_on()
    axs.set_xscale('linear')
        
    cbar = plt.colorbar(sc)
        
    axs.set_ylabel(r'$z/h$')
    axs.set_xlabel(f'{labels[p]}')

    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name +'.png',dpi=300,facecolor='white', edgecolor='white')
    
    fig2, axs = plt.subplots(nrows=3,ncols=3)
    plt.ion()
    
    y_ax = z_on_h[0:Nz_SLayer]
    
    cluster = 0
    for m in range(0,3):
        for n in range(0,3):
            
            cluster = cluster + 1
            
            print(f'cluster = {cluster}')
    
            tmp_z3D = z2D_1D[(AnisType_2D_1D == cluster)]
            tmp_TKE_term_1D = TKE_term_2D_1D[(AnisType_2D_1D == cluster)]
            tmp_yB = yB_2D_1D[(AnisType_2D_1D == cluster)]
            
            NumPoints = np.size(tmp_yB) #Total number of points in a given cluster. This should be larger than 100 to do statistics.
            
            print(f'Total number of points in cluster = {NumPoints}')
    
            if (cluster <= 9 and NumPoints > 100):
                x = tmp_TKE_term_1D
                y = tmp_z3D
        
                #Developing a "2D Density plot" to better visualize where there are more points.
                tmp = np.array([x,y])
                nbins = 40
    
                # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
                k = stats.gaussian_kde(tmp)
                xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
                density = k(np.vstack([xi.flatten(), yi.flatten()]))
     
                # plot the density with shading
                #axs.set_title('2D Density with shading')
                axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')  
        
            print(f'Done with cluster {cluster}')
            
            if (np.size(tmp_TKE_term_1D) > 0):
                sc = axs[m,n].scatter(TKE_term_median[cluster-1,0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[cluster-1,0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    
    #------------------------------------------------------------------------------
    
            axs[m,n].set_xlim(-60, 60)
            axs[m,n].set_ylim(y_ax[0], y_ax[-1])
            axs[m,n].axhline(y = 1, xmin=-60, xmax=80,color='gray',linestyle='--')
            axs[m,n].axhline(y = 3, xmin=-60, xmax=80,color='gray',linestyle='-.')
            axs[m,n].axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
            
            axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
            axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
            axs[m,n].minorticks_on()
        
            axs[m,n].set_xscale('linear')
            axs[m,n].set_title(f'cluster = {cluster}')
        
    cbar = plt.colorbar(sc)
        
    axs[0,0].set_ylabel(r'$(z)/h$')
    axs[1,0].set_ylabel(r'$(z)/h$')
    axs[2,0].set_ylabel(r'$(z)/h$')
    
    axs[2,0].set_xlabel(f'{labels[p]}')
    axs[2,1].set_xlabel(f'{labels[p]}')
    axs[2,2].set_xlabel(f'{labels[p]}')
    
    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name2 +'.png',dpi=300,facecolor='white', edgecolor='white')
        
    h1 = 0
    z_over_d_m = []
    tke_m = []
    for i in range(0,int(np.ceil(np.max(z_over_d)/0.5))):
        z_over_d_m.append(np.median(z_over_d.flatten()[(z_over_d.flatten()>h1) & (z_over_d.flatten()<h1+0.5)]))
        tke_m.append(np.median(TKE_term_2D.flatten()[(z_over_d.flatten()>h1) & (z_over_d.flatten()<h1+0.5)]))
        h1 = h1+0.5
    
    return fig,fig2,z_over_d,z_over_d_m,tke_m

#--------------------------------------------------------------------------------------------------------------------------------------------------------

def find_coordinates(elevation_map,N_twrs,loc='max'):
    '''
    Finds the coordinates on either the peaks or the the lowest points in a topography.
    Selects a number of those points based on N_twrs.

    Parameters
    ----------
    array : intf, topography height
    N_twrs : number of towers I want
    loc : max=peaks, min=valley floor

    Returns
    -------
    coordinates : coordinates of the towers

    '''
    import random
    import numpy as np
    
    peaks = []
    rows = len(elevation_map)
    cols = len(elevation_map[0])
    
    if loc=='max':
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                is_peak = True
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if elevation_map[i][j] < elevation_map[i + di][j + dj]:
                        is_peak = False
                        break
                if is_peak:
                    if elevation_map[i][j]>(9/10)*np.max(elevation_map):
                        peaks.append((i, j))
    else:
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                is_peak = True
                if elevation_map[i][j] >= 2*np.min(elevation_map):
                    is_peak = False
                if is_peak:
                    peaks.append((i, j))
    peaks = random.sample(peaks,N_twrs)
    return peaks
        
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def box_plot(Nx,Ny,Nz,Nz_SLayer,TKE_term,coord,AnisType_1D,dist,labels,n,z_on_h,level,dz,path,name,name2,saveFig=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    TKE_term_2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    AnisType_2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    z2D = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        loc = coord[i]
        z2D[i,:] = z_on_h[0:Nz_SLayer]
        TKE_term_2D[i,:] = TKE_term[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        AnisType_2D[i,:] = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]

    AnisType_2D_1D = np.ndarray.flatten(AnisType_2D)
    TKE_term_2D_1D = np.ndarray.flatten(TKE_term_2D)
    z2D_1D = np.ndarray.flatten(z2D)

    TKE_cluster = {}  #Note, I am not using a pd.DataFrame because in there all columns need to be the same length.
                      #I could  make the columns of same lengths by adding NaNs but that would increase the memory needs.

    nPoints_Cluster = {} #dictionary that keeps track of the total number of points per cluster
    Total_nPoints = 0
    cluster = 0
    for cluster in range(1,10):
        print(f'cluster = {cluster}')
        #tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster)]
        cluster_name = 'cluster ' + str(cluster)
        TKE_cluster[cluster_name] = TKE_term_2D_1D[(AnisType_2D_1D == cluster) & (z2D_1D>=level) & (z2D_1D<=level+2*dz)]
        nPoints_Cluster[cluster_name] = TKE_cluster[cluster_name].size
        Total_nPoints = Total_nPoints + nPoints_Cluster[cluster_name] 

    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in TKE_cluster.items()]))
    
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(8, 5))

    #Seaborn Boxplot Plot:
    cm = sns.color_palette( ["#410d00","#831901","#983e00","#b56601","#ab8437",
                             "#b29f74","#7f816b","#587571","#596c72"])

    sns.boxplot(data=df,orient='h',palette=cm,saturation=1,width=0.6,linewidth=0.8,fliersize = 0.5)

    x_min = -200
    x_max = 200

    axs.set_xlim(x_min, x_max)
    axs.axvline(x =0, ymin = 0, ymax=9,color='gray',linestyle='--')
                 
    axs.grid(which='major', axis='both',color='grey', alpha = 0.3, linestyle='-')
    axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
    axs.minorticks_on()
    axs.set_xscale('linear')
        
    #cbar = plt.colorbar(sc)
        
    axs.set_ylabel(r'$cluster$')
    axs.set_xlabel(f'{labels[n]}')

    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name +'.png',dpi=300,facecolor='white', edgecolor='white')
    
    percentage_Cluster = (pd.Series(nPoints_Cluster.values())/Total_nPoints)*100

    fig2, axs = plt.subplots(nrows=1,ncols=1,figsize=(6, 3))

    axs.scatter(np.arange(1,10),percentage_Cluster,c=["#410d00","#831901","#983e00","#b56601","#ab8437",
                             "#b29f74","#7f816b","#587571","#596c72"])

    axs.set_xlabel(r'$cluster$')
    axs.set_ylabel(r'$N_i/N_{total}$')

    plt.tight_layout()
    
    if saveFig:
        plt.savefig(path +'Figures/'+ name2 +'.png',dpi=300,facecolor='white', edgecolor='white')
    
    return fig,fig2

#------------------------------------------------------------------------------------------------------------------------------------------------------

def ChameckiIndex(Nz_SLayer,coord,dist,Adv,TotDis):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    AdvTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    TotDisTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    Ia = np.zeros((Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        
        loc = coord[i]
        
        AdvTwr[i,:] = Adv[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        TotDisTwr[i,:] = TotDis[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        
    Ia = np.mean(np.abs(AdvTwr),axis=0)/np.mean(TotDisTwr,axis=0) 
    
    return Ia

#------------------------------------------------------------------------------------------------------------------------------------------

def compute_ustar_twr(coord,U,V,Nz_SLayer,d_dim,zi,dz,dist):
    
    from scipy.optimize import curve_fit     
    import numpy as np
    
    kappa = 0.41
    ustar_twr = np.zeros((len(coord)),'d',order='F')
    z0hi_twr = np.zeros((len(coord)),'d',order='F')

    # Nz_SLayer = Nz-1
    """
    Define a function that will fit the mean velocity with a logarithmic fit
    """
    def log_fit(z, a, b):
        return a*np.log(b*z)


    """
    Define a function that will compute z0hi, and u_star using a logarithmic fit on the mean velocity profile.
    """
    def compute_ustar(Nz_SLayer,z_d,u,v):    

        U = np.sqrt(u**2 + v**2)
        # U_mean = np.nanmean(U[:,:,:Nz_SLayer],axis=(0,1))
        U_mean = U[:Nz_SLayer]

        #Data used for the Inertial logarithmic fit above the RSL. --------------
        #------------------------------------------------------------------------
        level1 = 70 #20 #70
        level2 = 95 #30 #95
        # level3 = 70
        # level4 = 90

        u1 = U_mean[level1] #This the velocity @ z-d/zi = 0.253
        u2 = U_mean[level2] #This the velocity @ z-d/zi = 0.350
        # u3 = U_mean[level3]
        # u4 = U_mean[level4]
        
        z_data = np.array([z_d[level1], z_d[level2]])#, z_d[level3], z_d[level4]])#
        U_data = np.array([u1, u2])#, u3, u4])


        coefs, pcov = curve_fit(log_fit, z_data, U_data)
        # pos_zd = np.maximum(z_d[0:Nz_SLayer], 1e-10)
        u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------

        z0hi = (1/coefs[1]) #Normalized values of 'z0hi', hence z0hi/zi.
        ustar = U_mean[level1]/((1/kappa)*np.log(z_d[level1]/z0hi)) #Normalized values of 'ustar', hence u*/uscale.


        return(z0hi,ustar,U_mean,U_data,z_data,u_fit)
    
    for i in range(len(coord)):
        loc = coord[i]
        z = np.arange(0,Nz_SLayer)*dz + 0.5*dz
        z_d = (z - ((d_dim[i])/zi))
        [z0hi_twr[i],ustar_twr[i],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer, z_d, \
                                                                              U[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):], \
                                                                              V[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):])
        
    return z0hi_twr, ustar_twr
    
        

    