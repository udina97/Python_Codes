"""
Program Author: Marc Calaf.

This program uses Giulia's LES data, to compute vertical gradients 
of the mean velocity as a function of z/z* in neutral conditions, and turbulence anisotropy.

Date created: 25 October 2022
Last date modified: 25 October 2022

To Do: ... Add the computation of turbulence anisotropy to color the dots on the gradient scatter plot.

"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr


os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


from Anisotropy_Functions import ColorAnisotropy, PhiM_Data_Neutral
from Stats import  ReynoldsStress

#%% Defining the path to the files:

#Giulia's Data:
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2022/LES_VegCanopy_Data/RAV_g400_1mps_ug/zcopy/'
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2022/LES_VegCanopy_Data/RAV_g400_9mps_ug/zcopy/'
#path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2022/LES_VegCanopy_Data/hom_9/'


directory = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/'
cases = ['RAV_hom-eq_9mps_ug','RAV_g1200_9mps_ug','RAV_g1200_inv_9mps_ug','RAV_g800_9mps_ug','RAV_g800i_9mps_ug','RAV_g400_9mps_ug']
num = 2 

path = directory + cases[num] +'/'

os.chdir(path)  


#%% Defining the main parameters of the simulations

NumVariables = 26
NumVariablesSC = 10

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

#Tscale = 290 #[K]
#u_scale = 0.45 #[m/s]
zi = 1000 #[m]

Nz_SLayer = 128 #This is the grid point that corresponds to about 100m height, surface alyer.

#For the sake of clarity, below I sepcify the variables included in "data" and "dataS":
#--------------------------------------------------------------------------------------------------
#data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                        'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                        'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                        'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                        'avgdudz','avgdvdz','avgNut','avgCs']})

#dataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
#                        'avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
#                        'avg_ds']})


#data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
#                dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})

#dataS_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
#                dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
#                'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})


#%% Loading the averaged Post-Processed Output_RAV data from the NetCDF files:


data = xr.open_dataarray('Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#dataS = xr.open_dataarray('Data_Scalar.nc')

data_2D = xr.open_dataarray('Data_Momentum_2D.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#dataS_2D = xr.open_dataarray('Data_Scalar_2D.nc')


#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor and Sensible Heat fluxes:

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                         data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])




#%% Running the PhiM_Data_Neutral Function, that computes the relevant variables for the Phi_M Analysis in neutral conditions

PhiM_AnisData_Neutral = xr.DataArray(np.ones(shape = (Nx*Ny*Nz_SLayer,5),order='C'),\
                    dims=('space','variable'), coords = {'variable':['phi_M_1D',\
                    'yB_1D','xB_1D','AnisType_1D','z3D_1D']})

xB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
yB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    
[PhiM_AnisData_Neutral,yB,xB] = PhiM_Data_Neutral(Nx,Ny,Nz,Nz_SLayer,dz,zi,NumVariables,data,data_2D)


#%% Save the Phi_M_Neutral data into a file:

os.chdir(path)
PhiM_AnisData_Neutral.to_netcdf('PhiM_AnisData_Neutral.nc')

anisotropy = xr.DataArray(np.ones(shape = (Nx,Ny,Nz_SLayer,2),order='C'),\
                        dims=('x','y','z','variable'), coords = {'variable':['xB','yB']})

anisotropy[:,:,:,0] = xB
anisotropy[:,:,:,1] = yB

anisotropy.to_netcdf('anisotropy.nc')

#%% Load Phi_M data from file:

    
# Note: PhiU has been computed using u* and dUdz local at each grid point. u* is a plane at h = canopyH +1
# Note: dUdz is a 3D matrix. < kz/u* dudz>_{xy}.
# Note: The velocity gradient is computed locally, like the u*.
# Note: We are using u* computed at one grid point above the vegetated canopy. 
# Note: u* is computed using the Reynolds stress and the SGS contribution.


    
NumFiles = 1

PhiM_AnisData_Neutral = xr.DataArray(np.ones(shape = (Nx*Ny*Nz_SLayer,5),order='C'),\
                    dims=('space','variable'), coords = {'variable':['phi_M_1D',\
                    'yB_1D','xB_1D','AnisType_1D','z3D_1D']})


    
#cases = ['hom_9'] #LES velocity cases    
    
for n in range(0,NumFiles):    
    
    #path = f"/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2022/LES_VegCanopy_Data/{cases[n]}/"
    #path = f"/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_g1200_inv_9mps_ug/"
    #path = f"/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_g1200_9mps_ug/"
    #path = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/RAV_hom-eq_9mps_ug/'
    print(f'Reading from file {path}')

    PhiM_AnisData = xr.open_dataarray(path + 'PhiM_AnisData_Neutral.nc')

    
    if (n == 0):
        
        phi_M_1D = np.copy(PhiM_AnisData.data[:,0]) 
        yB_1D = np.copy(PhiM_AnisData.data[:,1]) 
        xB_1D = np.copy(PhiM_AnisData.data[:,2]) 
        AnisType_1D = np.copy(PhiM_AnisData.data[:,3])
        z3D_1D = np.copy(PhiM_AnisData.data[:,4])
        
        tmp_phi_M_1D = np.copy(PhiM_AnisData.data[:,0]) 
        tmp_yB_1D = np.copy(PhiM_AnisData.data[:,1]) 
        tmp_xB_1D = np.copy(PhiM_AnisData.data[:,2]) 
        tmp_AnisType_1D = np.copy(PhiM_AnisData.data[:,3])
        tmp_z3D_1D = np.copy(PhiM_AnisData.data[:,4])
    
    else:
        
        phi_M_1D = np.concatenate((tmp_phi_M_1D,PhiM_AnisData.data[:,0]),axis=0)
        yB_1D = np.concatenate((tmp_yB_1D,PhiM_AnisData.data[:,1]),axis=0)
        xB_1D = np.concatenate((tmp_xB_1D,PhiM_AnisData.data[:,2]),axis=0)
        AnisType_1D = np.concatenate((tmp_AnisType_1D,PhiM_AnisData.data[:,3]),axis=0)
        z3D_1D = np.concatenate((tmp_z3D_1D,PhiM_AnisData.data[:,4]),axis=0)
        
        tmp_phi_M_1D = np.copy(phi_M_1D)  
        tmp_yB_1D = np.copy(yB_1D) 
        tmp_xB_1D = np.copy(xB_1D) 
        tmp_AnisType_1D = np.copy(AnisType_1D)
        tmp_z3D_1D = np.copy(z3D_1D)



# At this point we have computed the normalized Vertical Gradients & the turbulence anisotropy.
# So we can move forward with the rest of the analysis.


#%% PLotting the Log-Log profile, and traditional characteristics:
   
from scipy.optimize import curve_fit 

   
#Canopy Parameters:
height = 10 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy in meters.
kappa = 0.4
Nz_SLayer = int(Nz/2)

#Displacement height for the different cases in meters.
dispH = {'hom-eq_ug9':28.6652,'g1200_ug9':22.2601,'g1200i_ug9':19.3033,'g800_ug9':22.8086,'g400_ug9':24.2941,'g800i_ug9':19.3071}  

#case = 'hom-eq_ug9'
#case = 'g1200_ug9'
case = 'g1200i_ug9'

z = np.arange(0,Nz_SLayer)*dz #Vertical height non-dimensional with zi.
z_on_h = z/canopyH
z_d = (z - (dispH[case]/zi)) #(z-d)/h


# Computing phiM --------------------

u = data.data[:,:,:,0]
v = data.data[:,:,:,1]
avgdUdz = data.data[:,:,:,22]
avgdVdz = data.data[:,:,:,23]

U = np.sqrt(u**2 + v**2)
U_mean = np.mean(U[:,:,0:Nz_SLayer],axis=(0,1))

meandUdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
phi_m_3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

h_canopyTop = height + 1 #One grid point above the top of the canopy. 

ustar = ((-Rstress.data[:,:,0:Nz_SLayer,4]+data.data[:,:,0:Nz_SLayer,20])**2 + (-Rstress.data[:,:,0:Nz_SLayer,5] + data.data[:,:,0:Nz_SLayer,21])**2)**(1/4)

for k in range(0,Nz_SLayer): 
    meandUdz[:,:,k] = ((u[:,:,k]*avgdUdz[:,:,k]) + (v[:,:,k]*avgdVdz[:,:,k]))/(np.sqrt(u[:,:,k]**2 + v[:,:,k]**2))
    phi_m_3D[:,:,k] =  (kappa*z_d[k]/ustar[:,:,h_canopyTop])*meandUdz[:,:,k]


phi_m_1D = np.median(phi_m_3D,axis=(0,1))


#Data used for the Inertial logarithmic fit above the RSL. --------------
#------------------------------------------------------------------------
u1 = U_mean[52] #u_mean[57] This the velocity @ z-d/zi = 0.21186
u2 = U_mean[77] #u_mean[83] This the velocity @ z-d/zi = 0.302749
z_data = np.array([z_d[52], z_d[77]])#, z_ax[103]])
U_data = np.array([u1, u2])#, u3])


"""
Define a function that in principle will fit the data
"""
def func(z, a, b):
    return a*np.log(b*z)

"""
Loop over all Profiles --------------------------------------------------
"""

coefs, pcov = curve_fit(func, z_data, U_data)
u_fit = coefs[0]*np.log(coefs[1]*(z_d))

#------------------------------------------------------------------------
#------------------------------------------------------------------------


fig, ax=plt.subplots(1,2)

ax[0].plot(phi_m_1D,z_d/canopyH,'-k')
ax[0].hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax[0].vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax[0].set_xlabel(r'$\phi_m(z)$')
ax[0].set_ylabel(r'$\frac{z-d}{h}$')
ax[0].set_xlim([-0.5, 1.2])
ax[0].set_ylim([z_d[0]/canopyH, z_d[-1]/canopyH])


ax[1].plot(U_mean,z_d/canopyH,'-k')
ax[1].plot(U_data,z_data/canopyH,'ok')
ax[1].plot(u_fit,z_d/canopyH,'--k')
ax[1].hlines((canopyH-(dispH[case]/zi))/canopyH,0,30,colors='gray',linestyles='-')
ax[1].set_xlabel(r'$\overline{u}(z)$')
ax[1].set_ylabel(r'$\frac{z-d}{h}$')
ax[1].set_xlim([0, 30])
ax[1].set_ylim([z_d[0]/canopyH, z_d[-1]/canopyH])

plt.tight_layout()


#%% Create the colormap designed to study anisotropy:
    
cmap = ColorAnisotropy()




#%% Graphical Representation of the results with Clustering analysis, ploting individual cluster contributions:

import matplotlib.ticker
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import kde

canopyH = 39 #height of the canopy in meters.
z_ax = np.arange(0,Nz_SLayer)*dz
Nclusters = 10 #Number of clusters used to group the anisotorpy.

fig, axs = plt.subplots(nrows=3,ncols=3)
plt.ion()

#Loop through all the clusters (1 to 9) organized in a 3x3 subplot..
cluster = 0
for m in range(0,3):
    for n in range(0,3):

        cluster = cluster + 1

        tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
        tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
        tmp_yB = yB_1D[(AnisType_1D == cluster)]

        if (cluster < 9):
            x = tmp_phi_u
            y = (tmp_z3D/(canopyH/zi))
    
            #PLot when we don't cluster based on zoL: 
                #sc = axs.scatter(x,y,s = 30, marker='o',c = tmp_yB,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

            #Developing a "2D Density plot" to better visualize where there are more points.
            tmp = np.array([x,y])
            nbins = 40

            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            k = kde.gaussian_kde(tmp)
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
            # plot a density
            #axs.set_title('Calculate Gaussian KDE')
            #axs.pcolormesh(xi, yi, density.reshape(xi.shape), shading='auto', cmap='Greys')
 
            # add shading
            #axs.set_title('2D Density with shading')
            axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')


        #Overlay the median on the denisty plot:
        x_median = np.zeros(Nz_SLayer,dtype='float')
        yB_median = np.zeros(Nz_SLayer,dtype='float')


        #First we downselect data based on the yB clustering developed earlier.
        tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
        tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
    
        for i in range(0,Nz_SLayer-1):
            x_median[i] = np.median(tmp_phi_u[(tmp_z3D >= z_ax[i]) & (tmp_z3D < z_ax[i+1])])
            yB_median[i] = np.median(tmp_yB[(tmp_z3D >= z_ax[i]) & (tmp_z3D < z_ax[i+1])])     
    

        y_ax = z_ax/(canopyH/zi)
    
    
        sc = axs[m,n].scatter(x_median[0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

#------------------------------------------------------------------------------


        axs[m,n].set_xlim(-0.1, 3.5)
        axs[m,n].set_ylim(0, 13)
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
plt.show()




#%% Graphical Representation of the results with Clustering in a single subplot:

import matplotlib.ticker

z_ax = np.arange(0,Nz_SLayer)*dz
Nclusters = 10


fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

#Median of the gradient profiles as a function of cluster and height. 
#For a fixed cluster (e.g. cluster = 4), we average all values of the gradient
#at a specific height.
 
phiU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
yB_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')

for cluster in range(1,Nclusters):


    #First we downselect data based on the yB clustering developed earlier.
    tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
    tmp_yB = yB_1D[(AnisType_1D == cluster)]
    tmp_z3D = z3D_1D[(AnisType_1D == cluster)]

    for k in range(0,Nz_SLayer-1):            
        
        phiU_median[cluster,k] = np.median(tmp_phi_u[(tmp_z3D >= z_ax[k]) & (tmp_z3D < z_ax[k+1])]) 
        yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z_ax[k]) & (tmp_z3D < z_ax[k+1])]) 
        

    axs.scatter(phiU_median[cluster,0:-1],(z_ax[0:-1]/(canopyH/zi)),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)


axs.set_xlim(-0.1, 3.5)
axs.set_ylim(0, 13)
axs.axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
axs.axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')


axs.set_xscale('linear')
cbar = plt.colorbar(sc)
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(r'$\phi_M$')
axs.set_title(f'{cases[num]}')


axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')


plt.tight_layout()
plt.show()





#%% Loading the anisotropy metric of xB and yB


anisotropy = xr.DataArray(np.ones(shape = (Nx,Ny,Nz_SLayer,2),order='C'),\
                        dims=('x','y','z','variable'), coords = {'variable':['xB','yB']})
        
anisotropy = xr.open_dataarray('anisotropy.nc')    

xB = anisotropy.data[:,:,:,0]
yB = anisotropy.data[:,:,:,1]


#%% PLotting Turbulence Anisotropy in 2D Horizontal Slices (including location of Gaps):

x_ax = np.arange(0,Nx)*dx
y_ax = np.arange(0,Ny)*dy

#Gap400_Centers = xr.open_dataarray('newfile.nc')
#Canopy = xr.open_dataarray('new_LAD.nc')

iva_colors = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]

#xc = Gap400_Centers.data[0,:]*dx
#yc = Gap400_Centers.data[1,:]*dy


#area = (pi/4)*s
radius = 8*dx
area = np.pi*(radius**2)
s_test = area/(np.pi/4)

#Height of the horizontal slice:
z_slice = 6*height

#radius = np.ones(np.size(xc))*(8*dx)

#Canopy Plot:
#fig, axs = plt.subplots(ncols=1, nrows=1)
#axs.pcolormesh(x_ax,y_ax,Canopy.data)
#for i in range(0,np.size(xc)):
#    axs.add_patch(plt.Circle((xc[i],yc[i]), radius=8*dx, facecolor='None',edgecolor='k'))


fig, axs = plt.subplots(ncols=1, nrows=1)
axs.contourf(x_ax,y_ax,np.transpose(yB[:,:,z_slice]),levels=10,colors=iva_colors);#plt.colorbar()
axs.contour(x_ax,y_ax,np.transpose(yB[:,:,z_slice]),levels=[0.2,0.4,0.6],colors='k',linewidths=0.5)
#for i in range(0,np.size(xc)):
#    axs.add_patch(plt.Circle((xc[i],yc[i]), radius=8*dx, facecolor='None',edgecolor='k'))

axs.set_title(f"yB @ z/h = {z_slice/height}")
axs.set_xlabel(f"x [km]")
axs.set_ylabel(f"y [km]")

fig, axs = plt.subplots(ncols=1, nrows=1)
axs.contourf(x_ax,y_ax,np.transpose(xB[:,:,z_slice]),levels=10,colors=iva_colors);#plt.colorbar()
axs.contour(x_ax,y_ax,np.transpose(xB[:,:,z_slice]),levels=[0.2,0.4,0.6],colors='k',linewidths=0.5)
#for i in range(0,np.size(xc)):
#    axs.add_patch(plt.Circle((xc[i],yc[i]), radius=8*dx, facecolor='None',edgecolor='k'))

axs.set_title(f"xB @ z/h = {z_slice/height}")
axs.set_xlabel(f"x [km]")
axs.set_ylabel(f"y [km]")



#%% Vertical Slices of turbulence anisotropy:

x_ax = np.arange(0,Nx)*dx
yslice = 32
    
plt.figure();
#plt.contourf(x_ax,z_ax[0:nz_max],np.transpose(yB[:,yslice,0:nz_max]),levels=10,colors=iva_colors);#plt.colorbar()
sc = plt.pcolormesh(x_ax,z_ax[0:Nz_SLayer],np.transpose(yB[:,yslice,0:Nz_SLayer]),shading='auto',cmap = cmap,vmin=0,vmax=np.sqrt(3)/2)
#plt.contour(x_ax,z_ax[0:nz_max],np.transpose(yB[:,yslice,0:nz_max]),levels=[0.2,0.4,0.6],colors='k',linewidths=0.5)
plt.contour(x_ax,z_ax[0:Nz_SLayer],np.transpose(yB[:,yslice,0:Nz_SLayer]),levels=10,colors='k',linewidths=0.5)
plt.hlines(3*canopyH,0,x_ax[-1],colors='k',linestyles='--')
plt.hlines(canopyH,0,x_ax[-1],colors='k',linestyles='--')
#plt.hlines(0,0,x_ax[-1],colors='k',linestyles='--')
plt.title(f"yB @ y = {32}")
plt.xlabel(r'$x/z_i$')
plt.ylabel(r'$\frac{z}{z_i}$')
cbar = plt.colorbar(sc)


'''

plt.figure();
for j in range(0,2,2):
    plt.pcolormesh(x_ax,z_ax[0:nz_max],np.transpose(yB[:,j,0:nz_max]),shading='auto',cmap = cmap)#;plt.colorbar()
    #plt.contour(x_ax,z_ax[0:nz_max],np.transpose(yB[:,yslice,0:nz_max]),levels=[0.2,0.4,0.6],colors='k',linewidths=0.5)
    plt.contour(x_ax,z_ax[0:nz_max],np.transpose(yB[:,j,0:nz_max]),levels=10,colors='k',linewidths=0.5)
    plt.hlines(3*canopyH,0,x_ax[-1],colors='k',linestyles='--')
    plt.hlines(canopyH,0,x_ax[-1],colors='k',linestyles='--')
    plt.hlines(dispH_g400_9ms*dz,0,x_ax[-1],colors='k',linestyles='--')
    plt.title(f"yB @ y = {32}")
    plt.xlabel(f"x [km]")
    plt.ylabel(f"y [km]")
    plt.pause(0.5)
    #input("Press Enter to continue...")
    plt.clf()

'''


plt.figure();
sc = plt.pcolormesh(x_ax,z_ax[0:Nz_SLayer],np.transpose(xB[:,yslice,0:Nz_SLayer]),shading='auto',cmap = cmap,vmin=0,vmax=1)
#plt.contour(x_ax,z_ax[0:nz_max],np.transpose(xB[:,yslice,0:nz_max]),levels=[0.2,0.4,0.6],colors='k',linewidths=0.5)
plt.contour(x_ax,z_ax[0:Nz_SLayer],np.transpose(xB[:,yslice,0:Nz_SLayer]),levels=10,colors='k',linewidths=0.5)
plt.hlines(3*canopyH,0,x_ax[-1],colors='k',linestyles='--')
plt.hlines(canopyH,0,x_ax[-1],colors='k',linestyles='--')
#plt.hlines(0,0,x_ax[-1],colors='k',linestyles='--')
plt.title(f"xB @ y = {32}")
plt.xlabel(r'$x/z_i$')
plt.ylabel(r'$\frac{z}{z_i}$')

cbar = plt.colorbar(sc)


#%% Vorticity Thickness (wL)

wL = np.zeros((Nx,Ny))

for i in range(0,Nx):
    for j in range(0,Ny):
        wL[i,j] = u[i,j,dispH]/dudz[i,j,dispH]

wL_1D = wL.flatten()


plt.figure()
plt.hist(wL_1D)




plt.figure()
plt.pcolormesh(wL)
plt.clim(0.04,0.2)
plt.colorbar()








