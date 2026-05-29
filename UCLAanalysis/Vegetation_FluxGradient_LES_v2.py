"""
Program Author: Marc Calaf.

This program uses Giulia's LES data, to compute vertical gradients 
of the mean velocity as a function of z/z* in neutral conditions, and turbulence anisotropy.
This program is a streamlined version of "Vegetation_FluxGradient_LES.py"

Date created: 22 April 2023
Last date modified: 22 April 2023

To Do: 

"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr


os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering
from Stats import  ReynoldsStress

#%% Defining the path to the files:

#Giulia's Data:

directory = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/'
cases = ['hom-eq_9mps_ug','g1200_9mps_ug','g1200_inv_9mps_ug','g800_9mps_ug','g800i_9mps_ug','g400_9mps_ug']
num = 5

path = directory + 'RAV_' + cases[num] +'/'

os.chdir(path)  


#%% Defining the main parameters of the simulations

anisotropy_compute = 'false' #If 'True' turb. Anisotropy and clustering will be computed from scratch, otherwise read from saved file.

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


#Canopy Parameters:
height = 10 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy in meters.
kappa = 0.4
Nz_SLayer = int(Nz/2)

#Displacement height for the different cases in meters.
dispH = {'hom-eq_9mps_ug':28.6652,'g1200_9mps_ug':22.2601,'g1200_inv_9mps_ug':19.3033,'g800_9mps_ug':22.8086,'g800i_9mps_ug':19.3071,'g400_9mps_ug':24.2941}  #Older.
#dispH = {'hom-eq_9mps_ug':28.6449,'g1200_9mps_ug':21.8884,'g1200_inv_9mps_ug':19.2495,'g800_9mps_ug':22.3484,'g800i_9mps_ug':19.2309,'g400_9mps_ug':23.5872,'g400i_9mps_ug':19.2064} #Revised 04/25/2023  

case = cases[num]

z = (np.arange(0,Nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_on_h = z/canopyH
z_d = (z - (dispH[case]/zi)) #(z-d)/h



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


#Vertical profiles of the shear stress:
    
fig, axs = plt.subplots(nrows=2,ncols=3)
plt.ion()

l = 0
Rij_names = list(Rstress.coords['variable'].data[:])

for m in range(0,2):
    for n in range(0,3):
        
        
        print(l)

        y = z_d/canopyH
    
        if (l < 4):
            Rij = np.mean(Rstress.data[:,:,:,l],axis=(0,1))
        else:
            Rij = - np.mean(Rstress.data[:,:,:,l],axis=(0,1))
        
        SGSij = np.mean(data.data[:,:,:,16+l],axis=(0,1))
        Tauij = Rij + SGSij
        
        #PLotting arguments:
            
        axs[m,n].plot(Rij,y,color='gray',linestyle='--')
        axs[m,n].plot(SGSij,y,color='gray',linestyle='-.')
        axs[m,n].plot(Tauij,y,color='black',linestyle='-')

        axs[m,n].set_ylim(y[0], y[-1])
        #axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
        #axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()

        axs[m,n].set_xscale('linear')
        #axs[m,n].set_title(f'Case = {case}')
        axs[m,n].set_xlabel(f'{Rij_names[l]}')
        
        l = l+1
    
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')

plt.tight_layout()
plt.show()

# Vertical profile of the vertical shear stress together:
    
Dxz = np.mean((data.data[:,:,:,0]*data.data[:,:,:,2]),axis=(0,1)) - (np.mean(data.data[:,:,:,0],axis=(0,1))*np.mean(data.data[:,:,:,2],axis=(0,1)))
Dyz = np.mean((data.data[:,:,:,1]*data.data[:,:,:,2]),axis=(0,1)) - (np.mean(data.data[:,:,:,1],axis=(0,1))*np.mean(data.data[:,:,:,2],axis=(0,1)))

Dw =  np.sqrt(Dxz**2 + Dyz**2)
Rw = np.mean((np.sqrt(Rstress.data[:,:,:,4]**2 + Rstress.data[:,:,:,5]**2)),axis=(0,1))
SGSw = np.mean((np.sqrt(data.data[:,:,:,20]**2 + data.data[:,:,:,21]**2)),axis=(0,1))
    
tau_wall1D = Rw + SGSw + Dw

    
fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

axs.plot(tau_wall1D,y,color='black',linestyle='-')
axs.hlines((canopyH-(dispH[case]/zi))/canopyH,0,4.5,colors='gray',linestyles='-')
axs.set_ylim(y[0], y[-1])
axs.set_xlim(0, 4.5)

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xlabel(r'$\tau_w$')
axs.set_ylabel(r'$(z-d)/h$')
    

#%% Compute the phi_m and plot both the derivative and mean wind speed:
   
from scipy.optimize import curve_fit 


# Computing phiM --------------------


[phi_m_3D,U,ustar,z_d_ustar] = phi_m(Nx,Ny,Nz_SLayer,height,z_d,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,22],data.data[:,:,:,23],
                 Rstress.data[:,:,0:Nz_SLayer,4],data.data[:,:,0:Nz_SLayer,20],Rstress.data[:,:,0:Nz_SLayer,5],data.data[:,:,0:Nz_SLayer,21])

U_mean = np.mean(U[:,:,0:Nz_SLayer],axis=(0,1))
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
u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

#------------------------------------------------------------------------
#------------------------------------------------------------------------


## Plotting the Mean wind speed and the corresponding velocity gradient.

fig, ax=plt.subplots(1,2)

ax[0].plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-k')
ax[0].hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax[0].vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax[0].set_xlabel(r'$\phi_m(z)$')
ax[0].set_ylabel(r'$\frac{z-d}{h}$')
ax[0].set_xlim([-0.5, 1.5])
ax[0].set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])


ax[1].plot(U_mean,z_d[0:Nz_SLayer]/canopyH,'-k')
ax[1].plot(U_data,z_data/canopyH,'ok')
ax[1].plot(u_fit,z_d[0:Nz_SLayer]/canopyH,'--k')
ax[1].hlines((canopyH-(dispH[case]/zi))/canopyH,0,30,colors='gray',linestyles='-')
ax[1].set_xlabel(r'$\overline{u}(z)$')
ax[1].set_ylabel(r'$\frac{z-d}{h}$')
ax[1].set_xlim([0, 30])
ax[1].set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])

plt.tight_layout()
    
    
## Plotting the ustar values used for the normalization of the velocity gradient

X = np.arange(0,Nx)*dx
Y = np.arange(0,Ny)*dy

fig, ax=plt.subplots(1,1)  
img = ax.pcolormesh(X,Y,ustar,shading='gouraud')
ax.set_xlabel(r'$L_x$')
ax.set_ylabel(r'$L_y$')
ax.set_title(r'$u_* $')
plt.colorbar(img, ax=ax)

## Plotting the height at which the corresponding ustar values have been extracted

X = np.arange(0,Nx)*dx
Y = np.arange(0,Ny)*dy

fig, ax=plt.subplots(1,1)  
img = ax.pcolormesh(X,Y,z_d_ustar,shading='gouraud')
ax.set_xlabel(r'$L_x$')
ax.set_ylabel(r'$L_y$')
ax.set_title(r'$(z-d)_{u*_{max}} $')
plt.colorbar(img, ax=ax)


#%% Compute the velocity gradient for all study cases:

from scipy.signal import butter,filtfilt
    
nyq = 2 * (1/dz)  # Nyquist Frequency
order = 2
cutoff = 0.2*(1/dz)
fs = dz

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y





NumCases = 6

fig, ax=plt.subplots(1,1)

for num in range(0,NumCases):

    case = cases[num]
    path = directory + 'RAV_' + cases[num] +'/'
    os.chdir(path)
    
    print(path)
    
    data = xr.open_dataarray('Data_Momentum.nc')
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                             data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])
    
    [phi_m_3D,U,ustar,z_d_ustar] = phi_m(Nx,Ny,Nz_SLayer,height,z_d,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,22],data.data[:,:,:,23],
                     Rstress.data[:,:,0:Nz_SLayer,4],data.data[:,:,0:Nz_SLayer,20],Rstress.data[:,:,0:Nz_SLayer,5],data.data[:,:,0:Nz_SLayer,21])

    phi_m_1D = np.median(phi_m_3D,axis=(0,1))
    
    #derivative of the velocity gradient used to check where the velocity gradient is more parallel to the inertial limit of 1.
    dydz = np.gradient(phi_m_1D,dz)
    
    #Low pass filter the derivative:
    dydz_filt = butter_lowpass_filter(dydz, cutoff, fs, order)
    
    ax.plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-',label=case) #Plot of all PhiM
    #ax.plot(dydz,z_d[0:Nz_SLayer]/canopyH,'-',label=case)
    #ax.plot(dydz_filt,z_d[0:Nz_SLayer]/canopyH,'--',label=case)
    



ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax.vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z-d}{h}$')
#ax.set_xlim([-0.5, 1.5])
ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])


#%% PLot z* as obtained from phim as a function of \lambda*Lacunarity

z_phiM = {'hom-eq_9mps_ug':3.74,'g1200_9mps_ug':6.0,'g1200_inv_9mps_ug':6.78,'g800_9mps_ug':5.25,'g800i_9mps_ug':7.61,'g400_9mps_ug':4.9}
z_star = {'hom-eq_9mps_ug':0,'g1200_9mps_ug':0,'g1200_inv_9mps_ug':0,'g800_9mps_ug':0,'g800i_9mps_ug':0,'g400_9mps_ug':0}

factor = 7.05*0.2
lambda_lac = {'hom-eq_9mps_ug':1*factor,'g1200_9mps_ug':0.766*factor,'g1200_inv_9mps_ug':0.2647*factor,'g800_9mps_ug':0.7769*factor,'g800i_9mps_ug':0.2409*factor,'g400_9mps_ug':0.7702*factor}

for num in range(0,NumCases):
    case = cases[num]
    z_star[case] = ((z_phiM[case]*(canopyH*zi)) + dispH[case])/(canopyH*zi)



colors = ['#1f77b4ff','#ff7f0eff','#2ca02cff','#d62728ff','#9467bdff','#8c564bff']


fig, ax=plt.subplots(1,1)

scatter = ax.scatter(lambda_lac.values(),z_phiM.values(),c = colors)
ax.set_ylabel(r'$(z_*-d)/h$')
ax.set_xlabel(r'$\lambda \,L_c $')




fig, ax=plt.subplots(1,1)

ax.scatter(lambda_lac.values(),z_star.values(),c = colors)
ax.set_ylabel(r'$z_*/h$')
ax.set_xlabel(r'$\lambda \,L_c $')


#%% Plot of PhiM as a function of z/z*



fig, ax=plt.subplots(1,1)

for num in range(0,NumCases):
    case = cases[num]
    
    #Compute the corresponding velocity gradient for each case:
    #------------------------------------------------------------------------
    path = directory + 'RAV_' + cases[num] +'/'
    os.chdir(path)
    
    print(path)
    
    data = xr.open_dataarray('Data_Momentum.nc')
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                             data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])
    
    [phi_m_3D,U,ustar,z_d_ustar] = phi_m(Nx,Ny,Nz_SLayer,height,z_d,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,22],data.data[:,:,:,23],
                     Rstress.data[:,:,0:Nz_SLayer,4],data.data[:,:,0:Nz_SLayer,20],Rstress.data[:,:,0:Nz_SLayer,5],data.data[:,:,0:Nz_SLayer,21])

    phi_m_1D = np.median(phi_m_3D,axis=(0,1))
    
    #------------------------------------------------------------------------
    
    z_zstar = z/z_star[case]
    #z_zstar = z_d/z_star[case]
    
    ax.plot(phi_m_1D,z_zstar[0:Nz_SLayer],'-',label=case) #Plot of all PhiM
    

#ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
#ax.vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z}{z*}$')
#ax.set_ylabel(r'$\frac{z-d}{z*}$')
#ax.set_xlim([-0.5, 1.5])
#ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])

    
#%% Computing turbulence anisotropy and clustering:
    
# Decide whether to compute turbulence anisotropy and clustering from scratch or read it from file.

if (anisotropy_compute == 'true'):
    
    [xB,yB,AnisType_1D] = Anisotropy_Clustering(Nx,Ny,Nz_SLayer,Rstress)

    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)


    Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz_SLayer,3),order='F'),\
                        dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
        
    Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

    os.chdir(path)
    Anisotropy_clustering.to_netcdf('Anisotropy_clustering.nc')
    
else:
        
    Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering.nc')

    xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
    yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
    AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 

#------------ End of the IF statement.

cmap = ColorAnisotropy()    

z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_d[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)
phi_M_1D = np.ndarray.flatten(phi_m_3D)
   
    
    
#%% Graphical Representation of the results with Clustering analysis, ploting individual cluster contributions:


from scipy.stats import kde


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
            y = tmp_z3D/canopyH
    

            #Developing a "2D Density plot" to better visualize where there are more points.
            tmp = np.array([x,y])
            nbins = 40

            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            k = kde.gaussian_kde(tmp)
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
            # plot the density with shading
            #axs.set_title('2D Density with shading')
            axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')


        #Overlay the median on the denisty plot:
        phiM_median = np.zeros(Nz_SLayer,dtype='float')
        yB_median = np.zeros(Nz_SLayer,dtype='float')


        #First we downselect data based on the yB clustering developed earlier.
        #tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
        #tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
    
        for i in range(0,Nz_SLayer-1):
            phiM_median[i] = np.median(tmp_phi_u[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])
            yB_median[i] = np.median(tmp_yB[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])     
    

        y_ax = z_d[0:Nz_SLayer]/canopyH
    
    
        sc = axs[m,n].scatter(phiM_median[0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

#------------------------------------------------------------------------------


        axs[m,n].set_xlim(-0.5, 2)
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
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')
axs[2,0].set_ylabel(r'$(z-d)/h$')

axs[2,0].set_xlabel(r'$\phi_M$')
axs[2,1].set_xlabel(r'$\phi_M$')
axs[2,2].set_xlabel(r'$\phi_M$')


plt.tight_layout()
plt.show()


    
#%% Graphical Representation of the results with Clustering in a single subplot:

import matplotlib.ticker


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
        
        phiU_median[cluster,k] = np.median(tmp_phi_u[(tmp_z3D >= z_d[k]) & (tmp_z3D < z_d[k+1])]) 
        yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z_d[k]) & (tmp_z3D < z_d[k+1])]) 
        

    axs.scatter(phiU_median[cluster,0:-1],(z_d[0:Nz_SLayer-1]/canopyH),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)


axs.set_xlim(-0.5, 2)
axs.set_ylim(y_ax[0], y_ax[Nz_SLayer-1])
axs.axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
axs.axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')


axs.set_xscale('linear')
cbar = plt.colorbar(sc)
axs.set_ylabel(r'$(z-d)/h$')
axs.set_xlabel(r'$\phi_M$')
axs.set_title(f'{cases[num]}')


axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')


plt.tight_layout()
plt.show()

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
