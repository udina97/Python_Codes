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
import copy


os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")


from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering
from Stats import  ReynoldsStress, DispFluct

#%% Defining the path to the files:

#Giulia's Data:

# directory = '/Users/mcalaf/Documents/Utah/Students/Giulia_Salmaso/2023/LES_VegCanopy_Data/'
# cases = ['hom-eq_9mps_ug','g1200_9mps_ug','g1200_inv_9mps_ug','g800_9mps_ug','g800i_9mps_ug','g400_9mps_ug','g400i_9mps_ug']
# num = 3

# path = directory + 'RAV_' + cases[num] +'/'


# Ben data
directory = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simATTO1_256x256x384_rl_out10000'
path = directory + sim + '/'

os.chdir(path) 

#%%

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

pcnt3 = 10000 #18000; %Important! used in get_var 
avg_time = pcnt3*1 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
slice_profiles = False
shifting_z = True
tke_budget_flag = True
derivatives_flag = True
PCON_flag = False
PCON_TEMP_flag = False
v_fracflag = False
SAVEAS_flag = False
clear_var_flag = False

T_STC = 299 #320; %298.15; %[K], temperature scale
dt = 0.05 #05; %0.000005;
zi = 1000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

parentpath = path
file_name = 'ke.txt'

# Load and import data
ke = np.genfromtxt(parentpath + file_name)
ts = np.linspace(0, len(ke) * wbase, len(ke))

# Input and output paths
ipath = path + f'/output/ta1_field/'
opath = path + f'/output/'

# Read parameter file for ta1_field
with open(ipath + 'parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

nx = int(param[0])
ny = int(param[1])
nz = int(param[2])
lx = param[3]
ly = param[4]
lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
nt = int(param[9])
actLaunch = int(param[10])
mpiProc = int(param[11])
Re = param[12]
Ri = param[13]
Pr = param[14]
alpha = param[15]
SGS = param[16]
S_FLAG = param[17]
ibm = param[18]
rot = param[19]
nzTot = nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

if shifting_z:
    # Modify axes to move 0 to IBM surface
    z_uvp = z_uvp[5:]
    z_uvp = z_uvp - z_uvp[0]
    z_w = z_w[5:]
    z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))
    nt = len(ke) // pcnt3
elif ibm == 1:
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(ipath + '../phi_functions/', nx, ny, nzTot, mpiProc)
    intf, iintf = build_intf(phi, dz)
else:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))

if ibm:
    phi_uv = np.zeros_like(phi)
    for k in range(nz - 1):
        phi_uv[:, :, k] = (phi[:, :, k] + phi[:, :, k + 1]) / 2.0

if PCON_flag and v_fracflag:
    v_frac = get_var(opath + 'cut_cell/', 'v_frac', nx, ny, nzTot, 1, iintf, mpiProc)
    a_cut = get_var(opath + 'cut_cell/', 'a_cut', nx, ny, nzTot, 1, iintf, mpiProc)
    
#%% Defining the main parameters of the simulations

anisotropy_compute = 'false' #If 'True' turb. Anisotropy and clustering will be computed from scratch, otherwise read from saved file.

NumVariables = 26
NumVariablesSC = 10

#Tscale = 290 #[K]
u_scale = 0.4 #[m/s]
zi = 1000 #[m]

Nz_SLayer = 128 #This is the grid point that corresponds to about 100m height, surface alyer.


#Canopy Parameters:
height = 16 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy in meters.
kappa = 0.4
Nz_SLayer = int(nz/2)
# NumCases = 7 #Total number of study cases.

#Displacement height for the different cases in meters.
'''
From Giulia 27/09/2023
%             h Amzon,   h-eq,    g12,     g8,    g4,  h-eq-i,  g12-i,   g8-i,   g4-i
d_9mps     =  [29.480, 28.640, 21.911, 22.255, 23.300, 24.569, 19.250, 19.236, 19.202]
d_3mps     =  [28.932, 28.157, 21.714, 22.144, 23.309, 24.331, 19.232, 19.222, 19.191]
'''
# dispH = {'hom-eq_9mps_ug':28.640,'g1200_9mps_ug':21.911,'g1200_inv_9mps_ug':19.250,'g800_9mps_ug':22.255,
#          'g800i_9mps_ug':19.236,'g400_9mps_ug':23.300,'g400i_9mps_ug':19.202} #in [m]

#Here we just initialize the following two dictionaries. The corresponding values will computed inside.

# z0hi_dict = {'hom-eq_9mps_ug':0,'g1200_9mps_ug':0,'g1200_inv_9mps_ug':0,'g800_9mps_ug':0,
#          'g800i_9mps_ug':0,'g400_9mps_ug':0,'g400i_9mps_ug':0} #in [m]

# ustar_dict = {'hom-eq_9mps_ug':0,'g1200_9mps_ug':0,'g1200_inv_9mps_ug':0,'g800_9mps_ug':0,
#          'g800i_9mps_ug':0,'g400_9mps_ug':0,'g400i_9mps_ug':0} #in [m/s]

# case = cases[num]

z = (np.arange(0,nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_on_h = z/canopyH
# z_d = (z - (dispH[case]/zi)) #(z-d)/h



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


# data = xr.open_dataarray('Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#dataS = xr.open_dataarray('Data_Scalar.nc')

# data_2D = xr.open_dataarray('Data_Momentum_2D.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#dataS_2D = xr.open_dataarray('Data_Scalar_2D.nc')

#%%Import variables

nt_tot = nt * actLaunch - incskip
print('nt_tot is', nt_tot)

avgT = int(avg_time / pcnt3)
print('avgT is', avgT)

data = dict()
data_tavg = dict()

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
       'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
       'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
       'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']
    
for i in range(len(var)):
    data[var[i]] = get_var(ipath,var[i],nx,ny,nzTot,nt_tot,iintf,mpiProc,avgT)
    data_tavg[var[i]] = np.mean(data[var[i]],axis=3)
    
#%% Loading the topography data:

zeds = np.zeros((nx,ny,nz))

for i in range(0,nx):
    for j in range(0,ny):
        zeds[i,j,:] = np.arange(0,nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            
#%% Computing the Reynolds Stress

y = z/canopyH #z_d/canopyH

var_Rij = ['Rxx', 'Ryy', 'Rzz', 'Rxy', 'Rxz', 'Ryz']
var_SGSij = ['SGSxx', 'SGSyy', 'SGSzz', 'SGSxy', 'SGSxz', 'SGSyz']
var_Tauij = ['Tauxx', 'Tauyy', 'Tauzz', 'Tauxy', 'Tauxz', 'Tauyz']


Rij = dict()
Rij['Rxx'] = data_tavg['uu'] - data_tavg['u']**2
Rij['Ryy'] = data_tavg['vv'] - data_tavg['v']**2
Rij['Rzz'] = data_tavg['ww'] - data_tavg['w']**2
Rij['Rxy'] = data_tavg['uv'] - data_tavg['u']*data_tavg['v']
Rij['Rxz'] = data_tavg['uw'] - data_tavg['u']*data_tavg['w']
Rij['Ryz'] = data_tavg['vw'] - data_tavg['v']*data_tavg['w']

SGSij = dict()
SGSij['SGSxx'] = data_tavg['txx']
SGSij['SGSyy'] = data_tavg['tyy']
SGSij['SGSzz'] = data_tavg['tzz']
SGSij['SGSxy'] = data_tavg['txy']
SGSij['SGSxz'] = data_tavg['txz']
SGSij['SGSyz'] = data_tavg['tyz']

Tauij = dict()
Tauij['Tauxx'] = Rij['Rxx'] + SGSij['SGSxx']
Tauij['Tauyy'] = Rij['Ryy'] + SGSij['SGSyy']
Tauij['Tauzz'] = Rij['Rzz'] + SGSij['SGSzz']
Tauij['Tauxy'] = Rij['Rxy'] + SGSij['SGSxy']
Tauij['Tauxz'] = Rij['Rxz'] + SGSij['SGSxz']
Tauij['Tauyz'] = Rij['Ryz'] + SGSij['SGSyz']

for i in range(len(var_Rij)):
    Rij[var_Rij[i]][(dist < 0)] = float("nan")
    
for i in range(len(var_SGSij)):
    SGSij[var_SGSij[i]][(dist < 0)] = float("nan")
    
for i in range(len(var_Tauij)):
    Tauij[var_Tauij[i]][(dist < 0)] = float("nan")

#Shear Stress profiles
fig, axs=plt.subplots(2,3, constrained_layout=True)

axs[0,0].plot(np.nanmean(Rij['Rxx'],axis=(0,1)),y,color='gray',linestyle='--')
axs[0,0].plot(np.nanmean(SGSij['SGSxx'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[0,0].plot(np.nanmean(Tauij['Tauxx'],axis=(0,1)),y,color='black',linestyle='-')
axs[0,1].plot(np.nanmean(Rij['Ryy'],axis=(0,1)),y,color='gray',linestyle='--')
axs[0,1].plot(np.nanmean(SGSij['SGSyy'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[0,1].plot(np.nanmean(Tauij['Tauyy'],axis=(0,1)),y,color='black',linestyle='-')
axs[0,2].plot(np.nanmean(Rij['Rzz'],axis=(0,1)),y,color='gray',linestyle='--')
axs[0,2].plot(np.nanmean(SGSij['SGSzz'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[0,2].plot(np.nanmean(Tauij['Tauzz'],axis=(0,1)),y,color='black',linestyle='-')
axs[1,0].plot(np.nanmean(Rij['Rxy'],axis=(0,1)),y,color='gray',linestyle='--')
axs[1,0].plot(np.nanmean(SGSij['SGSxy'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[1,0].plot(np.nanmean(Tauij['Tauxy'],axis=(0,1)),y,color='black',linestyle='-')
axs[1,1].plot(np.nanmean(Rij['Rxz'],axis=(0,1)),y,color='gray',linestyle='--')
axs[1,1].plot(np.nanmean(SGSij['SGSxz'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[1,1].plot(np.nanmean(Tauij['Tauxz'],axis=(0,1)),y,color='black',linestyle='-')
axs[1,2].plot(np.nanmean(Rij['Ryz'],axis=(0,1)),y,color='gray',linestyle='--')
axs[1,2].plot(np.nanmean(SGSij['SGSyz'],axis=(0,1)),y,color='gray',linestyle='-.')
axs[1,2].plot(np.nanmean(Tauij['Tauyz'],axis=(0,1)),y,color='black',linestyle='-')

axs[0,0].set_ylim(y[0],y[-1]);axs[0,1].set_ylim(y[0],y[-1]);axs[0,2].set_ylim(y[0],y[-1])
axs[1,0].set_ylim(y[0],y[-1]);axs[1,1].set_ylim(y[0],y[-1]);axs[1,2].set_ylim(y[0],y[-1])

axs[0,0].set_xlabel('Rxx');axs[0,1].set_xlabel('Ryy');axs[0,2].set_xlabel('Rzz')
axs[1,0].set_xlabel('Rxy');axs[1,1].set_xlabel('Rxz');axs[1,2].set_xlabel('Ryz')

axs[0,0].set_ylabel('$z/h_c$');axs[1,0].set_ylabel('$z/h_c$');

# plt.savefig(figPath+'Rij.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

# # Vertical profile of the vertical shear stress together:
    
Dxz = np.mean((data_tavg['u']*data_tavg['w']),axis=(0,1)) - (np.mean(data_tavg['u'],axis=(0,1))*np.mean(data_tavg['w'],axis=(0,1)))
Dyz = np.mean((data_tavg['v']*data_tavg['w']),axis=(0,1)) - (np.mean(data_tavg['v'],axis=(0,1))*np.mean(data_tavg['w'],axis=(0,1)))

Dw =  np.sqrt(Dxz**2 + Dyz**2)
Rw = np.mean((np.sqrt(Rij['Rxz']**2 + Rij['Ryz']**2)),axis=(0,1))
SGSw = np.mean((np.sqrt(data_tavg['txz']**2 + data_tavg['tyz']**2)),axis=(0,1))
    
tau_wall1D = Rw + SGSw + Dw
    
fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

axs.plot(tau_wall1D,y,color='black',linestyle='-')
axs.plot(Rw,y,color='black',linestyle='--')
axs.plot(SGSw,y,color='black',linestyle='-.')
axs.plot(Dw,y,color='black',linestyle=':')
# axs.hlines((canopyH-(dispH[case]/zi))/canopyH,0,4.5,colors='gray',linestyles='-')
axs.set_ylim(y[0], y[-1])
axs.set_xlim(-1, 4.5)

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xlabel(r'$\tau_w$')
axs.set_ylabel(r'$(z-d)/h$')
    

#%% Compute the corresponding values of z0hi and u* above the canopy:

from scipy.optimize import curve_fit     


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
    U_mean = np.mean(U[:,:,0:Nz_SLayer],axis=(0,1))

    #Data used for the Inertial logarithmic fit above the RSL. --------------
    #------------------------------------------------------------------------
    level1 = 70
    level2 = 95

    u1 = U_mean[level1] #This the velocity @ z-d/zi = 0.253
    u2 = U_mean[level2] #This the velocity @ z-d/zi = 0.350
    z_data = np.array([z_d[level1], z_d[level2]])#
    U_data = np.array([u1, u2])#, u3])


    coefs, pcov = curve_fit(log_fit, z_data, U_data)
    u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    z0hi = (1/coefs[1]) #Normalized values of 'z0hi', hence z0hi/zi.
    ustar = U_mean[70]/((1/kappa)*np.log(z_d[70]/z0hi)) #Normalized values of 'ustar', hence u*/uscale.


    return(z0hi,ustar,U_mean,U_data,z_data,u_fit)


[z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,data_tavg['u'],data_tavg['v'])



print(f'The corresponding value of z0hi/zi is {z0hi}, and u*/uscale is {ustar}')


fig, ax=plt.subplots(1,2)

ax[0].plot(U_mean,z_d[0:Nz_SLayer]/canopyH,'-k')
ax[0].plot(U_data,z_data/canopyH,'ok')
ax[0].plot(u_fit,z_d[0:Nz_SLayer]/canopyH,'--k')
ax[0].hlines((canopyH-(dispH[case]/zi))/canopyH,0,30,colors='gray',linestyles='-')
ax[0].set_xlabel(r'$\overline{u}(z)/u_*$')
ax[0].set_ylabel(r'$\frac{z-d}{h}$')
ax[0].set_xlim([0, 30])
ax[0].set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])


ax[1].semilogx(z_d[0:Nz_SLayer],U_mean,'-k')
#ax.plot(U_data,z_data/canopyH,'ok')
ax[1].semilogx(z_d[0:Nz_SLayer],u_fit,'--k')
ax[1].hlines(0,z_d[0],z_d[-1],colors='gray',linestyles='-')
ax[1].plot(z0hi,0,'ok')
ax[1].set_ylabel(r'$\overline{u}(z)/u_*$')
ax[1].set_xlabel(r'$\frac{z-d}{z_i}$')
#ax.set_xlim([0, 30])
#ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])

plt.tight_layout()

#%% Compute the phi_m and plot both the derivative and mean wind speed:
   
#from scipy.optimize import curve_fit 


# Computing phiM --------------------
phi_m_3D = phi_m(nx,ny,Nz_SLayer,z_d,data_tavg['u'],data_tavg['v'],data_tavg['dudz'],data_tavg['dvdz'],ustar)

phi_m_1D = np.median(phi_m_3D,axis=(0,1))

## Plotting the Mean wind speed and the corresponding velocity gradient.

fig, ax=plt.subplots(1,1)

ax.plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-k')
ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax.vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z-d}{h}$')
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])



#%% Compute the velocity gradient for all study cases:

'''
#Stuff needed to filter a given signal:
    
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

#Low pass filter the derivative:
dydz_filt = butter_lowpass_filter(dydz, cutoff, fs, order)

'''

phiM_3D_all_cases = xr.DataArray(np.ones(shape = (Nx,Ny,Nz_SLayer,NumCases),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['hom-eq_9mps_ug','g1200_9mps_ug',\
                        'g1200_inv_9mps_ug','g800_9mps_ug','g800i_9mps_ug','g400_9mps_ug','g400i_9mps_ug']})

#Flag that determines whether we are pursuing an anisotorpy analysis.
anisotropy_analysis = 'true'
anisotropy_compute = 'false'

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
    
    
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,data.data[:,:,:,0],data.data[:,:,:,1])    
        
    z0hi_dict[cases[num]] = z0hi
    ustar_dict[cases[num]] = ustar    
        
    #Compute phi_M:
    #---------------------
    phi_m_3D = phi_m(Nx,Ny,Nz_SLayer,z_d,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,22],data.data[:,:,:,23],ustar)

    phiM_3D_all_cases[:,:,:,num] = phi_m_3D

    phi_m_1D = np.median(phi_m_3D,axis=(0,1))
    
    #ax.plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-',label=case) #Plot of all PhiM
    ax.plot(phi_m_1D,z_on_h[0:Nz_SLayer],'-',label=case) #Plot of all PhiM
    
    
    #Anisotropy analysis:
    if (anisotropy_analysis == 'true'):
        
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
    #--- End of Anisotropy analysis.
    

#Saving the phiM_3D_all_cases variable to a file.
os.chdir(directory)
phiM_3D_all_cases.to_netcdf('phiM_3D_all_cases.nc')

#ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax.vlines(1,z_on_h[0],z_on_h[-1],colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z}{h}$')
#ax.set_xlim([-0.5, 1.5])
ax.set_ylim([z_on_h[0], z_on_h[Nz_SLayer]])

axs.axhline(y = 2.754, xmin=-10, xmax=8,color='gray',linestyle='-.') #Homogenous
axs.axhline(y = 5.859, xmin=-10, xmax=8,color='gray',linestyle='-.') #g1200
axs.axhline(y = 4.557, xmin=-10, xmax=8,color='gray',linestyle='-.') #g800
axs.axhline(y = 3.255, xmin=-10, xmax=8,color='gray',linestyle='-.') #g400


#%% PLot z* as obtained from phim as a function of \lambda*Lacunarity

import pandas as pd

#From the plot in the previous cell (dudz profiles of all cases), I manually determine the height from whereon 
#Phi_m = 1. We will use this height to pull the value of z_star.
z_phiM = {'hom-eq_9mps_ug':6.06,'g1200_9mps_ug':3.99,'g1200_inv_9mps_ug':6.22,
          'g800_9mps_ug':4.26,'g800i_9mps_ug':6.48,'g400_9mps_ug':4.85,'g400i_9mps_ug':6.70}

#From the values of z_phiM (vertical axis) we can backtrack the values of z_star
z_star = {'hom-eq_9mps_ug':0,'g1200_9mps_ug':0,'g1200_inv_9mps_ug':0,'g800_9mps_ug':0,
          'g800i_9mps_ug':0,'g400_9mps_ug':0,'g400i_9mps_ug':0}


factor = 7.05*0.2
lambda_lac = {'hom-eq_9mps_ug':1*factor,'g1200_9mps_ug':0.766*factor,'g1200_inv_9mps_ug':0.2647*factor,
              'g800_9mps_ug':0.7769*factor,'g800i_9mps_ug':0.2409*factor,'g400_9mps_ug':0.7702*factor,'g400i_9mps_ug':0.7702*factor}


H = {'hom-eq_9mps_ug':0,'g1200_9mps_ug':0.2197,'g1200_inv_9mps_ug':0.6456,
              'g800_9mps_ug':0.1966,'g800i_9mps_ug':0.6566,'g400_9mps_ug':0.1953,'g400i_9mps_ug':0.6612}

Lambda_gaps = 0.2705


for num in range(0,NumCases):   
    z_star[cases[num]] = ((z_phiM[cases[num]]*(canopyH*zi)) + dispH[cases[num]])/(canopyH*zi)


##Plotting:
#----------------------------------------

colors = ['#1f77b4ff','#ff7f0eff','#2ca02cff','#d62728ff','#9467bdff','#8c564bff','#e377c2ff']


fig, ax=plt.subplots(1,1)

scatter = ax.scatter(lambda_lac.values(),z_phiM.values(),c = colors,label=case)
ax.set_ylabel(r'$(z_*-d)/h$')
ax.set_xlabel(r'$\lambda \,L_c $')


fig, ax=plt.subplots(1,1)

ax.scatter(lambda_lac.values(),z_star.values(),c = colors)
ax.set_ylabel(r'$z_*/h$')
ax.set_xlabel(r'$\lambda \,L_c $')

fig, ax=plt.subplots(1,1)
y = pd.DataFrame(z0hi_dict, index=[0])
x = pd.DataFrame(H, index=[0])
ax.scatter(x.values,y.values/canopyH,c = colors)
ax.set_ylabel(r'$z_{0,hi}/h$')
ax.set_xlabel(r'$\lambda \,H $')


#%% Plot of PhiM as a function of z/z*



fig, ax=plt.subplots(1,1)

for num in range(0,6):
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
    
    [phi_m_3D,U,ustar] = phi_m(Nx,Ny,Nz_SLayer,height,z_d,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,22],data.data[:,:,:,23],
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

    
    
#%% Graphical Representation of the results with Clustering analysis, ploting individual cluster contributions:


from scipy import stats

cmap = ColorAnisotropy()    

#Create a 3D matrix that includes the corresponding heights.
z3D = np.zeros((nx,ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,nx):
        for j in range(0,ny):
            z3D[i,j,k] = z_d[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)


num = 3
case = cases[num]
print(f'The current study case is {cases[num]}')

#Read the phi_m data from the stored file: 
phiM_3D_all_cases = xr.open_dataarray(directory + 'phiM_3D_all_cases.nc')

phi_M_1D = np.ndarray.flatten(phiM_3D_all_cases.data[:,:,:,num])

#Need to load the corresponding Anisotorpy file:
#------------------------------------------------------------------------------

path = directory + 'RAV_' + cases[num] +'/'
os.chdir(path)

print(path)

Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering.nc')
xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
AnisType_1D = np.copy(Anisotropy_clustering.data[:,2])
#------------------------------------------------------------------------------

Nclusters = 9 #Number of clusters used to group the anisotorpy.

fig, axs = plt.subplots(nrows=3,ncols=3)
plt.ion()

#Loop through all the clusters (1 to 9) organized in a 3x3 subplot..

#Overlay the median on the denisty plot:
phiM_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); phiM_median.fill(np. NaN)
yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)

y_ax = z_d[0:Nz_SLayer]/canopyH

cluster = 0
for m in range(0,3):
    for n in range(0,3):
        
        cluster = cluster + 1
        
        print(f'cluster = {cluster}')

        tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
        tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
        tmp_yB = yB_1D[(AnisType_1D == cluster)]
        
        NumPoints = np.size(tmp_yB) #Total number of points in a given cluster. This should be larger than 100 to do statistics.
        
        print(f'Total number of points in cluster = {NumPoints}')

        if (cluster <= 9 and NumPoints > 100):
            x = tmp_phi_u
            y = tmp_z3D/canopyH
    

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


# plt.savefig(path +'Figures/'+'dUdz_anisotropy_subplots_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')

#%% Graphical Representation of the results with Clustering in a single subplot:

import matplotlib.ticker


fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

#Median of the gradient profiles as a function of cluster and height. 
#For a fixed cluster (e.g. cluster = 4), we average all values of the gradient
#at a specific height.
 
for cluster in range(1,Nclusters):
    #print(f'{np.count_nonzero(np.isnan(yB_median[cluster,0:-1]))}')
    if (np.count_nonzero(np.isnan(yB_median[cluster,0:-1])) < Nz_SLayer-1):
        #sc = axs.scatter(phiM_median[cluster,0:-1],(z_d[0:Nz_SLayer-1]/canopyH),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
        sc = axs.scatter(phiM_median[cluster,0:-1],(z_on_h[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs.plot(np.mean(phiM_3D_all_cases.data[:,:,:,num],axis=(0,1)),(z_on_h[0:Nz_SLayer]),linestyle='-',color='k')


axs.set_xlim(-0.5, 2)
#axs.set_ylim(y_ax[0], y_ax[Nz_SLayer-1])
axs.set_ylim(z_on_h[0], z_on_h[Nz_SLayer-1])
#axs.axhline(y = (canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle=':')
#axs.axhline(y = 3*(canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')
axs.axhline(y = (canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle=':')
#axs.axhline(y = 3*(canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle='-.')

#axs.axhline(y = 2.754, xmin=-10, xmax=8,color='gray',linestyle='-.') #Homogenous
#axs.axhline(y = 5.859, xmin=-10, xmax=8,color='gray',linestyle='-.') #g1200
#axs.axhline(y = 4.557, xmin=-10, xmax=8,color='gray',linestyle='-.') #g800
axs.axhline(y = 3.255, xmin=-10, xmax=8,color='gray',linestyle='-.') #g400



axs.set_xscale('linear')
cbar = plt.colorbar(sc)
#axs.set_ylabel(r'$(z-d)/h$')
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(r'$\phi_M$')
axs.set_title(f'{cases[num]}')


axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')


plt.tight_layout()
plt.show()

# plt.savefig(path +'Figures/'+'dUdz_anisotropy_allinone_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')



#%% Combining the Anisotropy analysis with the TKE analysis:

#list of study cases:
#['hom-eq_9mps_ug','g1200_9mps_ug','g1200_inv_9mps_ug','g800_9mps_ug','g800i_9mps_ug','g400_9mps_ug','g400i_9mps_ug']    

num = 5
case = cases[num]

print(f'The current study case is {cases[num]}')


#Need to load the corresponding Anisotorpy file:
#------------------------------------------------------------------------------

path = directory + 'RAV_' + cases[num] +'/'
os.chdir(path)

print(path)

Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering.nc')
xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
AnisType_1D = np.copy(Anisotropy_clustering.data[:,2])
#------------------------------------------------------------------------------

Nclusters = 9 #Number of clusters used to group the anisotorpy.

#Need to load the corresponding TKE file:
#------------------------------------------------------------------------------

path_in = directory + '/RAV_' + cases[num] +'/TKE_RAV_Output/' 
print(path_in)

terms_bdg = np.load(path_in + 'TKE_terms.npy',allow_pickle='TRUE').item()




#%% Analyze the TKE terms as a function of Turb. Anisotropy

#-------Set the TKE terms to study:--------------------------------------------
n = 5
labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.
#keys = ['adv_h','adv_v','prod_h','prod_v','uturb_h','uturb_v','pturb_h','pturb_v','prod_dudz','canopy','dissip','sum']

terms = [terms_bdg['prod_v'],-terms_bdg['dissip'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
         terms_bdg['pturb_h'] + terms_bdg['pturb_v'],terms_bdg['prod_v'] - terms_bdg['dissip'],
         terms_bdg['adv_v'] + terms_bdg['adv_h']]

TKE_term = terms[n]



#-------Run the script --------------------------------------------


#------------------------------------------------------------------------------

#Create a 3D matrix that includes the corresponding heights.
z3D = np.zeros((nx,ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,nx):
        for j in range(0,ny):
            z3D[i,j,k] = z_on_h[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)

#----------------------------------------


TKE_term_1D = np.ndarray.flatten(TKE_term[:,:,0:Nz_SLayer])

#Overlay the median on the denisty plot:
TKE_term_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); TKE_term_median.fill(np.NaN)
yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)

fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

cmap = ColorAnisotropy()  

cluster = 0
for cluster in range(1,10):
        
    print(f'cluster = {cluster}')
    tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster)]
    tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
    tmp_yB = yB_1D[(AnisType_1D == cluster)]
    
    for i in range(0,Nz_SLayer-1):
        TKE_term_median[cluster-1,i] = np.median(tmp_TKE_term_1D[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])
        yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])     
    
        
    print(f'Done with cluster {cluster}')
        
    if (np.size(tmp_TKE_term_1D) > 0):
        sc = axs.scatter(TKE_term_median[cluster-1,:],z_on_h[0:Nz_SLayer],s=10,marker='o',c = yB_median[cluster-1,:],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)


axs.plot(np.median(TKE_term[:,:,0:Nz_SLayer],axis=(0,1)),z_on_h[0:Nz_SLayer],'-k')

#------------------------------------------------------------------------------


x_min = -1500
x_max = 1500

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
axs.set_xlabel(f'{labels[n]}')


plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+'Anisotropy_TKE_'+ labels[n] + '_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')




#%% New Colormap:
    
    
    
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



#%% Some Pcolor figures:

iva_colors = ["#410d00","#831901","#983e00","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]

yB = np.reshape(yB_1D,(nx,ny,Nz_SLayer))
#height = 24
height = 14  

fig, axs = plt.subplots(nrows=1,ncols=1)

sc = axs.contourf(np.arange(0,nx)*dx,np.arange(0,ny)*dy,np.transpose(yB[:,:,height]),levels=9,colors=iva_colors)
#sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,np.transpose(yB[:,:,height]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)


cbar = plt.colorbar(sc)
    
axs.set_ylabel(r'$y/z_i$')
axs.set_xlabel(r'$x/z_i$')
axs.set_title(r'$y_B(z/h = 2.5)$')


plt.tight_layout()


# plt.savefig(path_in + 'Figures/' + 'yB' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')



#%% #%% Gathering the desired TKE term for analysis as a function of yB cluster:


import seaborn as sns
import pandas as pd

#-------Set the TKE terms to study:--------------------------------------------
n = 4
labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.
#keys = ['adv_h','adv_v','prod_h','prod_v','uturb_h','uturb_v','pturb_h','pturb_v','prod_dudz','canopy','dissip','sum']

#TKE_term = terms_bdg['prod_v']
#TKE_term = -terms_bdg['dissip']
#TKE_term = terms_bdg['uturb_h'] + terms_bdg['uturb_v']
#TKE_term = terms_bdg['pturb_h'] + terms_bdg['pturb_v']
TKE_term = terms_bdg['prod_v'] - terms_bdg['dissip']

#------------------------------------------------------------------------------


TKE_term_1D = np.ndarray.flatten(TKE_term[:,:,0:Nz_SLayer])

TKE_cluster = {}  #Note, I am not using a pd.DataFrame because in there all columns need to be the same length.
                  #I could  make the columns of same lengths by adding NaNs but that would increase the memory needs.

nPoints_Cluster = {} #dictionary that keeps track of the total number of points per cluster
Total_nPoints = 0
cluster = 0
for cluster in range(1,10):
    print(f'cluster = {cluster}')
    #tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster)]
    cluster_name = 'cluster ' + str(cluster)
    TKE_cluster[cluster_name] = TKE_term_1D[(AnisType_1D == cluster)]
    nPoints_Cluster[cluster_name] = TKE_cluster[cluster_name].size
    Total_nPoints = Total_nPoints + nPoints_Cluster[cluster_name] 

df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in TKE_cluster.items()]))

    
#%% PLotting the Results as a Boxplot


fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(8, 5))

#Box Plot:
#axs.boxplot(TKE_cluster.values(),vert=False, notch = False, showfliers = False,
#            patch_artist = True,boxprops = dict(facecolor = "lightblue"),
#            medianprops = dict(color = "orange", linewidth = 1),
#            whiskerprops = dict(color = "red", linewidth = 1),
#            capprops = dict(color = "red", linewidth = 1))

#axs.set_yticklabels(TKE_cluster.keys())

#Seaborn Boxplot Plot:
cm = sns.color_palette( ["#410d00","#831901","#983e00","#b56601","#ab8437",
                         "#b29f74","#7f816b","#587571","#596c72"])

sns.boxplot(data=df,orient='h',palette=cm,saturation=1,width=0.6,linewidth=0.8,fliersize = 0.5)


x_min = -1000
x_max = 1000

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
plt.show()


# plt.savefig(path_in +'Figures/'+ labels[n] + '_boxplot_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')


#%% PLotting the total number of points per cluster:

percentage_Cluster = (pd.Series(nPoints_Cluster.values())/Total_nPoints)*100

fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(6, 3))

axs.scatter(np.arange(1,10),percentage_Cluster,c=["#410d00","#831901","#983e00","#b56601","#ab8437",
                         "#b29f74","#7f816b","#587571","#596c72"])

axs.set_xlabel(r'$cluster$')
axs.set_ylabel(r'$N_i/N_{total}$')

plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+ labels[n] + '_Npoints_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')




#%% PLotting the TKE Residual as a function of yB

'''
import matplotlib.colors as mcolors


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

norm = MidpointNormalize( midpoint = 1 )
'''

#-------Set the TKE terms to study:--------------------------------------------
n = 4
labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.
#keys = ['adv_h','adv_v','prod_h','prod_v','uturb_h','uturb_v','pturb_h','pturb_v','prod_dudz','canopy','dissip','sum']

#TKE_term = terms_bdg['prod_v']
#TKE_term = -terms_bdg['dissip']
#TKE_term = terms_bdg['uturb_h'] + terms_bdg['uturb_v']
#TKE_term = terms_bdg['pturb_h'] + terms_bdg['pturb_v']
TKE_term = terms_bdg['prod_v'] - terms_bdg['dissip']

#------------------------------------------------------------------------------

#Create a 3D matrix that includes the corresponding heights.
z3D = np.zeros((nx,ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,nx):
        for j in range(0,ny):
            z3D[i,j,k] = z_on_h[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)

#----------------------------------------



TKE_term_1D = np.ndarray.flatten(TKE_term[:,:,0:Nz_SLayer])

fig, axs = plt.subplots(nrows=1,ncols=1)



cluster = 0
for cluster in range(1,10):
        
    print(f'cluster = {cluster}')
    tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster)]
    tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
    tmp_yB = yB_1D[(AnisType_1D == cluster)]
    
    for i in range(0,Nz_SLayer-1):
        TKE_term_median[cluster-1,i] = np.median(tmp_TKE_term_1D[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])
        yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])     
    
        
    print(f'Done with cluster {cluster}')
    
    y_canopy = 10 #Index for which z/h > 1; hence it is the minimum height above which we are above the canopy
        
    if (np.size(tmp_TKE_term_1D) > 0):
        #sc = axs.scatter(TKE_term_median[cluster-1,:],yB_median[cluster-1,:],s=10,marker='o',c = yB_median[cluster-1,:],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
#        sc = axs.scatter(TKE_term_median[cluster-1,:],yB_median[cluster-1,:],s=10,marker='o',c = z_on_h[0:Nz_SLayer], alpha=0.7,cmap='bwr',norm = norm)
            sc = axs.scatter(TKE_term_median[cluster-1,y_canopy:Nz_SLayer],yB_median[cluster-1,y_canopy:Nz_SLayer],s=10,marker='o',c = z_on_h[y_canopy:Nz_SLayer], alpha=0.7,cmap='hot', vmin = z_on_h[0],vmax = z_on_h[Nz_SLayer])



#sc = axs.scatter(TKE_term_1D,yB_1D,s=10,marker='o',c = yB_1D,alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

y_min = 0
y_max = np.sqrt(3)/2

x_min = -100
x_max = 500

axs.set_xlim(x_min, x_max)
axs.set_ylim(y_min, y_max)


axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xscale('linear')
    
cbar = plt.colorbar(sc)
    
axs.set_ylabel(r'$y_B$')
axs.set_xlabel(f'{labels[n]}')

plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+ labels[n] + '_vs_yB_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')











#%% Continuation of the analysis:

#1) Investigating the sigma_w vertical profiles (as per the suggestion of Katul et al 2023 work in Physics of Fluids):

fig, axs=plt.subplots(1,1)

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
    
    Dww = np.mean((data.data[:,:,:,2]*data.data[:,:,:,2]),axis=(0,1)) - (np.mean(data.data[:,:,:,2],axis=(0,1))*np.mean(data.data[:,:,:,2],axis=(0,1)))
    
    sigma_w = (np.mean((Rstress.data[:,:,:,2] + data.data[:,:,:,18]),axis=(0,1)) + Dww)/ustar_dict[cases[num]]
    
    axs.plot(sigma_w[0:Nz_SLayer],z_d[0:Nz_SLayer]/canopyH,'-',label=case) #Plot of all sigma_w


axs.axhline(y = (canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle=':')
axs.axhline(y = 3*(canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')

axs.set_ylabel(r'$(z-d)/h$')
axs.set_xlabel(r'$\sigma_w$')
axs.set_title(f'{cases[num]}')

plt.tight_layout()
plt.show()
    




#%% Dispersive TKE Production profile:
    

Disp_Rxz = DispFluct(nx, ny, Nz_SLayer, Rstress.data[:,:,0:Nz_SLayer,4])
Disp_Ryz = DispFluct(nx, ny, Nz_SLayer, Rstress.data[:,:,0:Nz_SLayer,5])

Disp_dUdz = DispFluct(nx, ny, Nz_SLayer, data.data[:,:,0:Nz_SLayer,22])
Disp_dVdz = DispFluct(nx, ny, Nz_SLayer, data.data[:,:,0:Nz_SLayer,23])

Disp_Prod = np.mean((Disp_Rxz*Disp_dUdz + Disp_Ryz*Disp_dVdz),axis=(0,1))*(canopyH/ustar_dict[cases[num]]**3)
Turb_Prod = np.mean((Rstress.data[:,:,0:Nz_SLayer,4]*data.data[:,:,0:Nz_SLayer,22]+Rstress.data[:,:,0:Nz_SLayer,5]*data.data[:,:,0:Nz_SLayer,23]),axis=(0,1))*(canopyH/ustar_dict['hom-eq_9mps_ug']**3)

fig, axs=plt.subplots(1,1)
axs.plot(Disp_Prod[0:Nz_SLayer],z_on_h[0:Nz_SLayer],'--k',label='D_Prod')
axs.plot(Turb_Prod[0:Nz_SLayer],z_on_h[0:Nz_SLayer],'-k',label='T_Prod')
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(r'$Prod$')
axs.set_title(f'{cases[num]}')
axs.set_ylim(0,z_on_h[Nz_SLayer])

plt.legend()
plt.tight_layout()
plt.show()

    

