# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 01:50:44 2025

@author: udina
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

#%%Set case and path to data


# Path to the data files:
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
os.chdir(path)  

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps','Empty_9mps']
case = 0

#%% Defining the main parameters of the simulations

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_w = np.arange(0,Nz)*dz
z_uvp = np.arange(0,Nz)*dz + dz/2

zi = 1000
uscale = 0.313
Hcanopy = 39/zi
kappa = 0.4  # von Karman constant
LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
height = math.ceil(Hcanopy/dz)

Ntwr = 100
Nz_Slayer = 126

# Loading the 3D Momentum, 2D Momentum, and 3D TKE Budget data:

#For the sake of clarity, below I sepcify the variables included in "data" and "dataS":
#--------------------------------------------------------------------------------------------------
#data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                        'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                        'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                        'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                        'avgdudz','avgdvdz','avgNut','avgCs']})



#data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
#                dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})


#data_TKE = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u','v','w','p',\
#                        'uu','vv','ww','uv','uw','vw',\
#                        'dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
#                        'txx','tyy','tzz','txy','txz','tyz',\
#                        'uuu','uvv','uww','vuu','vvv','vww','wuu','wvv','www',\
#                        'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz',\
#                        'vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
#                        'up','vp','wp',\
#                        'dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']}) 

#%%Load the data
    
data = xr.open_dataarray(cases[case]+'/Data_Momentum_4TKE.nc')
terms_ptb = xr.open_dataarray(path+cases[case]+'/terms_ptb.nc')
terms_bdg = xr.open_dataarray(path+cases[case] + '/TKE_terms.nc')
anisotropy = xr.open_dataarray(path+cases[case] + '/anisotropy.nc')

#%%Check anisotropy

xB = anisotropy[:,:,:,0]
yB = anisotropy[:,:,:,1]

if np.any(xB < 0) or np.any(yB < 0):
    print(f"Anisotropy has negative values")
    if np.any(xB < 0) and np.any(yB < 0):
        print("Both xB and yB have negatives")
    elif np.any(xB < 0):
        print("Only xB has negatives")
    elif np.any(yB < 0):
        print("Only yB has negatives")
else:
    print(f"Anisotropy has all positive values")

#%%Plot Anisotropy

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
xslice = 100
yslice = 100
zslice = 2

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,6))

axs[0,0].pcolormesh(x,y,xB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = 1)
axs[0,1].pcolormesh(x,z_uvp,xB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = 1)
p1 = axs[0,2].pcolormesh(y,z_uvp,xB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = 1)

axs[1,0].pcolormesh(x,y,yB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
axs[1,1].pcolormesh(x,z_uvp,yB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
p2 = axs[1,2].pcolormesh(y,z_uvp,yB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)

# 1. x-y plane @ zslice
neg_xB = np.where(xB[:, :, zslice] < 0)
axs[0, 0].plot(x[neg_xB[0]], y[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[:, :, zslice] < 0)
axs[1, 0].plot(x[neg_yB[0]], y[neg_yB[1]], 'r.', markersize=10)

# 2. x-z plane @ yslice
neg_xB = np.where(xB[:, yslice, :] < 0)
axs[0, 1].plot(x[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[:, yslice, :] < 0)
axs[1, 1].plot(x[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=10)

# 3. y-z plane @ xslice
neg_xB = np.where(xB[xslice, :, :] < 0)
axs[0, 2].plot(y[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[xslice, :, :] < 0)
axs[1, 2].plot(y[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=10)

axs[1,1].contour(x,z_uvp,yB[:,yslice,:].T,levels=[0.38],colors='black')
axs[1,2].contour(y,z_uvp,yB[xslice,:,:].T,levels=[0.38],colors='black')

cbar1 = plt.colorbar(p1,label='xB')
cbar2 = plt.colorbar(p2,label='yB')

axs[0,0].set_title(f"zslice = {zslice}",fontsize=15)
axs[0,1].set_title(f"yslice = {yslice}",fontsize=15)
axs[0,2].set_title(f"xslice = {xslice}",fontsize=15)

plt.show()

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%Compute the residual
    
Res = (terms_bdg.data[:,:,:,-1]+terms_bdg.data[:,:,:,11])#*(Hcanopy)/(ustar[:,:,np.newaxis]**3)
# ResNorm[(abs(ResNorm) < 50)] = 0

#%%Compute ustar

T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))

cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
ustar = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

del T_13,T_23,cov_turb

#%%Normalize the residual with canopy H and ustar at top of the canopy

ResNorm = Res*(Hcanopy)/(ustar[:,:,np.newaxis]**3)
# ResNorm[(abs(ResNorm) < 0.3)] = 0


#%%Plot ustar 2D

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,ustar.T,cmap='jet',vmin=0,vmax=3)
# axs.axhline(167*dy,c='k')

cbar = plt.colorbar(p)

plt.show()

#%%Pcolor plots of Residual
# Res_yavg = np.mean(ResNorm,axis=(1))
# ustar_yavg = np.mean(ustar,axis=(1))
yslice = 190

fig,axs = plt.subplots(1,1,tight_layout=True)

# p = axs.contourf(x,z_uvp/(39/zi),(Res_yavg*Hcanopy/ustar_yavg[:,np.newaxis]**3).T,cmap='bwr',levels=[-100,-0.14,0.14,100])
# p = axs.pcolormesh(x,z_uvp/(39/zi),np.mean(ResNorm_v2,axis=(0)).T,cmap='bwr',vmin=-5,vmax=5)
p = axs.pcolormesh(x,z_uvp/(39/zi),ResNorm[:,yslice,:].T,cmap='bwr',vmin=-5,vmax=5)
# p = axs.pcolormesh(x,z_uvp/(39/zi),ResNorm[:,yslice,:].T,cmap='bwr',vmin=-500,vmax=500)
# p = axs.pcolormesh(x,z_uvp,terms_bdg.data[:,:,20,8].T,cmap='bwr',vmin=-500,vmax=500)
# p = axs.pcolormesh(x,y,terms_bdg.data[:,:,20,8].T,cmap='bwr',vmin=-500,vmax=500)

# axs.axhline(Hcanopy,ls='--',c='k')
# axs.axhline(230*dy,ls='--')
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$z/z_i$',fontsize=15)
axs.set_title(f'{cases[case]}',fontsize=15)
axs.set_ylim(0,8)

cbar = plt.colorbar(p)

plt.show()

#%%Plot Anisotropy

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()

yslice = 190

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
levels=np.linspace(0,np.sqrt(3)/2,10)
plt1=axs.contourf(x,z_uvp/Hcanopy,anisotropy.data[:,yslice,:,1].T,cmap=cmap,levels=levels,vmin=0,vmax=np.sqrt(3)/2)
axs.contour(x,z_uvp/Hcanopy,anisotropy.data[:,yslice,:,1].T,levels=[0.38],colors='black')
# plt1=axs.contourf(x,z[0:Nz_SLayer],np.transpose(np.mean(yB,axis=1)),cmap=cmap,levels=8)
# axs.pcolormesh(x,z,np.transpose(mask[:,Ny_Slice,:]),cmap='gray')
# axs.plot(x,z_tpg/zi,color='black')
axs.axhline(1,ls='--',c='k')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
axs.set_title(f'{cases[case]}',fontsize=15)
# plt.savefig(figPath+'yB.png',dpi=300,facecolor='white', edgecolor='white')


plt.show()

#%%Profile of the residual

# RESprof = np.mean(terms_bdg[:, :, :, 2]+terms_bdg[:, :, :, 5]+terms_bdg[:, :, :, 8]+terms_bdg[:, :, :, 11]+terms_bdg[:, :, :, 14],axis=(0,1))
RESprof = np.mean(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11],axis=(0,1))

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(6,6))

axs.plot(RESprof,z_uvp/Hcanopy,'k',label='Residual')
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.set_xlabel(r"$<\overline{\frac{\partial e}{\partial t}}>$", fontsize=15)
axs.set_ylim(0,(z_uvp/Hcanopy)[-1])
axs.axhline(1,c='k',ls='--')
axs.grid()

plt.show()

#%% Plot TKE residual normalized by Dissipation with yB contour line

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
# tmpRES = np.where(np.abs(ResNorm_v2) < 0.14, 0, tmpRES)
tmpRES = np.where(np.abs(Res) < 20, 0, tmpRES)

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
levels=[-100,-1,1,100]
levels_2 = [0.36]
colors=['blue','white','red']

yslice = 190

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.contourf(x,z_uvp/(39/zi),tmpNorm[:,yslice,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')

p.cmap.set_under('blue')
p.cmap.set_over('red')

sc = axs.contour(x,z_uvp/(39/zi),anisotropy[:,yslice,:,1].T,levels=levels_2,colors=['black'])
axs.axhline(1,ls='--',c='k')
# axs.set_ylim(0,8)

plt.show()

#%%Plot the correlation the scatter distribution and correlation

import numpy as np
import matplotlib.pyplot as plt
import copy

# Optional: remove if not needed
# from scipy.stats import gaussian_kde

l = 0
level = 5

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
# tmpDIS[(np.abs(tmpDIS.values)<10)] = 0 
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
# tmpRES = np.where(np.abs(ResNorm) < 10, 0, tmpRES)
# tmpRES = np.where(np.abs(ResNorm_v2) < 0.05, 0, tmpRES)

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

# Select y_B and TKE slices
x_yB = anisotropy[:, :, :60, 1]
y_TKE = tmpNorm[:, :, :60]

# Compute binned statistics
TKE_median = []
TKE_std = []
yB_mean = []

j = 0.1
for i in range(28):
    if i == 0:
        mask = x_yB < j
        yB_mean.append(0.05)
    else:
        mask = (x_yB > j) & (x_yB < j + 0.025)
        yB_mean.append(j + 0.0125)
    TKE_vals = y_TKE[mask]
    TKE_median.append(np.nanmedian(TKE_vals))
    TKE_std.append(np.nanstd(TKE_vals))
    j += 0.025

# Plot results
axs.plot(yB_mean, TKE_median, c='k', marker='o')
axs.fill_between(
    yB_mean,
    np.array(TKE_median) - np.array(TKE_std),
    np.array(TKE_median) + np.array(TKE_std),
    alpha=0.5,
)

axs.axhline(0, color='k', linestyle='-.')
axs.axvline(0.38, color='k', linestyle='-.')
axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
axs.axvline(0.36, color='k', linestyle='-.')
axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)
axs.set_title(f'{cases[case]}',fontsize=15)

# Correlation text
# corr = np.corrcoef(x_yB.values.flatten(), y_TKE.flatten())[0, 1]
# axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

plt.tight_layout()
plt.show()


#%%Compute ustar at the w node

# 1. Get turbulent covariances
T_13 = ((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) + (data.data[:,:,:,23]))
T_23 = ((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) + (data.data[:,:,:,24]))

# 2. Compute friction velocity u*
# cov_turb = -np.sqrt(T_13**2 + T_23**2)
# us_2d = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height
# us_prof = np.sqrt(-cov_turb)
# us_prof = us_prof.mean(axis=(0, 1))  # average over horizontal plane

# 3. Dispersive component
# Remove planar mean at each height level
u_pp = data.data[:,:,:,0] - data.data[:,:,:,0].mean(axis=(0, 1), keepdims=True)
v_pp = data.data[:,:,:,1] - data.data[:,:,:,1].mean(axis=(0, 1), keepdims=True)
w_pp = data.data[:,:,:,2] - data.data[:,:,:,2].mean(axis=(0, 1), keepdims=True)

# Compute dispersive stresses
D13 = uvpnode2wnode(u_pp) * w_pp
D23 = uvpnode2wnode(v_pp) * w_pp

cov_turb = -np.sqrt((T_13 + D13)**2 + (T_23 + D23)**2)
ustar_w = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

del T_13,T_23,u_pp,v_pp,w_pp,D13,D23,cov_turb

#%%TKE budget profile

prod = np.mean(uvpnode2wnode(terms_bdg[:,:,:,-1]),axis=(0,1))*Hcanopy/(np.mean(ustar_w)**3)
diss = np.mean(uvpnode2wnode(terms_bdg[:,:,:,10]),axis=(0,1))*Hcanopy/(np.mean(ustar_w)**3)
ptrans = np.mean(uvpnode2wnode(terms_bdg[:,:,:,8]),axis=(0,1))*Hcanopy/(np.mean(ustar_w)**3)
ttrans = np.mean(uvpnode2wnode(terms_bdg[:,:,:,5]),axis=(0,1))*Hcanopy/(np.mean(ustar_w)**3)

fig,axs = plt.subplots(1,1,figsize=(6,8),tight_layout=True)

axs.plot(prod,z_w/(39/zi),c='b')
axs.plot(-diss,z_w/(39/zi),c='g')
axs.plot(ptrans,z_w/(39/zi),c='r')
axs.plot(ttrans,z_w/(39/zi),c='k')

axs.set_ylim(0,5)
axs.set_xlim(-8,8)
axs.axhline(1,c='k',ls='--')
plt.show()


































