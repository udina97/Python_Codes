#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:33:14 2024

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/')

from functions import load_3d_UCLAdata, load_UCLAnpy_files, load_2d_UCLAdata

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc
from Stats import  ReynoldsStress, DispFluct

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

#%% Simulation parameters

sim = 'Giulia_Flat_nocanopy'
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/Giulia_Flat_nocanopy/'

Nx = 256
Ny = 256
Nz = 256
lx = 2*np.pi
ly = 2*np.pi
lz = 1
dx = lx/Nx
dy = ly/Ny
dz = lz/Nz

z_uvp = np.arange(0, Nz) * dz + 0.5 * dz
z_w = np.arange(0, Nz) * dz

Nz_SLayer = Nz


# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.05 #05; %0.000005;
zi = 1000.0
uscale = 0.313 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000
kappa = 0.41
canopyH = 1

plot_profile = 0
plot_color = 0

#%% Import data and TKE terms

var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs']
    
varTKE = ['avgU','avgV','avgW','avgP','avgUU','avgVV','avgWW','avgUV','avgUW','avgVW','avgDUDX','avgDUDY','avgDUDZ','avgDVDX','avgDVDY','avgDVDZ',\
          'avgDWDX','avgDWDY','avgDWDZ','avgTXX','avgTYY','avgTZZ','avgTXY','avgTXZ','avgTYZ','avgUUU','avgUVV','avgUWW',\
          'avgVUU','avgVVV','avgVWW','avgWUU','avgWVV','avgWWW','avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ',\
          'avgWTXX','avgWTYY','avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ','avgUTXZ','avgVTYZ','avgUP','avgVP','avgWP','avgDXX','avgDYY','avgDZZ',\
          'avgDXY','avgDXZ','avgDYZ','avgFDX','avgFDY','avgFDZ','avgUFDX','avgVFDY','avgWFDZ']
    
varSGS = ['avgTXX','avgTYY','avgTZZ','avgTXY','avgTXZ','avgTYZ','avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ','avgWTXX','avgTYY',\
          'avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ','avgUTXZ','avgVTYZ','avgDXX','avgDYY','avgDZZ','avgDXY','avgDXZ','avgDYZ']
    

dataM = xr.open_dataarray(path+'Data_Momentum_5hr.nc')
dataTKE = xr.open_dataarray(path+'Data_GigiTKE_5hr.nc')

data_mom = dict()
data_tke = dict()

for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
for i in range(len(varTKE)):
    data_tke[varTKE[i]] = dataTKE.data[:,:,:,i]
    
for i in range(len(varSGS)):
    data_tke[varSGS[i]] = -data_tke[varSGS[i]]
    
terms_bdg = np.load(path + 'TKE_terms.npy',allow_pickle='TRUE').item()

#%% pcolor test plots

# tmp = copy.deepcopy(data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))
# tmp = copy.deepcopy(terms_ptb['pu_t'])
# tmp = copy.deepcopy((terms_bdg['prod']-terms_bdg['totdis'])/abs(terms_bdg['totdis']))
tmp = copy.deepcopy(terms_bdg['prod'] + terms_bdg['totdis'])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,Nx)*dx
z_ax = np.arange(0,Nz)*dz

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

levels=[-100,-4,-1,1,4,100]
colors=['blue','green','white','yellow','red']

plt1 = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi),tmp[:,150,:].T,levels=levels,colors=colors)
# sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,np.transpose(yB[:,yslice,:]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

#y-vorticity vertical slice
# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,150,:].T,cmap= 'bwr',vmin=-30,vmax=30)
# axs.streamplot(x_ax, z_ax, DispFluct_withTopo(data_tavg['u'],dist,dm).T, DispFluct_withTopo(data_tavg['w'],dist,dm).T,
#                density = 1,color=[0.7,0.7,0.7])
# axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
axs.set_ylim(z_ax[0],z_ax[-1])
# axs.set_title('Vorticity - y & Dispersive Streamlines')
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.5])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'pu_avg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()


#%% Import/Compute anisotropy

anisotropy_analysis = 'true'
anisotropy_compute = 'false'

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,data_tke['avgU'],data_tke['avgV'],data_tke['avgW'],data_tke['avgUU'],\
                          data_tke['avgVV'],data_tke['avgWW'],data_tke['avgUV'],data_tke['avgUW'],data_tke['avgVW'])


#Anisotropy analysis:
if (anisotropy_analysis == 'true'):
    
    if (anisotropy_compute == 'true'):
        
        [xB,yB,AnisType_1D] = Anisotropy_Clustering(Nx,Ny,Nz,Rstress)

        yB_1D = np.ndarray.flatten(yB)
        xB_1D = np.ndarray.flatten(xB)


        Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz,3),order='F'),\
                            dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
            
        Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

        os.chdir(path)
        Anisotropy_clustering.to_netcdf('Anisotropy_clustering_'+sim+'.nc')
        
    else:
            
        Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering_'+sim+'.nc')

        xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
        yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
        AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 

    #------------ End of the IF statement.
#--- End of Anisotropy analysis.


#Saving the phiM_3D_all_cases variable to a file.
os.chdir(path)

#%%

z_d = z_uvp
 
#%% Compute the corresponding values of z0hi and u* above the canopy:

from scipy.optimize import curve_fit     

# Nz_SLayer = Nz-1
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
    U_mean = np.nanmean(U[:,:,0:Nz_SLayer],axis=(0,1))
    # U_mean = U[0:Nz_SLayer]

    #Data used for the Inertial logarithmic fit above the RSL. --------------
    #------------------------------------------------------------------------
    level1 = 20 #70
    level2 = 30 #95
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

[z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,data_tke['avgU'],data_tke['avgV'])

print(f'The corresponding value of z0hi/zi is {z0hi}, and u*/uscale is {ustar}')


fig, ax=plt.subplots(1,2)

ax[0].plot(U_mean,z_d[0:Nz_SLayer]/canopyH,'-k',label='$\overline{U}$')
ax[0].plot(U_data,z_data/canopyH,'ok',label='$\overline{U}$ @ lvl 1,2')
ax[0].plot(u_fit,z_d[0:Nz_SLayer]/canopyH,'--k',label='log fit')
# ax[0].hlines((canopyH-(dispH[case]/zi))/canopyH,0,30,colors='gray',linestyles='-')
ax[0].set_xlabel(r'$\overline{u}(z)/u_*$')
ax[0].set_ylabel(r'$\frac{z-d}{h}$')
ax[0].set_xlim([0, 30])
ax[0].set_ylim([z_d[0]/canopyH, z_d[-1]/canopyH])
ax[0].legend()


ax[1].semilogx(z_d[0:Nz_SLayer],U_mean,'-k',label='$\overline{U}$')
#ax.plot(U_data,z_data/canopyH,'ok')
ax[1].semilogx(z_d[0:Nz_SLayer],u_fit,'--k',label='log fit')
ax[1].hlines(0,z_d[0],z_d[-1],colors='gray',linestyles='-')
ax[1].plot(z0hi,0,'ok',label='$z_{0,hi}$')
ax[1].set_ylabel(r'$\overline{u}(z)/u_*$')
ax[1].set_xlabel(r'$\frac{z-d}{z_i}$')
ax[1].legend()
#ax.set_xlim([0, 30])
#ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])

plt.tight_layout()

#%% Compute velocity gradient phi

phi_m_3D = phi_m(Nx,Ny,Nz_SLayer,z_d,data_tke['avgU'],data_tke['avgV'],data_tke['avgDUDZ'],data_tke['avgDVDZ'],ustar)
phi_m_1D = np.nanmedian(phi_m_3D,axis=(0,1))


## Plotting the Mean wind speed and the corresponding velocity gradient.

fig, ax=plt.subplots(1,1)

ax.plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-k')
# ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax.vlines(1,z_d[0]/canopyH,z_d[-1]/canopyH,colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z-d}{h}$')
ax.set_xlim([-0.5, 2])
ax.set_ylim([z_d[0]/canopyH, z_d[-1]/canopyH])


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

#%%
yB = np.reshape(yB_1D,(Nx,Ny,Nz))
# yB[(dist[:,:,:]<0)] = float('nan')

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

terms = [(terms_bdg['prod']) + (terms_bdg['totdis']),terms_bdg['uturb_h'] + terms_bdg['uturb_v'], terms_bdg['pturb_h'] + terms_bdg['pturb_v'],\
         terms_bdg['adv_v'] + terms_bdg['adv_h']]

n = 0
    
TKE_term = terms[n]
tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
# tmp_TKE_term[(dist<0)] = float('nan')

xslice = 150

fig, axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)
sc = axs.scatter(yB[xslice,:,:],tmp_TKE_term[xslice,:,:],c=yB[xslice,:,:],cmap=cmap)
axs.set_xlabel('yB')
axs.set_ylabel('TKE term')
cbar = plt.colorbar(sc,label='yB')
axs.set_title(f'{labels[n]}')
fig.suptitle(f'$x = {xslice*dy*zi}m$')

# print(f'The correlation coefficient is {np.corrcoef(yB[xslice,:,:][~np.isnan(yB[xslice,:,:])],tmp_TKE_term[xslice,:,:][~np.isnan(tmp_TKE_term[xslice,:,:])])[0,1]}')
# print(f'The correlation coefficient is {np.corrcoef(yB[:,yslice,:][~np.isnan(yB[:,yslice,:])],tmp_TKE_term[:,yslice,:][~np.isnan(tmp_TKE_term[:,yslice,:])])[0,1]}')

# from scipy.stats import gaussian_kde

# # Generate fake data
# x_data = tmp_TKE_term[~np.isnan(tmp_TKE_term)]
# y_data = yB[~np.isnan(yB)]

# # Calculate the point density
# xy = np.vstack([x_data,y_data])
# z = gaussian_kde(xy)(xy)

# fig, ax = plt.subplots()
# ax.scatter(x_data, y_data, c=z, s=100)
# plt.show()


# import scipy as sp
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.use('Agg')
# sns.set_theme(style="ticks")

# x_data = tmp_TKE_term[~np.isnan(tmp_TKE_term)]
# y_data = yB[~np.isnan(yB)]
# sns.scatterplot(x=x_data, y=y_data)


# plt.xlabel('TKE term')
# plt.ylabel('yB')

# r, p = sp.stats.pearsonr(x=x_data, y=y_data)
# ax = plt.gca() # Get a matplotlib's axes instance
# plt.text(.05, .8, "Pearson's r ={:.2f}".format(r), transform=ax.transAxes)

#%% Some Pcolor figures for yB 

iva_colors = ["#410d00","#831901","#983e00","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]

yB = np.reshape(yB_1D,(Nx,Ny,Nz))
# yB[(dist[:,:,:]<0)] = float('nan')
xB = np.reshape(xB_1D,(Nx,Ny,Nz))
# xB[(dist[:,:,:]<0)] = float('nan')
# height = 350
z = z_uvp
z_on_h = z/(39/zi)
levels=[-100,-4,-1,1,4,100]
colors=['blue','green','white','yellow','red']

###########################################################################
## yslices
###########################################################################

yslice = 75

fig, axs = plt.subplots(nrows=1,ncols=1)

# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi,np.transpose(yB[:,yslice,:]),cmap=cmap,levels=[0,0.33,0.36,0.38,0.9],alpha=0.5)
# p = axs.contour(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(((terms_bdg['prod']) - (terms_bdg['totdis']))[:,yslice,:]),levels=[-100,-5,-1,1,5,100])
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[:,yslice,:]),levels=[0,0.3,0.32,0.34,0.36,0.38,0.4,0.9],colors=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black'])#vmin = 0, vmax = np.sqrt(3)/2)
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(xB[:,yslice,:]),cmap=cmap, levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# axs.clabel(p, p.levels, inline=True, fontsize=10)
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi),yB[:,yslice,:].T,levels=levels,colors=colors)
sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,np.transpose(yB[:,yslice,:]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

cbar = plt.colorbar(sc,label='yB')
    
axs.set_ylabel(r'$z/h$',fontsize=15)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_title(f'$y_B(y = {yslice*dy*zi}m)$')

plt.tight_layout()

###########################################################################
## xslices
###########################################################################

# xslice = 200

# fig, axs = plt.subplots(nrows=1,ncols=1)

# sc = axs.contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),cmap=cmap,levels=[0,0.33,0.36,0.38,0.9],alpha=0.5)
# # p = axs.contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(((terms_bdg['prod']) - (terms_bdg['totdis']))[xslice,:,:]),levels=[-100,-4,-1,1,4,100],colors=['blue','green','white','yellow','red'])
# # sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[:,yslice,:]),levels=[0,0.3,0.32,0.34,0.36,0.38,0.4,0.9],colors=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black'])#vmin = 0, vmax = np.sqrt(3)/2)
# # sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(xB[:,yslice,:]),cmap=cmap, levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
# # axs.clabel(p, p.levels, inline=True, fontsize=10)
# #sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,np.transpose(yB[:,:,height]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

# cbar = plt.colorbar(sc,label='yB')
    
# axs.set_ylabel(r'$z/h$',fontsize=15)
# axs.set_xlabel(r'$y/z_i$',fontsize=15)
# axs.set_title(f'$y_B(x = {xslice*dx*zi}m)$')

# plt.tight_layout()

# plt.savefig(pathTurb + 'xB.png',dpi=300,facecolor='white', edgecolor='white')
# plt.savefig(pathOUT + 'Figures/' + 'yB_y' + str(yslice) +'_cluster.png',dpi=300,facecolor='white', edgecolor='white')

#%% Pcolor plots of TKE terms with contours of yB - yslices

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

terms = [(terms_bdg['prod']) + (terms_bdg['totdis']),terms_bdg['uturb_h'] + terms_bdg['uturb_v'], terms_bdg['pturb_h'] + terms_bdg['pturb_v'],\
         terms_bdg['adv_v'] + terms_bdg['adv_h']]
    
levels=[-100,-4,-1,1,4,100]
colors=['blue','green','white','yellow','red']

###########################################################################
## yslices
###########################################################################

yslice = 75

fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

for i in range(0,len(axs)):
    for j in range(0,len(axs[0])):
        if i == 0:
            n = j
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            # tmp_TKE_term[(dist<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*dx,z_uvp*zi,tmp_TKE_term[:,yslice,:].T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
            sc = axs[i,j].contour(np.arange(0,Nx)*dx,z_uvp*zi,np.transpose(yB[:,yslice,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
            # axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
            # axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi),color='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
        else:
            n = j+2
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            # tmp_TKE_term[(dist<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*dx,z_uvp*zi,tmp_TKE_term[:,yslice,:].T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
            sc = axs[i,j].contour(np.arange(0,Nx)*dx,z_uvp*zi,np.transpose(yB[:,yslice,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
            # axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
            # axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi),color='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
    
axs[0,0].set_ylabel(r'$z/h$',fontsize=15),axs[1,0].set_ylabel(r'$z/h$',fontsize=15)
axs[1,0].set_xlabel(r'$x/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$x/z_i$',fontsize=15)
fig.suptitle(f'$y = {yslice*dy*zi}m$')

###########################################################################
## xslices
###########################################################################

# xslice = 110

# fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

# for i in range(0,len(axs)):
#     for j in range(0,len(axs[0])):
#         if i == 0:
#             n = j
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
#             tmp_TKE_term[(dist<0)] = float('nan')
#             p2 = axs[i,j].contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[xslice,:,:].T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
#             sc = axs[i,j].contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
#         else:
#             n = j+2
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
#             tmp_TKE_term[(dist<0)] = float('nan')
#             p2 = axs[i,j].contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[xslice,:,:].T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
#             sc = axs[i,j].contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
    
# axs[0,0].set_ylabel(r'$z/h$',fontsize=15),axs[1,0].set_ylabel(r'$z/h$',fontsize=15)
# axs[1,0].set_xlabel(r'$y/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$y/z_i$',fontsize=15)
# fig.suptitle(f'$x = {xslice*dx*zi}m$')

#%%Computing the advection index from Chamecki 2023

import random

def ChameckiIndex(Nz_SLayer,coord,Adv,TotDis):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    AdvTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    TotDisTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    Ia = np.zeros((Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        
        loc = coord[i]
        
        AdvTwr[i,:] = Adv[loc,:Nz_SLayer]
        TotDisTwr[i,:] = TotDis[loc,:Nz_SLayer]
        
    Ia = np.mean(np.abs(AdvTwr)/TotDisTwr,axis=0)
    
    return Ia


N_twrs = 50

coord = []
for i in range(0,N_twrs):
    # coord.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
    coord.append((random.randrange(0,Nx,1)))
    
    
terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
         terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
         terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

Adv = np.mean(terms[5],axis=(1))
TotDis = np.mean(terms[1],axis=(1))

z = z_uvp
z_on_h = z/(39/zi)

Ia = ChameckiIndex(Nz_SLayer, coord, Adv, TotDis)

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
axs.plot(Ia[0:30],z_on_h[0:30])
axs.set_ylim(z_on_h[0],z_on_h[30])
axs.set_xlabel('Ia')
axs.set_ylabel('z/h')