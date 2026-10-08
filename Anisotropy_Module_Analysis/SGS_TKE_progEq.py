#%%Import libraries
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 5120
ly = 5120
lz = 2048

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1024
zbl = 1099
uscale = 1
wstar = 2.05
Tscale = 300
qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load surface checkpoint/istantaneous fields 
# sim = 'yb_test_v2'

sim_off = 'test_tkeprog_off_256'
sim_on_Smag = 'test_tkeprog_on_SmagSGS_256'
sim_on_TKE = 'test_tkeprog_on_tkeSGS_256'

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

step = 100000
# avgtime = 15   

mom_off = read_checkpoint_mom(path+sim_off+'/output_checkpoint/',[step],nx,ny,nz, True)
mom_on_Smag = read_checkpoint_mom(path+sim_on_Smag+'/output_checkpoint/',[step],nx,ny,nz, True)
mom_on_TKE = read_checkpoint_mom(path+sim_on_TKE+'/output_checkpoint/',[step],nx,ny,nz, False)
aniso_off = read_checkpoint_aniso(path+sim_off+'/output_checkpoint/',[step],nx,ny,nz)
aniso_on_Smag = read_checkpoint_aniso(path+sim_on_Smag+'/output_checkpoint/',[step],nx,ny,nz)
aniso_on_TKE = read_checkpoint_aniso(path+sim_on_TKE+'/output_checkpoint/',[step],nx,ny,nz)
tke_on_Smag = read_checkpoint_tke(path+sim_on_Smag+'/output_checkpoint/',[step],nx,ny,nz)
tke_on_TKE = read_checkpoint_tke(path+sim_on_TKE+'/output_checkpoint/',[step],nx,ny,nz)

# checkpnt_sfc = read_checkpoint_sfc_L(path+sim+'/output_checkpoint/',[step],nx,ny)
# RAV_aniso = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
# RAV_mom = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(avgtime)+'min.nc').data

#%%Plot profiles on velocity

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,6),sharey=True)

axs[0].plot(np.mean(mom_off['u'],axis=(0,1))[:-1],z_uvp,c='blue',label='Off')
axs[0].plot(np.mean(mom_on_Smag['u'],axis=(0,1))[:-1],z_uvp,c='r',label='On_Smag')
axs[0].plot(np.mean(mom_on_TKE['u'],axis=(0,1))[:-1],z_uvp,c='g',label='On_TKE')

axs[1].plot(np.mean(mom_off['v'],axis=(0,1))[:-1],z_uvp,c='blue')
axs[1].plot(np.mean(mom_on_Smag['v'],axis=(0,1))[:-1],z_uvp,c='r')
axs[1].plot(np.mean(mom_on_TKE['v'],axis=(0,1))[:-1],z_uvp,c='g')

axs[2].plot(np.mean(mom_off['w'],axis=(0,1))[:-1],z_w,c='blue')
axs[2].plot(np.mean(mom_on_Smag['w'],axis=(0,1))[:-1],z_w,c='r')
axs[2].plot(np.mean(mom_on_TKE['w'],axis=(0,1))[:-1],z_w,c='g')

axs[0].set_ylabel('z')
axs[0].set_ylim(0,z_uvp[-1])
axs[0].legend()
axs[0].set_xlabel('u')
axs[1].set_xlabel('v')
axs[2].set_xlabel('w')

plt.show()

#%%Plot pcolor of velocity

fig,axs = plt.subplots(3,3,tight_layout=True,figsize=(10,10),sharex=True,sharey=True)

axs[0,0].pcolormesh(x,y,mom_off['u'][:,:,10].T,cmap='viridis',vmin=10,vmax=20)
axs[0,1].pcolormesh(x,y,mom_on_Smag['u'][:,:,10].T,cmap='viridis',vmin=10,vmax=20)
axs[0,2].pcolormesh(x,y,mom_on_TKE['u'][:,:,10].T,cmap='viridis',vmin=10,vmax=20)

axs[1,0].pcolormesh(x,y,mom_off['v'][:,:,10].T,cmap='viridis',vmin=-1,vmax=1)
axs[1,1].pcolormesh(x,y,mom_on_Smag['v'][:,:,10].T,cmap='viridis',vmin=-1,vmax=1)
axs[1,2].pcolormesh(x,y,mom_on_TKE['v'][:,:,10].T,cmap='viridis',vmin=-1,vmax=1)

axs[2,0].pcolormesh(x,y,mom_off['w'][:,:,10].T,cmap='viridis')
axs[2,1].pcolormesh(x,y,mom_on_Smag['w'][:,:,10].T,cmap='viridis')
axs[2,2].pcolormesh(x,y,mom_on_TKE['w'][:,:,10].T,cmap='viridis')

plt.show()

#%%Anisotropy and SGS TKE

yB_off = aniso_off['yB']
yB_on_Smag = aniso_on_Smag['yB']
yB_on_TKE = aniso_on_TKE['yB']

e_off = aniso_off['tke_sgs']
e_on_Smag = aniso_on_Smag['tke_sgs']
e_on_TKE = aniso_on_TKE['tke_sgs']

#%%Plot vertical profiles of sgs tke

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.plot(np.mean(e_off,axis=(0,1))[:-1],z_w,c='blue')
axs.plot(np.mean(e_on_Smag,axis=(0,1))[:-1],z_w,c='r')
axs.plot(np.mean(e_on_TKE,axis=(0,1))[:-1],z_w,c='g')

plt.show()

#%%Plot pclormesh of yB

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,4),sharey=True)

# axs[0].pcolormesh(x,z_uvp,aniso_off['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x,z_uvp,aniso_on_Smag['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[2].pcolormesh(x,z_uvp,aniso_on_TKE['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs[0].pcolormesh(x,z_uvp,np.mean(aniso_off['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].pcolormesh(x,z_uvp,np.mean(aniso_on_Smag['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[2].pcolormesh(x,z_uvp,np.mean(aniso_on_TKE['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[2].contour(x,z_uvp,np.mean(aniso_on_TKE['yB'],axis=(1)).T,levels=[0.37],colors='k',linewidths=1)

axs[0].set_ylim(0,0.1)

plt.show()

#%%Plot pdfs of yB for the mean vertical slice

yB_off_flat = (yB_off[:,:,0]).flatten()
yB_on_Smag_flat = (yB_on_Smag[:,:,0]).flatten()
yB_on_TKE_flat = (yB_on_TKE[:,:,0]).flatten()

x_pdf = np.linspace(0,np.sqrt(3)/2,1000)

kde_off = gaussian_kde(yB_off_flat)
pdf_off = kde_off(x_pdf)

kde_on_Smag = gaussian_kde(yB_on_Smag_flat)
pdf_on_Smag = kde_on_Smag(x_pdf)

kde_on_TKE = gaussian_kde(yB_on_TKE_flat)
pdf_on_TKE = kde_on_TKE(x_pdf)

fig,axs=plt.subplots(1,1,tight_layout=True)

axs.plot(x_pdf,pdf_off,c='b')
axs.plot(x_pdf,pdf_on_Smag,c='g')
axs.plot(x_pdf,pdf_on_TKE,c='r')

plt.show()

#%%

def centered_difference_z(field, dz, axis=2):
    """Compute d(field)/dz with centered differences on interior levels."""
    field = np.asarray(field)
    gradient = np.empty_like(field, dtype=np.float64)

    interior = [slice(None)] * field.ndim
    lower = [slice(None)] * field.ndim
    upper = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    lower[axis] = slice(0, -2)
    upper[axis] = slice(2, None)
    gradient[tuple(interior)] = (field[tuple(upper)] - field[tuple(lower)])/(2*dz)

    first = [slice(None)] * field.ndim
    second = [slice(None)] * field.ndim
    first[axis] = 0
    second[axis] = 1
    gradient[tuple(first)] = (field[tuple(second)] - field[tuple(first)])/dz

    last = [slice(None)] * field.ndim
    next_to_last = [slice(None)] * field.ndim
    last[axis] = -1
    next_to_last[axis] = -2
    gradient[tuple(last)] = (field[tuple(last)] - field[tuple(next_to_last)])/dz

    return gradient

#%%Compute the TKE as in Patton 2011

start = 55000
step = 5500
nfiles = 14
u_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
v_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
w_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
tke_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
TT_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
gamma_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
wT_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
B_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
T_prof_ts = np.zeros((nfiles,nz),dtype=float,order='F')
path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simC_Patton2011'
delta = np.ones((nx,ny,nz))*(1.5*dx*1.5*dy*dz)**(1/3)
z3D_uvp = np.broadcast_to(z_uvp,(nx,ny,z_uvp.size))
# filename = 'checkpoint_wsc_'

for i in range(0,nfiles):
    u = (read_checkpoint(path+sim+'/output_checkpoint/',[start + i*step],nx,ny,nz, False)['u'][:,:,:nz])*uscale
    v = (read_checkpoint(path+sim+'/output_checkpoint/',[start + i*step],nx,ny,nz, False)['v'][:,:,:nz])*uscale
    w = wnode2uvpnode(read_checkpoint(path+sim+'/output_checkpoint/',[start + i*step],nx,ny,nz, False)['w'][:,:,:nz])*uscale
    tke_sgs = (read_checkpoint_tke(path+sim+'/output_checkpoint/',[start + i*step],nx,ny,nz)['k_sgs_uvp'][:,:,:nz])*(uscale**2)
    T = (read_checkpoint(path+sim+'/output_checkpoint/',[start + i*step],nx,ny,nz, False)['SC'][:,:,:nz])*Tscale
    dTdz = wnode2uvpnode(centered_difference_z(T, dz))
    N = ((9.8/T)*dTdz)
    lf = np.where(N<=0, np.minimum(delta,2.2*z3D_uvp), np.minimum(np.minimum(delta,2.2*z3D_uvp),0.76*(np.sqrt(tke_sgs)/np.sqrt(N))))
    nu_t = 0.1*lf*np.sqrt(tke_sgs)
    nu_H = (1 + 2*lf/delta)*nu_t
    B = - nu_H*dTdz
    gamma = -(2*lf*B*dTdz)/(2.02*np.sqrt(tke_sgs))
    
    u_prof_ts[i,:] = np.mean((u - np.broadcast_to(np.mean(u,axis=(0,1)), (nx, ny, np.mean(u,axis=(0,1)).size)))**2,axis=(0,1))
    v_prof_ts[i,:] = np.mean((v - np.broadcast_to(np.mean(v,axis=(0,1)), (nx, ny, np.mean(v,axis=(0,1)).size)))**2,axis=(0,1))
    w_prof_ts[i,:] = np.mean((w - np.broadcast_to(np.mean(w,axis=(0,1)), (nx, ny, np.mean(w,axis=(0,1)).size)))**2,axis=(0,1))
    TT_prof_ts[i,:] = np.mean((T - np.broadcast_to(np.mean(T,axis=(0,1)), (nx, ny, np.mean(T,axis=(0,1)).size)))**2,axis=(0,1))
    T_prof_ts[i,:] = np.mean(T,axis=(0,1))
    wT_prof_ts[i,:] = np.mean((w - np.broadcast_to(np.mean(w,axis=(0,1)), (nx, ny, np.mean(w,axis=(0,1)).size)))*\
                              (T - np.broadcast_to(np.mean(T,axis=(0,1)), (nx, ny, np.mean(T,axis=(0,1)).size))),axis=(0,1))
    B_prof_ts[i,:] = np.mean(B,axis=(0,1))
    tke_prof_ts[i,:] = np.mean(tke_sgs,axis=(0,1))
    gamma_prof_ts[i,:] = np.mean(gamma,axis=(0,1))
    
u_prof_ta = np.mean(u_prof_ts,axis=(0))
v_prof_ta = np.mean(v_prof_ts,axis=(0))
w_prof_ta = np.mean(w_prof_ts,axis=(0))
tke_prof_ta = np.mean(tke_prof_ts,axis=(0))
TT_prof_ta = np.mean(TT_prof_ts,axis=(0))
gamma_prof_ta = np.mean(gamma_prof_ts, axis=(0))
wT_prof_ta = np.mean(wT_prof_ts,axis=(0))
B_prof_ta = np.mean(B_prof_ts,axis=(0))
T_prof_ta = np.mean(T_prof_ts,axis=(0))

tke_prof = (u_prof_ta + v_prof_ta + w_prof_ta)/2 + tke_prof_ta

#%%Plot TKE profile

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot(tke_prof/(wstar**2), z_uvp/zbl, c='k')
axs[1].plot((TT_prof_ta)/((qstar/wstar)**2), z_uvp/zbl, c='k')
axs[0].set_ylim(0,1.5)
axs[0].set_xlim(0,0.5)
axs[1].set_xlim(0,100)

plt.show()

#%%Plot profiles of vertical and horizontal velocity variance

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot((w_prof_ta + 2*tke_prof_ta/3)/(wstar**2),z_uvp/zbl,c='k')
axs[1].plot((u_prof_ta + v_prof_ta + 2*tke_prof_ta/3)/(wstar**2),z_uvp/zbl,c='k')

axs[0].set_ylim(0,1.5)

plt.show()

#%%Plot profile of potential temperature and heatflux

fig,axs = plt.subplots(1,2,tight_layout=True)

axs[0].plot((T_prof_ta),z_uvp,c='k')
theta = np.select([z_uvp <= 974.0, (z_uvp > 974.0) & (z_uvp < 1074.0), z_uvp >= 1074.0], [300.0, 300.0 + (z_uvp - 974.0) * 0.08, 308.0 + (z_uvp - 1074.0) * 0.003])
axs[0].plot(theta,z_uvp,c='k',ls='--')
axs[1].plot((wT_prof_ta + B_prof_ta)/qstar,z_uvp/zbl,c='k')
axs[1].axvline(0,c='k',lw=0.5)

axs[0].set_xlim(301,310)
axs[0].set_ylim(0,1500)
axs[1].set_ylim(0,1.5)
axs[1].set_xlim(-0.2,1)

plt.show()


#%%

#analyze RAV data

#%%Load RAV data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simC_Patton2011'

step = 128000

aniso3D = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
mom3D = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(step)+'.nc').data
sc3D = xr.open_dataarray(path+sim+'/data/Scalar3D/Data_Scalar_'+str(step)+'.nc').data
tke3D = xr.open_dataarray(path+sim+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(step)+'.nc').data
mom2D = xr.open_dataarray(path+sim+'/data/Momentum2D/Data_Momentum_2D_'+str(step)+'.nc').data
sc2D = xr.open_dataarray(path+sim+'/data/Scalar2D/Data_Scalar_2D_'+str(step)+'.nc').data

#Variable = ['u','v','w','p','u2','v2','w2','u3','v3','w3','u4','v4','w4',\
#            'uv','uw','vw','txx','tyy','tzz','txy','txz','tyz',\
#            'dudz','dvdz','nut','cs','tke_sgs']

# vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
#                         'avgUT','avgVT','avgWT','avgUTT','avgVTT','avgWTT','avgDTDX','avgDTDY','avgDTDZ',\
#                         'avgDTDX2','avgDTDY2','avgDTDZ2','avgTUT_sgs','avgTVT_sgs','avgTWT_sgs',\
#                         'avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
#                         'avg_ds']}) 

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
#                 dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
#                 'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesE),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgE_w','avgE_uvp',\
#                     'avgRHS','avgProd','avgDiss','avgNut','avgNuh','avgKTX','avgKTY',\
#                     'avgKTZ']})

#%%Plot surface friction velocity and wstar (Deardorff vertical velocity scale)

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(x,y,(mom2D[:,:,0]*uscale).T,cmap='viridis',vmin=0)
p2 = axs[1].pcolormesh(x,y,(sc2D[:,:,0]*uscale).T,cmap='viridis')

axs[0].set_title(r'$u_* [ms^{-1}]$',fontsize=14)
axs[1].set_title(r'$w_* [ms^{-1}]$',fontsize=14)
axs[0].set_xlabel(r'$x [m]$', fontsize=12)
axs[0].set_ylabel(r'$y [m]$', fontsize=12)
axs[1].set_xlabel(r'$x [m]$', fontsize=12)
cbar1 = plt.colorbar(p1)
cbar2 = plt.colorbar(p2)

plt.show()

#%%Plot temperature profile and heatflux

T = sc3D[:,:,:,0]
T_w = uvpnode2wnode(T)
T_w[:,:,0] = sc2D[:,:,-2]
# wT_sgs = - sc3D[:,:,:,-2]*sc3D[:,:,:,10]
# wT_sgs[:,:,0] = sc2D[:,:,-1]
wT = sc3D[:,:,:,4] - mom3D[:,:,:,2]*T_w #- sc3D[:,:,:,19]
wT_sgs = - sc3D[:,:,:,19]
wT_disp = (mom3D[:,:,:,2] - np.broadcast_to(np.mean(mom3D[:,:,:,2],axis=(0,1)), (nx,ny,nz))) * (T_w - np.broadcast_to(np.mean(T_w,axis=(0,1)), (nx,ny,nz)))

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot(np.mean(T,axis=(0,1))*Tscale,z_uvp/zbl,c='k')
theta = np.select([z_uvp <= 974.0, (z_uvp > 974.0) & (z_uvp < 1074.0), z_uvp >= 1074.0], [300.0, 300.0 + (z_uvp - 974.0) * 0.08, 308.0 + (z_uvp - 1074.0) * 0.003])
axs[0].plot(theta,z_uvp/zbl,c='k',ls='--')
axs[1].plot(np.mean(wT,axis=(0,1))*Tscale*uscale/qstar,z_uvp/zbl,c='orange')
axs[1].plot(np.mean(wT_sgs,axis=(0,1))*Tscale*uscale/qstar,z_uvp/zbl,c='r')
axs[1].plot(np.mean(wT_disp,axis=(0,1))*Tscale*uscale/qstar,z_uvp/zbl,c='g')
axs[1].plot(np.mean(wT + wT_sgs + wT_disp,axis=(0,1))*Tscale*uscale/qstar,z_uvp/zbl,c='k')
axs[1].axvline(0,c='k',ls='--')

axs[0].set_ylim(0,1.5)
axs[0].set_xlabel(r'$\theta [K]$', fontsize=12)
axs[0].set_ylabel(r'$z/z_i$', fontsize=12)
axs[1].set_xlabel(r"$\overline{w'T'}/Q_*$", fontsize=12)

plt.show()

#%%Plot TKE and temperature variance profile

uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0]
vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1]
ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
TT = sc3D[:,:,:,1] - sc3D[:,:,:,0]*sc3D[:,:,:,0]
TT_disp = (sc3D[:,:,:,0] - np.broadcast_to(np.mean(sc3D[:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2
uu_disp = (mom3D[:,:,:,0] - np.broadcast_to(np.mean(mom3D[:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2
vv_disp = (mom3D[:,:,:,1] - np.broadcast_to(np.mean(mom3D[:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2
ww_disp = wnode2uvpnode((mom3D[:,:,:,2] - np.broadcast_to(np.mean(mom3D[:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2)
N = ((9.81/(uscale**2/zi))/sc3D[:,:,:,0])*wnode2uvpnode(sc3D[:,:,:,10])
delta = ((1.5*dx*1.5*dy*dz)**(1/3))/zi
z3D_uvp = np.broadcast_to(z_uvp, (nx,ny,nz))
lf = np.where(N<=0, np.minimum(delta,2.2*z3D_uvp), np.minimum(np.minimum(delta,2.2*z3D_uvp),0.76*(np.sqrt(tke3D[:,:,:,1])/np.sqrt(N))))
gamma = -(2*sc3D[:,:,:,10]*(-sc3D[:,:,:,19])*lf)/(2.02*np.sqrt(tke3D[:,:,:,1]))

tke_res = 0.5*(uu + vv + ww)
tke_sgs = tke3D[:,:,:,1]
tke_disp = 0.5*(uu_disp + vv_disp + ww_disp)

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot(np.mean(tke_res,axis=(0,1))*(uscale**2)/(wstar**2),z_uvp/zbl,c='orange')
axs[0].plot(np.mean(tke_sgs,axis=(0,1))*(uscale**2)/(wstar**2),z_uvp/zbl,c='r')
axs[0].plot(np.mean(tke_disp,axis=(0,1))*(uscale**2)/(wstar**2),z_uvp/zbl,c='g')
axs[0].plot(np.mean(tke_disp + tke_sgs + tke_res,axis=(0,1))*(uscale**2)/(wstar**2),z_uvp/zbl,c='k')
axs[1].plot(np.mean(TT,axis=(0,1))*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='orange')
axs[1].plot(np.mean(TT_disp,axis=(0,1))*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='g')
axs[1].plot(np.mean(gamma,axis=(0,1))*(Tscale**2)/(qstar/wstar),z_uvp/zbl,c='r')
axs[1].plot(np.mean((TT_disp + TT + gamma),axis=(0,1))*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='k')

axs[0].set_ylim(0,1.5)
axs[0].set_xlabel(r'$\overline{E}/w_*^2$', fontsize=12)
axs[0].set_ylabel(r'$z/z_i$', fontsize=12)
axs[1].set_xlabel(r"$\overline{\theta'^2}/\theta_*$",fontsize=12)

plt.show()

#%%Plot profiles of vertical and horizontal velocity variance

ww_res = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
ww_sgs = - mom3D[:,:,:,18] + (2/3)*tke3D[:,:,:,1]
ww_disp = wnode2uvpnode((mom3D[:,:,:,2] - np.broadcast_to(np.mean(mom3D[:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2)
uh_res = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0] + mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1] 
uh_sgs = - mom3D[:,:,:,16] - mom3D[:,:,:,17] + (4/3)*tke3D[:,:,:,1]
uh_disp = (mom3D[:,:,:,0] - np.broadcast_to(np.mean(mom3D[:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2 + (mom3D[:,:,:,1] - np.broadcast_to(np.mean(mom3D[:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot(np.mean(ww_res,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='orange')
axs[0].plot(np.mean(ww_sgs,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='r')
axs[0].plot(np.mean(ww_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='g')
axs[0].plot(np.mean(ww_res + ww_sgs + ww_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='k')
axs[1].plot(np.mean(uh_res,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='orange')
axs[1].plot(np.mean(uh_sgs,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='r')
axs[1].plot(np.mean(uh_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='g')
axs[1].plot(np.mean(uh_res + uh_sgs + uh_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='k')

axs[0].set_ylim(0,1.1)
axs[0].set_ylabel(r'$z/z_i$',fontsize=12)
axs[0].set_xlabel(r"$\overline{w'^2}/w_*^2$", fontsize=12)
axs[1].set_xlabel(r"$\overline{U'^2}/w_*^2$", fontsize=12)

plt.show()

#%%

nuH_sc = sc3D[:,:,:,-2]*(uscale*zi)
nuH_tke = tke3D[:,:,:,6]*(uscale*zi)

tmp = abs(nuH_sc - nuH_tke)

#%%Plot anisotropy verical slice

fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.pcolormesh(x,z_uvp,aniso3D[:,ny//2,:,1].T,cmap=cmap, vmin = 0, vmax = np.sqrt(3)/2)
axs.pcolormesh(x,z_uvp,np.mean(aniso3D[:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

plt.show()

































