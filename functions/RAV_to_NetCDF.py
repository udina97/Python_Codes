"""
Program Name: RAV_to_NetCDF.py
Program purpose: This program reads data from the LES RAV binari files, and 
                compacts it the NetCDF format structure, with all variables in a single structure.

Program Author: Marc Calaf.

Date created: 12 July 2022
Last date modified: 6 April 2023

"""

#%%

#Libraries and Functions
import numpy as np
import os
import xarray as xr
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from loadData import load_momentum
from loadData import load_scalar
from loadData import load_scalar_2D
from loadData import load_momentum_2D
from loadData import load_2Dfield
from loadData import load_tkevar
from loadData import load_tkevarGigi
from loadData import load_anisotropy

#%% Defining the main variables and loading/averaging the RAV_output data from the LES

# Variables that define characteristics of the data saved in the RAV_output files:

Nx = 64
Ny = 64
Nz = 64
# Lx = 2*math.pi
# Ly = 2*math.pi
# Lz = 1
# dx = Lx/Nx
# dy = Ly/Ny
# dz = Lz/Nz

NumVariables = 28 #29 if only xb and yb #31 if phim and phih 3D #27 if none of the previous variables are there #28 if L3D is computed
NumVariablesAniso = 2 #9
NumVariablesTKE = 47
NumVariablesSC = 10 #10 if not studying the temperature variance #22 if studying T variance
NumVariables_2D_SC = 8
NumVariables_2D = 1
NumVariablesTKEGigi = 64

NumIt = 3000 #Giulia's Data
# NumFiles = 3
# NumRAV = 1
# StartTime = 1488000

anisotropy_flag = True

#Define the path where to RAV data files are located:

# sim = 'RandTurbStats/Aniso_1ms_128'
# path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_RAV/'
# pathTKE = '/scratch/general/nfs1/u1450851/LES_Sims/MarcTempVar/output_tke_3D/'
# pathTKEGigi = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/no_forest_9mps/tke_output/'

#%% Loading/averaging the anisotropy 3D RAV_output data from the LES

NumFile = [1,2,4,6]
StartTimes = [78000,75000,69000,63000]
name_ext = ['5','10','20','30']
# name = ['K','L','M','N','Q','R','S','T']
# name = ['A','B','C','D','E','F','G','H','I','J']
# name = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
name = ['test_lag_1ms']
for j in range(len(name)):
    # sim = 'RandTurbStats/256/Aniso_1ms_' + name[j]
    path = '/scratch/general/nfs1/u1450851/LES_Sims/'+name[j]+'/output_RAV/'
    for i in range(0,len(NumFile)):
    # for i in range(0,NumRAV):
        
        # StartTime = 63000 + NumIt*i
        StartTime = StartTimes[i]
        NumFiles = NumFile[i]
        
        #############################################################################################################################################
    
        # print('Calling function: load_anisotropy')
        
        # ##-For the anisotropy outputs
        # dataA = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesAniso),order='F'),\
        #                         dims=('x','y','z','variable'), coords = {'variable':['avgXB','avgYB']})#,'avgPHIM','avgPHIH','avgPSIM',\
        #                                                                               #'avgPSIH','avgL3D','avgustar3D','avgSCF3D']}) 
            
        # vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesAniso),order='F'),\
        #                     dims=('x', 'y','z','variable'), coords = {'variable':['avgXB','avgYB']})#,'avgPHIM','avgPHIH','avgPSIM',\
        #                                                                           #'avgPSIH','avgL3D','avgustar3D','avgSCF3D']})
        
        # dataA = load_anisotropy(Nx,Ny,Nz,NumVariablesAniso,NumIt,NumFiles,StartTime,path,vardata,anisotropy_flag)
        
        ############################################################################################################################################
        
        # % Loading/averaging the momentum 3D RAV_output data from the LES
        
        print('Calling function: load_momentum')
        
        # data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
        #                         dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
        #                         'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
        #                         'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
        #                         'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
        #                         'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']}) 
            
        # vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
        #                     dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
        #                     'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
        #                     'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
        #                     'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
        #                     'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']})
        
        ###-For the anisotropy outputs
        data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
                                dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
                                'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
                                'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                                'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
                                'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']}) 
            
        vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
                            dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
                            'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
                            'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                            'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
                            'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']})
        
        data = load_momentum(Nx,Ny,Nz,NumVariables,NumIt,NumFiles,StartTime,path,vardata,anisotropy_flag)
        
        ################################################################################################################################################
        
        #% Loading/averaging the tke budget variables
        
        # print('Calling function: load_tkevar')
        
        # dataTKE = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesTKE),order='F'),\
        #                     dims=('x', 'y','z','variable'), coords = {'variable':['avgDUDX','avgDUDY',\
        #                     'avgDUDZ','avgDVDX','avgDVDY','avgDVDZ','avgDWDX','avgDWDY','avgDWDZ',\
        #                     'avgUV2','avgUW2','avgVU2','avgVW2','avgWU2','avgWV2','avgUTXX',\
        #                     'avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ','avgWTXX',\
        #                     'avgWTYY','avgWTZZ','avgPU','avgPV','avgPW','avgPclean',\
        #                     'avgdxx','avgdyy','avgdzz','avgdxy','avgdxz','avgdyz',\
        #                     'avgUTXY','avgUTXZ','avgVTXY','avgVTYZ','avgWTXZ','avgWTYZ',\
        #                     'avgPU2','avgPV2','avgPW2','avgPU3','avgPV3','avgPW3','avgPclean2']})
        
        # vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesTKE),order='F'),\
        #                     dims=('x', 'y','z','variable'), coords = {'variable':['avgDUDX','avgDUDY',\
        #                     'avgDUDZ','avgDVDX','avgDVDY','avgDVDZ','avgDWDX','avgDWDY','avgDWDZ',\
        #                     'avgUV2','avgUW2','avgVU2','avgVW2','avgWU2','avgWV2','avgUTXX',\
        #                     'avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ','avgWTXX',\
        #                     'avgWTYY','avgWTZZ','avgPU1','avgPV1','avgPW1','avgPclean1',\
        #                     'avgdxx','avgdyy','avgdzz','avgdxy','avgdxz','avgdyz',\
        #                     'avgUTXY','avgUTXZ','avgVTXY','avgVTYZ','avgWTXZ','avgWTYZ',\
        #                     'avgPU2','avgPV2','avgPW2','avgPU3','avgPV3','avgPW3','avgPclean2']})
            
        # dataTKE = load_tkevar(Nx,Ny,Nz,NumVariablesTKE,NumIt,NumFiles,StartTime,pathTKE,vardata)
        
        ##############################################################################################################################################
        
        #% Loading/averaging the tke budget variables from Gigi output
        
        # print('Calling function: load_tkevar')
            
        # dataTKEGigi = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesTKEGigi),order='F'),\
        #                         dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
        #                         'avgW','avgP','avgUU','avgVV','avgWW','avgUV','avgUW',\
        #                         'avgVW','avgDUDX','avgDUDY','avgDUDZ','avgDVDX','avgDVDY','avgDVDZ',\
        #                         'avgDWDX','avgDWDY','avgDWDZ','avgTXX','avgTYY','avgTZZ',\
        #                         'avgTXY','avgTXZ','avgTYZ','avgUUU','avgUVV','avgUWW',\
        #                         'avgVUU','avgVVV','avgVWW','avgWUU','avgWVV','avgWWW',\
        #                         'avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ',\
        #                         'avgWTXX','avgWTYY','avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ',\
        #                         'avgUTXZ','avgVTYZ','avgUP','avgVP','avgWP','avgDXX','avgDYY','avgDZZ',\
        #                         'avgDXY','avgDXZ','avgDYZ','avgFDX','avgFDY','avgFDZ','avgUFDX','avgVFDY','avgWFDZ']})
        
        # vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesTKEGigi),order='F'),\
        #                     dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
        #                     'avgW','avgP','avgUU','avgVV','avgWW','avgUV','avgUW',\
        #                     'avgVW','avgDUDX','avgDUDY','avgDUDZ','avgDVDX','avgDVDY','avgDVDZ',\
        #                     'avgDWDX','avgDWDY','avgDWDZ','avgTXX','avgTYY','avgTZZ',\
        #                     'avgTXY','avgTXZ','avgTYZ','avgUUU','avgUVV','avgUWW',\
        #                     'avgVUU','avgVVV','avgVWW','avgWUU','avgWVV','avgWWW',\
        #                     'avgUTXX','avgUTYY','avgUTZZ','avgVTXX','avgVTYY','avgVTZZ',\
        #                     'avgWTXX','avgWTYY','avgWTZZ','avgVTXY','avgWTXZ','avgUTXY','avgWTYZ',\
        #                     'avgUTXZ','avgVTYZ','avgUP','avgVP','avgWP','avgDXX','avgDYY','avgDZZ',\
        #                     'avgDXY','avgDXZ','avgDYZ','avgFDX','avgFDY','avgFDZ','avgUFDX','avgVFDY','avgWFDZ']})
        
        # dataTKEGigi = load_tkevarGigi(Nx,Ny,Nz,NumVariablesTKEGigi,NumIt,NumFiles,StartTime,pathTKEGigi,vardata)
        
        ################################################################################################################################################
        
        #% Loading/averaging the scalar 3D RAV_output data from the LES
        
        print('Calling function: load_scalar')
        
        # StartTime = 2572000
        
        #For normal analysis:
        dataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
                                dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
                                'avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
                                'avg_ds']})
        
        vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
                            'avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
                            'avg_ds']})
        
        #For temperature variance analysis:
        # dataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
        #                         dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
        #                         'avgUT','avgVT','avgWT','avgUTT','avgVTT','avgWTT','avgDTDX','avgDTDY','avgDTDZ',\
        #                         'avgDTDX2','avgDTDY2','avgDTDZ2','avgTUT_sgs','avgTVT_sgs','avgTWT_sgs',\
        #                         'avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
        #                         'avg_ds']}) 
        
        # vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
        #                         dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
        #                         'avgUT','avgVT','avgWT','avgUTT','avgVTT','avgWTT','avgDTDX','avgDTDY','avgDTDZ',\
        #                         'avgDTDX2','avgDTDY2','avgDTDZ2','avgTUT_sgs','avgTVT_sgs','avgTWT_sgs',\
        #                         'avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
        #                         'avg_ds']}) 
        
        dataS = load_scalar(Nx,Ny,Nz,NumVariablesSC,NumIt,NumFiles,StartTime,path,vardataS)
        
        ###########################################################################################################################################
        
        #% Loading/averaging the momentum 2D RAV_output data from the LES
        
        print('Calling function: load_momentum_2D')
        # StartTime = 2556000
        
        data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
                        dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})
            
        vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
                        dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})
        
        data_2D = load_momentum_2D(Nx,Ny,NumVariables_2D,NumIt,NumFiles,StartTime,path,vardata)
        
        #############################################################################################################################################
        
        #% Loading/averaging the scalar 2D RAV_output data from the LES
        
        print('Calling function: load_scalar_2D')
        
        
        dataS_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
                        dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
                        'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})
            
        vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
                        dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
                        'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})
        
        dataS_2D = load_scalar_2D(Nx,Ny,NumVariables_2D_SC,NumIt,NumFiles,StartTime,path,vardata)
        
        ###########################################################################################################################################
        
        #% Loading input surface temperature field
        
        # data_sfcT = np.zeros((Nx,Ny),'d')
        
        # data_sfcT = load_2Dfield(Nx,Ny,path)
        
        # data_sfcT = xr.DataArray(data_sfcT)
        
        ###########################################################################################################################################
        
        #% Saving the loaded Data into NumPy files
        
        #Define the path where to save the NetCDF data:
    
        # pathOUT_A = path + '../data/Anisotropy/'
        # if os.path.exists(pathOUT_A):
        #     print(f"The forlder '{pathOUT_A}' aready exists.")
        # else:
        #     os.makedirs(pathOUT_A)
        #     print(f"Created folder '{pathOUT_A}'.")
        # os.chdir(pathOUT_A)
        # dataA.to_netcdf('Data_Anisotropy_'+str(StartTime)+'.nc')
        
        pathOUT_M3D = path + '../data/Momentum3D/'
        if os.path.exists(pathOUT_M3D):
            print(f"The forlder '{pathOUT_M3D}' aready exists.")
        else:
            os.makedirs(pathOUT_M3D)
            print(f"Created folder '{pathOUT_M3D}'.")
        os.chdir(pathOUT_M3D)
        # data.to_netcdf('Data_Momentum_'+str(StartTime)+'.nc')
        data.to_netcdf('Data_Momentum_'+name_ext[i]+'min.nc')
        
        pathOUT_M2D = path + '../data/Momentum2D/'
        if os.path.exists(pathOUT_M2D):
            print(f"The forlder '{pathOUT_M2D}' aready exists.")
        else:
            os.makedirs(pathOUT_M2D)
            print(f"Created folder '{pathOUT_M2D}'.")
        os.chdir(pathOUT_M2D)
        # data_2D.to_netcdf('Data_Momentum_2D_'+str(StartTime)+'.nc')
        data_2D.to_netcdf('Data_Momentum_2D_'+name_ext[i]+'min.nc')
        
        pathOUT_S3D = path + '../data/Scalar3D/'
        if os.path.exists(pathOUT_S3D):
            print(f"The forlder '{pathOUT_S3D}' aready exists.")
        else:
            os.makedirs(pathOUT_S3D)
            print(f"Created folder '{pathOUT_S3D}'.")
        os.chdir(pathOUT_S3D)
        # dataS.to_netcdf('Data_Scalar_'+str(StartTime)+'.nc')
        dataS.to_netcdf('Data_Scalar_'+name_ext[i]+'min.nc')
        
        pathOUT_S2D = path + '../data/Scalar2D/'
        if os.path.exists(pathOUT_S2D):
            print(f"The forlder '{pathOUT_S2D}' aready exists.")
        else:
            os.makedirs(pathOUT_S2D)
            print(f"Created folder '{pathOUT_S2D}'.")
        os.chdir(pathOUT_S2D)
        # dataS_2D.to_netcdf('Data_Scalar_2D_'+str(StartTime)+'.nc')
        dataS_2D.to_netcdf('Data_Scalar_2D_'+name_ext[i]+'min.nc')

# data.to_netcdf('Data_Momentum_'+str(StartTime)+'.nc')  #This saves the data as NetCDF files, that keep the xarray form.
# # dataS.to_netcdf('Data_Scalar_'+str(StartTime)+'.nc')
# dataS_2D.to_netcdf('Data_Scalar_2D_'+str(StartTime)+'.nc')
# dataA.to_netcdf('Data_Anisotropy_'+str(StartTime)+'.nc')
# data_2D.to_netcdf('Data_Momentum_2D_'+str(StartTime)+'.nc')
# dataTKE.to_netcdf('Data_TKEbudget3D_3hr.nc')
# dataTKEGigi.to_netcdf('Data_GigiTKE_5hr.nc')
# data_sfcT.to_netcdf('Surface_T_in.nc')

# np.save('Data_Momentum.npy',data) #This saves the data as numpy file. It does not maintain the Xarray form.
#np.save('Data_Scalar.npy',dataS)

# %%
