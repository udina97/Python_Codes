"""
Program Name: Topo_SubDomain.py
Program purpose: This program takes the SRTM data and extracts a subdomain of topography.
Program Author: Marc Calaf.

Date created: 20 November 2021
Last date modified: 20 November 2021

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import cm 
from numpy import save

os.chdir("/Users/mcalaf/Documents/Utah/Proposals/2021/TEAMx-2021/TEAMx-US-PreAnalysis/SRTM-TopoData/")


#----------------------------------------------------------------------------------
# Function that generates a subsample of the larger data set using
# a center point and a Domain of Analysis Size.
#---------------------------------------------------------------------------------

def Data_Subset1(lat0,lon0,height,DomainSize):
    "Calling the Data_Subset() function"

    res = dr #Equivalent to 1-arc second sampling (SRTM1). 
    #DomainSize = 10000 #in meters

    #Adjusting the domain on which we want to compute the Lacunarity function.

    tmp = np.where(np.abs(latitude-lat0) == np.min(np.abs(latitude-lat0)))
    indlat_0 = int(tmp[0][0])

    tmp = np.where(np.abs(longitude-lon0) == np.min(np.abs(longitude-lon0)))
    indlon_0 = int(tmp[0][0])

    indlat_min = int(indlat_0 - int((DomainSize/2)/res))
    indlat_max = int(indlat_0 + int((DomainSize/2)/res))
    indlon_min = int(indlon_0 - int((DomainSize/2)/res))
    indlon_max = int(indlon_0 + int((DomainSize/2)/res)) 


    Rheight = height[indlat_min:indlat_max,indlon_min:indlon_max] #Reduced subdomain with measures of terrain height

    Rlon = longitude[indlon_min:indlon_max] #Longitude coordinates for the reduced subdomain.
    Rlat = latitude[indlat_min:indlat_max]  #Latitude coordinates for the reduced subdomain
    
    return(Rlon,Rlat,Rheight,indlon_min,indlon_max,indlat_min,indlat_max)

#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#Define a reduce subdomain of topography and plot it together with respect to the Reference Point station selected.
#-----------------------------------------------------------------------------

#Definition of the Sites and corresponding coordinates:
DomainSize = 30000 #in meters

cases = ['Kolsass', 'Borghetto','MonteBaldo']
studycase = cases[2]

site = pd.DataFrame(index = ['Kolsass','Borghetto','MonteBaldo'], columns = ['Lat','Lon'],data= np.zeros((3,2)))

#Coordinate of the location used as Reference Point.
if (studycase == 'Kolsass'):
    site.loc['Kolsass'] = {'Lat': 47.300278, 'Lon': 11.630833}
    file_name = 'Topo_data_Innsbruck_SRTM1.npz'
elif (studycase == 'Borghetto'):
    site.loc['Borghetto'] = {'Lat': 45.6980, 'Lon': 10.9283}
    file_name = 'Topo_data_Adige_SRTM1.npz'
elif (studycase == 'MonteBaldo'):
    site.loc['MonteBaldo'] = {'Lat': 45.7264, 'Lon': 10.8439}
    file_name = 'Topo_data_Adige_SRTM1.npz'


npzfile = np.load(file_name)
height = npzfile['elevation_data']
latitude = npzfile['latitude']
longitude = npzfile['longitude']
dr = 30 #the resolution of the Topodata is of 30m (STRM1).

#Create a subdomain of terrain elevation:
lat0 = site.loc[studycase]['Lat']
lon0 = site.loc[studycase]['Lon']
[Rlon,Rlat,Rheight,indlon_min,indlon_max,indlat_min,indlat_max] = Data_Subset1(lat0,lon0,height,DomainSize)
Nx_new = np.shape(Rheight)[0]
Ny_new = np.shape(Rheight)[1]

#Plot the reduced domain topography.
#-----------------------------------------------------------------------------  
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_box_aspect(aspect = (3,3,1))

X, Y = np.meshgrid(Rlon, Rlat)
surf = ax.plot_surface(X, Y, Rheight, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=.8)   

Xp, Yp = np.meshgrid(lon0, lat0)
hp = Rheight[int(np.round(Nx_new/2)),int(np.round(Nx_new/2))]
ax.scatter(lon0,lat0, hp, marker='o', c="black") 

ax.set_title(f"Topography around {studycase} site")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Height")
#ax.set_xticks([100,125,150,175,200]) 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.pcolormesh(X,Y,Rheight,shading='auto')
ax1.scatter(lon0,lat0, s = 20, marker='o', c="red",) 
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")


#%% Decomposition of the Topography in FFT modes and reconstruction with just a few of them.

#-----------------------------------------------------------------------------
#Compute the FFT decomposition of the reduced topography.
#-----------------------------------------------------------------------------

from heapq import nlargest

nmax = 2 #50 #Total number of MOST energetic modes we intend to use to reconstruct the topography.

# Forward FFT
mean_topo = Rheight.mean()
print(f'The mean height of the subdomain is {mean_topo} m')
topo = Rheight - mean_topo

print('  Fourier transforming the topography...')
#kx = np.fft.fftfreq(Nx_new, d=1/Nx_new)
#ky = np.fft.fftfreq(Ny_new, d=1/Ny_new)
kx = np.fft.fftfreq(Nx_new, d=dr)
ky = np.fft.fftfreq(Ny_new, d=dr)
k = np.zeros(len(kx))

#Decompose FFT2
ftopo = np.fft.fft2(topo, norm="ortho")
spec_topo = np.abs(ftopo)**2
var_topo = np.sum(spec_topo)/2

#Recompose the topography by using just a few modes:
ftopo_acc = np.zeros(ftopo.shape, dtype=np.complex64) # We initialize with zeros the matrix of amplitude coefficents for the reconstruction.
# Only those locations in the matrix where the amplitude is different than zero represent a mode use for the reconstruction.

valmax = nlargest(2*nmax, spec_topo.flatten())  #Selects the values of the 2*nmax most energetic modes. 
                                                #We use 2*nmax because we also need to take the amplitude of the complex conjugate.

# In this loop we find the corresponding matrix indices of those most energetic amplitudes,
# and place their values on the initialized with zero matrix of the accumulated ftopo_acc.
count = 0
for inmax in range(2*nmax):
  ind = list(np.where(spec_topo==valmax[inmax]))
  #print(ind)
  if (len(ind[0])>1):
    ind[0]=np.array([ind[0][np.mod(inmax, 2)], ])
    ind[1]=np.array([ind[1][np.mod(inmax, 2)], ])
  #print(ky[ind[0]], kx[ind[-1]])
  ftopo_acc[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
  if (np.remainder(inmax, 2)==0):
      k[count] = np.sqrt(kx[ind[0]]**2 + ky[ind[1]]**2)
      count = count +1
  

#At this point we are done including amplitudes in the reconstructed ftopo_acc matrix.  

# Inversed FFT
print('  Inversed Fourier transforming the topography...')
topo_acc = np.fft.ifft2(ftopo_acc, norm="ortho")
topo_acc = topo_acc.real + mean_topo # We add the mean to the reconstructed topography, since it was initially removed.

#-----------------------------------------------------------------------------


#Plot the recomposition of the reduced topography.
#-----------------------------------------------------------------------------

#Plot the Recomposition of the topography:   
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_box_aspect(aspect = (3,3,1))

surf = ax.plot_surface(X, Y, topo_acc, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=.5)   

Xp, Yp = np.meshgrid(lon0, lat0)
hp = topo_acc[int(np.round(Nx_new/2)),int(np.round(Nx_new/2))]
ax.scatter(lon0,lat0, hp, marker='o', c="black") 

ax.set_title(f"Reduced Topography at {studycase} with N = {nmax} FFT modes")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Height")
#ax.set_xticks([100,125,150,175,200])  


#%% Storing the Topography data in a DataFrame


#First we create the structure of the Series that will hold the relevant variables for each case.
variables = { 'Topo' : np.zeros((Nx_new,Ny_new)), 'Nx_new' : 0., 'Ny_new': 0.}
Topo_variables = pd.Series(variables)

#For each site we create a Data Frame of "Sites"
frame = { 'Kolsass_Full': Topo_variables, 'Kolsass_50': Topo_variables, 'Kolsass_20': Topo_variables, 'Kolsass_2': Topo_variables,
         'MonteBaldo_Full': Topo_variables, 'MonteBaldo_50': Topo_variables, 'MonteBaldo_20': Topo_variables, 'MonteBaldo_2': Topo_variables}
DataTopo = pd.DataFrame(frame)


DataTopo['Kolsass_Full'].Topo = Rheight; DataTopo['Kolsass_Full'].Nx_new = Nx_new; DataTopo['Kolsass_Full'].Ny_new = Ny_new
DataTopo['Kolsass_50'].Topo = topo_acc; DataTopo['Kolsass_50'].Nx_new = Nx_new; DataTopo['Kolsass_50'].Ny_new = Ny_new
DataTopo['Kolsass_20'].Topo = topo_acc; DataTopo['Kolsass_20'].Nx_new = Nx_new; DataTopo['Kolsass_20'].Ny_new = Ny_new
DataTopo['Kolsass_2'].Topo = topo_acc; DataTopo['Kolsass_2'].Nx_new = Nx_new; DataTopo['Kolsass_2'].Ny_new = Ny_new


DataTopo['MonteBaldo_Full'].Topo = Rheight; DataTopo['MonteBaldo_Full'].Nx_new = Nx_new; DataTopo['MonteBaldo_Full'].Ny_new = Ny_new
DataTopo['MonteBaldo_50'].Topo = topo_acc; DataTopo['MonteBaldo_50'].Nx_new = Nx_new; DataTopo['MonteBaldo_50'].Ny_new = Ny_new
DataTopo['MonteBaldo_20'].Topo = topo_acc; DataTopo['MonteBaldo_20'].Nx_new = Nx_new; DataTopo['MonteBaldo_20'].Ny_new = Ny_new
DataTopo['MonteBaldo_2'].Topo = topo_acc; DataTopo['MonteBaldo_2'].Nx_new = Nx_new; DataTopo['MonteBaldo_2'].Ny_new = Ny_new


DataTopo.to_pickle("DataTopo.pkl") #This seems to work better to keep the structure of the DataFrame
##DataTopo.to_csv('DataTopo.csv', index ='False') #Need to test this version with Index = false

#%%

#DataTopo = pd.read_pickle('DataTopo.pkl') #This seems to work better to keep the structure of the DataFrame
##DataTopo = pd.read_csv('DataTopo.csv')

#-----------------------------------------------------------------------------
#Computing the Lacunarity of the real topography and low order topography
#-----------------------------------------------------------------------------

os.chdir("/Users/mcalaf/Documents/Utah/Proposals/2021/TEAMx-2021/TEAMx-US-PreAnalysis/")
from Lacunarity import Lacunarity2D_MAT 

#Define the DataFrame structure that will contain the Lacunarity data for all cases:
#--------------------------------------------------
#First we create the structure of the Series that will hold the relevant variables for each case.
LacVariables = { 'Lac' : np.zeros(Nx_new), 'Mean' : np.zeros(Nx_new), 'Var': np.zeros(Nx_new), 'R': np.zeros(Nx_new)}
LacVar = pd.Series(LacVariables)

#For each site we create a Data Frame of "Sites"
Lacframe = { 'Kolsass_Full': LacVar, 'Kolsass_50': LacVar, 'Kolsass_20': LacVar, 'Kolsass_2': LacVar,
         'MonteBaldo_Full': LacVar, 'MonteBaldo_50': LacVar, 'MonteBaldo_20': LacVar, 'MonteBaldo_2': LacVar}
DataLac = pd.DataFrame(Lacframe)

#--------------------------------------------------

StudyCases = ['Kolsass_Full', 'Kolsass_50', 'Kolsass_20', 'Kolsass_2',
              'MonteBaldo_Full', 'MonteBaldo_50', 'MonteBaldo_20', 'MonteBaldo_2']


for i in range(0,len(StudyCases)):
    [tmp_Lac,tmp_Mean,tmp_Variance,tmp_r] = Lacunarity2D_MAT(DataTopo[StudyCases[i]].Nx_new,DataTopo[StudyCases[i]].Ny_new,DataTopo[StudyCases[i]].Topo)
    DataLac[StudyCases[i]].Lac = tmp_Lac
    DataLac[StudyCases[i]].Mean = tmp_Mean
    DataLac[StudyCases[i]].Var = tmp_Variance
    DataLac[StudyCases[i]].R = tmp_r


DataLac.to_pickle("DataLac.pkl")

#%%

#Integrated measure of cross-scale spatial heterogeneity:
# Lac_total = 1/Lac(1) * Sum_r Lac(r)
tmp = {'Kolsass_Full': 0, 'Kolsass_50': 0, 'Kolsass_20': 0, 'Kolsass_2': 0,
       'MonteBaldo_Full': 0, 'MonteBaldo_50': 0, 'MonteBaldo_20': 0, 'MonteBaldo_2': 0}
Lac_total = pd.Series(tmp)    

for i in range(0,len(StudyCases)):
    Lac_total[StudyCases[i]] = (1/DataLac[StudyCases[i]].Lac[0])*np.sum(DataLac[StudyCases[i]].Lac)

fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)
ax2.plot(Lac_total,'o')
plt.xticks(rotation=45)
ax2.set_ylabel('$ \Lambda_{total} = \,[\sum_r \Lambda (r)]\,/\,\Lambda (r_0) $')
plt.tight_layout()


#----------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot( DataLac['Kolsass_Full'].R*dr/1000, DataLac['Kolsass_Full'].Lac,'-k',label='Kolsass Real')
ax1.plot( DataLac['Kolsass_50'].R *dr/1000, DataLac['Kolsass_50'].Lac, '--k', label='Kolsass 50')
ax1.plot( DataLac['Kolsass_20'].R *dr/1000, DataLac['Kolsass_20'].Lac, '-.k', label='Kolsass 20')
ax1.plot( DataLac['Kolsass_2'].R *dr/1000, DataLac['Kolsass_2'].Lac, ':k', label='Kolsass 2')
ax1.plot( DataLac['MonteBaldo_Full'].R *dr/1000, DataLac['MonteBaldo_Full'].Lac, '-', color='silver', label='MonteBaldo Real')
ax1.plot( DataLac['MonteBaldo_50'].R *dr/1000, DataLac['MonteBaldo_50'].Lac, '--', color='silver', label='MonteBaldo 50')
ax1.plot( DataLac['MonteBaldo_20'].R *dr/1000, DataLac['MonteBaldo_20'].Lac, '-.', color='silver', label='MonteBaldo 20')
ax1.plot( DataLac['MonteBaldo_2'].R *dr/1000, DataLac['MonteBaldo_2'].Lac, ':', color='silver', label='MonteBaldo2')

ax1.set_xscale("log")
plt.legend(loc = 'upper right')
ax1.set_ylabel('$ \Lambda (r) $')
ax1.set_xlabel('$ r [km] $')

#----------------------------------

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot( DataLac['Kolsass_Full'].R*dr/1000, DataLac['Kolsass_Full'].Lac/DataLac['Kolsass_Full'].Lac[0],'-k',label='Kolsass Real')
ax1.plot( DataLac['Kolsass_50'].R *dr/1000, DataLac['Kolsass_50'].Lac/DataLac['Kolsass_50'].Lac[0], '--k', label='Kolsass 50')
ax1.plot( DataLac['Kolsass_20'].R *dr/1000, DataLac['Kolsass_20'].Lac/DataLac['Kolsass_20'].Lac[0], '-.k', label='Kolsass 20')
ax1.plot( DataLac['Kolsass_2'].R *dr/1000, DataLac['Kolsass_2'].Lac/DataLac['Kolsass_2'].Lac[0], ':k', label='Kolsass 2')
ax1.plot( DataLac['MonteBaldo_Full'].R *dr/1000, DataLac['MonteBaldo_Full'].Lac/DataLac['MonteBaldo_Full'].Lac[0], '-', color='silver', label='MonteBaldo Real')
ax1.plot( DataLac['MonteBaldo_50'].R *dr/1000, DataLac['MonteBaldo_50'].Lac/DataLac['MonteBaldo_50'].Lac[0], '--', color='silver', label='MonteBaldo 50')
ax1.plot( DataLac['MonteBaldo_20'].R *dr/1000, DataLac['MonteBaldo_20'].Lac/DataLac['MonteBaldo_20'].Lac[0], '-.', color='silver', label='MonteBaldo 20')
ax1.plot( DataLac['MonteBaldo_2'].R *dr/1000, DataLac['MonteBaldo_2'].Lac/DataLac['MonteBaldo_2'].Lac[0], ':', color='silver', label='MonteBaldo2')

ax1.set_xscale("log")
plt.legend(loc = 'lower left')
ax1.set_ylabel('$ \Lambda (r)\Lambda (r_0) $')
ax1.set_xlabel('$ r [km] $')


#----------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot( DataLac['Kolsass_Full'].R*dr/1000, DataLac['Kolsass_Full'].Lac/DataLac['Kolsass_Full'].Lac[0],'-k',label='Kolsass Real')
ax1.plot( DataLac['Kolsass_50'].R *dr/1000, DataLac['Kolsass_50'].Lac/DataLac['Kolsass_50'].Lac[0], '--k', label='Kolsass 50')
ax1.plot( DataLac['Kolsass_20'].R *dr/1000, DataLac['Kolsass_20'].Lac/DataLac['Kolsass_20'].Lac[0], '-.k', label='Kolsass 20')
ax1.plot( DataLac['Kolsass_2'].R *dr/1000, DataLac['Kolsass_2'].Lac/DataLac['Kolsass_2'].Lac[0], ':k', label='Kolsass 2')
ax1.plot( DataLac['MonteBaldo_Full'].R *dr/1000, DataLac['MonteBaldo_Full'].Lac/DataLac['MonteBaldo_Full'].Lac[0], '-', color='silver', label='MonteBaldo Real')
ax1.plot( DataLac['MonteBaldo_50'].R *dr/1000, DataLac['MonteBaldo_50'].Lac/DataLac['MonteBaldo_50'].Lac[0], '--', color='silver', label='MonteBaldo 50')
ax1.plot( DataLac['MonteBaldo_20'].R *dr/1000, DataLac['MonteBaldo_20'].Lac/DataLac['MonteBaldo_20'].Lac[0], '-.', color='silver', label='MonteBaldo 20')
ax1.plot( DataLac['MonteBaldo_2'].R *dr/1000, DataLac['MonteBaldo_2'].Lac/DataLac['MonteBaldo_2'].Lac[0], ':', color='silver', label='MonteBaldo2')

plt.legend(loc = 'lower left')
ax1.set_ylabel('$ \Lambda (r)\Lambda (r_0) $')
ax1.set_xlabel('$ r [km] $')
  
    
    