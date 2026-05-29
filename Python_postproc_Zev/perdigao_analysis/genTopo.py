#!/usr/bin/env python3

"""
Program: genTopo
Smooth the Amazon topography for the simulation domain.
"""

### Histories:
### 2019/11/24 -- Bicheng Chen (chabby@ucla.edu) -- First created.
### 2021/09/20 -- Zev Underwood (zev.underwood@gmail.com) -- adapted.


## Prerequisite Module
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d
from scipy.ndimage.interpolation import rotate
import dted as dd
from pathlib import Path



## User-specified Variable
# File
dt2_file = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_simulation_results/validation/n39_w008_1arc_v3.dt2'
fn_hgt = './data/S03W060.hgt'



#Data type
data_type = 'dt2' # hgt dt2
#fn_fig = './Amazon_topography/amazon_topo.png'
#!!!IMPORTANT!!!!!!!
save_topo = False
fn_npz = './data/ATTO_large.npz'
#!!!!!!!!!!!!!!!!!!!

## !!!!!Specifically for hgt files
ndim = 3601
dx = 30
# !!!!!!!!!!!

# Choose The LES domain requires choice of filetype, location, limits of plots( xr_lg and yr_lg), ndim
#loc_cen = (-2.413, -60.504)   # K34-like tower Bicheng et al. (−2.413, −60.504) the values used have been adjusted dt2 file
                              # xr_lg = ((-61, -60), (loc_cen[-1]-0.12, loc_cen[-1]+0.12)), yr_lg = ((-2.8, -2.2), (loc_cen[0]-0.06, loc_cen[0]+0.06))
                              # Use dt2 file 61
#--------------------------------
#loc_cen = (-2.6028, - 60.2098) # Exact K34 tower location :: xr_lg = ((-61, -60), (loc_cen[-1]-0.12, loc_cen[-1]+0.12)) :: yr_lg = ((-3.0, -2.4), (loc_cen[0]-0.06, loc_cen[0]+0.06))
                                # Use dt2 file 61
#--------------------------------
#loc_cen =  (-2.1468, -59.0068) #ATTO Tall Towrlocation
#loc_cen = (-2.1448, -59.0008) #ATTO Samller tower ## xr_lg = ((-59.5, -58.5), (loc_cen[-1]-0.12, loc_cen[-1]+0.12)) and yr_lg = ((-2.50, -2.00), (loc_cen[0]-0.06, loc_cen[0]+0.06))
#loc_cen = (-2.1458, -59.0038) # midpoint between small and tall tower

loc_cen = (39.8, -7.72)

# Combine two dt2 files order in increasing latitude :: Needed for ATTO topography
combine_dt2 = False
dt2s = ['60', '59']

# # For 101 point domain
# xspan = 3/111/2
# yspan = 3/111/2

npoint = 255 # generates LES domain of 100 grid points -- is 1 more than num input
xspan = npoint*30/1000*1/111/2 # domain grid number * dx / m to km * 1 lat to km /2 for both directions
yspan = npoint*30/1000*1/111/2 # domain grid number * dx / m to km * 1 lat to km /2 for both directions

xr_lg = ((-7.0, -8.0), (loc_cen[-1]-0.12, loc_cen[-1]+0.12)) # [0] Large plot Long dimensions [1] dashed boundries
yr_lg = ((39.0, 40), (loc_cen[0]-0.06, loc_cen[0]+0.06)) # [0] Large plot lon dimensions
xr_les = (loc_cen[-1]-xspan, loc_cen[-1]+xspan) # Used to define LES domain
yr_les = (loc_cen[0]-yspan, loc_cen[0]+yspan)

ag_rot = 0 # Degree

# Filter
npd = 10

# Figure
sz = dict(left=0.06, right=0.88, bottom=0.04, top=0.98,
  wspace=0.1, hspace=0.25)
lbd_amazon = (0, 130)
lbd_les = (0, 60)
lw = 2


## Function
def read_elevation_from_file(hgt_file, res=1):
  if res == 1:
    ndim = 3601
  elif res == 3:
    ndim = 1201

  with open(hgt_file, 'rb') as hgt_data:
    # Each data is 16bit signed integer(i2) - big endian(>)
    elevations = np.fromfile(hgt_data, np.dtype('>i2'), ndim*ndim)\
                            .reshape((ndim, ndim))
    return np.flip(elevations, axis=0)

def get_coord(hgt_file, res=1):
  if res == 1:
    ndim = 3601
  elif res == 3:
    ndim = 1201
  
  basename = os.path.basename(hgt_file)
  if basename[0] == 'N':
    lat0 = int(basename[1:3])
  elif  basename[0] == 'S':
    lat0 = -int(basename[1:3])
  
  if basename[3] == 'E':
    lon0 = int(basename[4:7])
  elif  basename[3] == 'W':
    lon0 = -int(basename[4:7])

  lat = np.linspace(lat0, lat0+1, ndim)
  lon = np.linspace(lon0, lon0+1, ndim)

  return lat, lon

def return_topo(dt2_file):
  dted_file = Path(dt2_file)
  tile = dd.Tile(dted_file)
  assert isinstance(tile.data, np.ndarray)
  ele_amazon = np.transpose(tile.data)
  origin = tile.dsi.south_west_corner
  x_point = tile.dsi.south_east_corner
  y_point = tile.dsi.north_west_corner
  ndim = np.shape(ele_amazon)[0]
  lat = np.linspace(origin.latitude, y_point.latitude, ndim)
  lon = np.linspace(origin.longitude, x_point.longitude, ndim)
  return ele_amazon, lat, lon

## Read the topography
if os.path.isfile(fn_hgt) or os.path.isfile(dt2_file):
  if data_type == 'hgt':
    ele_amazon = read_elevation_from_file(fn_hgt, res= 3)
    lat, lon = get_coord(fn_hgt, res= 3)
  elif data_type == 'dt2':
    if not(combine_dt2):
      ele_amazon, lat, lon = return_topo(dt2_file)
      ndim = np.shape(ele_amazon)[0]
      # dted_file = Path(dt2_file)
      # tile = dd.Tile(dted_file)
      # assert isinstance(tile.data, np.ndarray)
      # ele_amazon = np.transpose(tile.data)
      # origin = tile.dsi.south_west_corner
      # x_point = tile.dsi.south_east_corner
      # y_point = tile.dsi.north_west_corner
      # ndim = np.shape(ele_amazon)[0]
      # lat = np.linspace(origin.latitude,y_point.latitude, ndim)
      # lon = np.linspace(origin.longitude, x_point.longitude, ndim)
    elif combine_dt2:
      dt2_file1 = './data/s03_w0' + dt2s[0] + '_1arc_v3.dt2'
      dt2_file2 = './data/s03_w0' + dt2s[1] + '_1arc_v3.dt2'
      ele_amazon1, lat1, lon1 = return_topo(dt2_file1)
      ele_amazon2, lat2, lon2 = return_topo(dt2_file2)
      ele_amazon = np.hstack((ele_amazon1[:,:-1], ele_amazon2))
      lat = lat1
      lon = np.hstack((lon1[:-1],lon2))
      ndim = np.shape(ele_amazon1)[0]


    if ndim == 3601:
      dx = 30
  else:
    RuntimeError('No file found ')


# Load data from Marc's code rather than using the functions above
# topo = np.load('./data/Topo_data_Amazon_S03W59to60.npz')
# ele_amazon = topo['elevation_data']
# lat = topo['latitude']
# lon = topo['longitude']

## Process the data
# Amazon topography
n_lg = len(xr_lg)

# Crop topography in LES domain
ym, xm = np.mgrid[0:(npoint+1)*dx:dx, 0:(npoint+1)*dx:dx]

rad_rot = np.deg2rad(ag_rot)
if np.mod(ag_rot, 180) != 0:
  np_rot = int(
    np.ceil(npoint*(np.abs(np.cos(rad_rot))+np.abs(np.sin(rad_rot))))+10)
else:
  np_rot = npoint
dnp = (np_rot - npoint)//2
ixs = int(np.floor((xr_les[0]-np.floor(xr_les[0]))*(ndim-1)))-dnp
ixe = ixs+np_rot
iys = int(np.floor((yr_les[0]-np.floor(yr_les[0]))*(ndim-1)))-dnp
iye = iys+np_rot
ele_org = ele_amazon[iys:iye+1, ixs:ixe+1]
if data_type == 'hgt':
  xr_les = [np.min(lon)+ixs/1200.0, np.min(lon)+ixe/1200.0]
  yr_les = [np.min(lat)+ iys/1200.0, np.min(lat) + iye/1200.0]
# elif data_type == 'dt2':
#   xr_les = [np.min(lon)+ixs/3600., np.min(lon)+ixe/3600.0]
#   yr_les = [np.min(lat)+ iys/3600.0, np.min(lat) + iye/3600.0]
if np.mod(ag_rot, 180) != 0:
  nm = np_rot//2
  inds = nm - npoint//2
  inde = nm + npoint//2 + (npoint-npoint//2*2) + 1
  ele_org = rotate(ele_org, ag_rot, reshape=False)[inds:inde, inds:inde]

# Detrain the data
assert np.shape(ele_org)[0] != 0
ele_org = ele_org-ele_org.min()

# Smooth and make it period at the boundaries
ele = np.zeros(ele_org.shape)
ele_s = gaussian_filter(ele_org, sigma=npd/3, mode='wrap')
for ind in range(npd-1):
  ele[ind: npoint-ind+1, ind] =\
    ele_org[ind: npoint-ind+1, ind] * (ind+1)/npd\
    + ele_s[ind: npoint-ind+1, ind] * (npd-ind-1)/npd
  ele[ind: npoint-ind+1, npoint-ind] =\
    ele_org[ind: npoint-ind+1, npoint-ind] * (ind+1)/npd\
    + ele_s[ind: npoint-ind+1, npoint-ind] * (npd-ind-1)/npd
  ele[ind, ind+1:npoint-ind] =\
    ele_org[ind, ind+1:npoint-ind] * (ind+1)/npd\
    + ele_s[ind, ind+1:npoint-ind] * (npd-ind-1)/npd
  ele[npoint-ind, ind+1:npoint-ind] =\
    ele_org[npoint-ind, ind+1:npoint-ind] * (ind+1)/npd\
    + ele_s[npoint-ind, ind+1:npoint-ind] * (npd-ind-1)/npd
ele[npd-1: npoint-npd+2, npd-1: npoint-npd+2] =\
  ele_org[npd-1: npoint-npd+2, npd-1: npoint-npd+2]

mean = ele.mean()
ele = ele - ele.min()

# Save the new topography
if save_topo:
  print('Saving the npz file')
  np.savez(fn_npz, topo=ele)

ele_org = ele_org - ele_org.mean() + ele.mean()
print(ele_org.min(), ele_org.max(), ele_org.mean())
print(ele.min(), ele.max(), ele.mean())



## Plot the data
ifig = 0
fig = plt.figure(ifig, figsize=[8, 10])
gs = gridspec.GridSpec(3, 2, height_ratios=(1, 1, 0.8))
gs.update(**sz)

# Plot the Amazon topography
levels_ele = np.arange(np.min(ele_amazon), np.max(ele_amazon), 10)
cmap = cm.get_cmap("rainbow")

for ic in range(n_lg):
  ax = fig.add_subplot(gs[ic, :])
  ax.set_aspect('equal', adjustable='box')

  cax = ax.contourf(lon, lat, ele_amazon, levels_ele, cmap=cmap, extend='both')
  if ic != n_lg-1:
    ax.plot([xr_lg[ic+1][0], xr_lg[ic+1][0],
      xr_lg[ic+1][1], xr_lg[ic+1][1], xr_lg[ic+1][0]],
      [yr_lg[ic+1][0], yr_lg[ic+1][1],
      yr_lg[ic+1][1], yr_lg[ic+1][0], yr_lg[ic+1][0]],
      color='k', lw=lw, ls='--')
  else:
    # xr_les = (loc_cen[-1] + 1.5*xspan, loc_cen[-1] + 4*xspan)  # redefine for plot
    # yr_les = (loc_cen[0] - yspan, loc_cen[0] + yspan)
    ax.plot([xr_les[0], xr_les[0], xr_les[1], xr_les[1], xr_les[0]],
      [yr_les[0], yr_les[1], yr_les[1], yr_les[0], yr_les[0]],
      color='k', lw=lw, ls='--')
  if np.max(xr_lg) > np.max(lon):
    xr_lg = [xr_lg[0], (xr_lg[1][0], np.max(lon))]

  # Axis properties
  ax.set_xlim(xr_lg[ic])
  ax.set_ylim(yr_lg[ic])

  # Axis labels
  ax.set_xlabel('LON (deg.)')
  ax.set_ylabel('LAT (deg.)')
  ax.text(0, 1., '({seq})'.format(seq=chr(97+ic)),
    transform=ax.transAxes, weight='bold', ha='left', va='bottom')

grid_pos=gs.get_grid_positions(fig=fig)
right = grid_pos[3][-1]
bottom = grid_pos[0][n_lg-1]
rpad = 0.02
bpad = 0.15
wid = 0.02
hgt = 0.3
cbar_ax = fig.add_axes([right+rpad, bottom+bpad, wid, hgt])
cbar = fig.colorbar(cax, cax=cbar_ax, ticks = levels_ele)
cbar.ax.set_xlabel(r'hgt (m)')
cbar.ax.xaxis.set_label_position('top')

# Plot the topography in LES domain
levels = np.arange(lbd_les[0], lbd_les[1]+10, 5)

ax = plt.subplot(gs[n_lg, 0])
ax.set_aspect('equal', adjustable='box')
cax = ax.contourf(xm, ym, ele_org, levels, cmap=cmap, extend='both')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Detrended topography')
ax.text(0, 1., '({seq})'.format(seq=chr(97+n_lg)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

ax = plt.subplot(gs[n_lg, 1])
ax.set_aspect('equal', adjustable='box')
ax.contourf(xm, ym, np.abs(ele_org - ele), levels, cmap=cmap, extend='both')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Filtered topography')
ax.text(0, 1., '({seq})'.format(seq=chr(97+n_lg+1)),
  transform=ax.transAxes, weight='bold', ha='left', va='bottom')

right = grid_pos[3][-1]
bottom = grid_pos[0][n_lg]
rpad = 0.02
bpad = 0.01
wid = 0.02
hgt = 0.2
cbar_ax = fig.add_axes([right+rpad, bottom+bpad, wid, hgt])
cbar = fig.colorbar(cax, cax=cbar_ax, ticks = levels)
cbar.ax.set_xlabel(r'hgt (m)')
cbar.ax.xaxis.set_label_position('top')

plt.show()
# Save the figure
#fig.savefig(fn_fig, dpi=600)
