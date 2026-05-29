
from scipy.io import netcdf
import netCDF4 as nc
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import utm
from math import sin, cos, sqrt, atan2, radians, ceil
import photutils.utils as ph
import access_data as ad
import data_statistics as ds
from photutils.utils import ShepardIDWInterpolator as idw
from scipy.interpolate import Rbf
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import random as rand
import plot_data as plotd
from scipy.ndimage.interpolation import rotate
import itertools







def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z



def dist(long1, lat1, long2, lat2):
    R = 6373.0
    lat1 = radians(lat1*10**-3)
    lon1 = radians(long1*10**-3)
    lat2 = radians(lat2*10**-3)
    lon2 = radians(long2*10**-3)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
def tower_loc_lat():
    locations = {}
    locations['tnw02'] = [-7.7505872,39.7104228]
    locations['tse02'] = [-7.7487822, 39.7031292]

def distance_cartesian(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2

    return sqrt(dx * dx + dy * dy)


def tower_location():
    path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_data/SENSOR_COORD.xlsx'
    locs = pd.read_excel(path,index_col=0)
    lat = locs['Northing']
    long = locs['Easting']
    tower_names = locs['NCAR name']
    towers = {}
    for i in range(1,locs.shape[0]):
        tmp_str = tower_names[i]
        towers[tmp_str] = [long[i],lat[i]]

    return towers


def Rbf_opt(posR1,posR2,val):
    k_pos1 = []; k_pos2 = []; k_val = []
    minerrv = []; epsminv = [];  errdict = {}

    for j in range(0,50):
        minerr = 1000; epsmin = 0; err = 1000;
        pos1 = posR1.copy();pos2 = posR2.copy(); v = val.copy()
        for i in range(0,6):
            it = rand.randrange(0, len(pos1), 1)
            k_pos1.append(pos1.pop(it)); k_pos2.append(pos2.pop(it)); k_val.append(v.pop(it))
        for eps in range(0,100,2):
            rbfi = Rbf(pos1, pos2, v, function='cubic', epsilon=eps)
            err = np.sum(np.abs(rbfi(k_pos1,k_pos2)-k_val));
            if err<minerr:
                epsmin = eps; minerr = err;
        minerrv.append(minerr); epsminv.append(epsmin)
    mintot = 1000; minit= 0; maxdict = 0; maxeps = 0;
    for j in range(0,50):
        if epsminv[j] in errdict.keys():
            errdict[epsminv[j]] += 1
        else:
            errdict[epsminv[j]] = 1
        if mintot>minerrv[j]:
            minit = j
            mintot = minerrv[j]
    errdict[epsminv[minit]] += 6
    for key in errdict:
        if errdict[key] > maxdict:
            maxdict = errdict[key]; maxeps = key

    return maxeps


def interp_temp(towers,temp_mat):
    pos = [];posR1 = [];posR2 = []
    val = []
    temp_mat2 = np.zeros(np.shape(temp_mat))
    # posR1.append(5); posR1.append(5);posR1.append(122);posR1.append(122); posR1.append(61); posR1.append(5); posR1.append(122); posR1.append(61)
    # posR2.append(5);posR2.append(130);posR2.append(5);posR2.append(130);  posR2.append(5); posR2.append(65) ; posR2.append(65); posR2.append(130)
    # val.append(mean); val.append(mean); val.append(mean); val.append(mean); val.append(mean); val.append(mean);val.append(mean); val.append(mean)

    shape = np.shape(temp_mat);
    x = shape[0];y = shape[0]; X,Y = np.meshgrid(x,y)
    for key in towers:
        pos.append(towers[key][2]);posR1.append(towers[key][2][0])
        val.append(towers[key][0][0][0]);posR2.append(towers[key][2][1])

    # plot_matrix(temp_mat)
    #eps  = Rbf_opt(posR1,posR2,val)
    rbfi = Rbf(posR1, posR2, val, function='cubic', epsilon=1)

    # pos = np.array(np.zeros([len(posR1),2]))
    # for i in range(0,len(posR1)):
    #     pos[i,0] = posR1[i]; pos[i,1] = posR2[i]
    # f = idw(pos,val)
    # for i in range(0,np.shape(temp_mat)[0]):
    #    for j in range(0,np.shape(temp_mat)[1]):
    #        temp_mat[i,j] = f([i,j])
    # temp_mat = savgol_filter(temp_mat,9,3)


    # temp_mat2 = float(rbfi(X, Y))
    # for i in range(0, np.shape(temp_mat)[0]):
    #     for j in range(0,np.shape(temp_mat)[1]):
    #         temp_mat2[i,j] = float(rbfi(i,j))


    # fig = plt.figure(figsize=(10, 6))
    # c = plt.pcolor(temp_mat2)
    # fig.colorbar(c)
    # plt.show
    return temp_mat2

def main_fit(dat):
    # Generate Data...
    x = np.array([]); y = np.array([])
    for i in range(0,len(dat)):
        x = np.append(x,dat[i][0]); y = np.append(y,dat[i][1]);

    z = np.polyfit(x,y,3); f = np.poly1d(z)
    # xt = np.linspace(70,180,100); yt = f(xt)
    # plt.figure()
    # plt.scatter(xt,yt)
    # plt.scatter(x,y)
    # plt.show()
    return f

def construct_mat(matrix,line,f):
    max = 0; min = 1000
    matrix_new = np.zeros(np.shape(matrix))
    for point in line:
        xit = point[0]
        if min > xit:
            min = xit
        elif max < xit:
            max= xit
    for i in range(min,max):
        matrix_new[:,i] = f(i)
    return matrix_new

def line_fit_method(temp_mat_rot, towers):
    mask = temp_mat_rot > 0.0;
    mean = np.mean(temp_mat_rot[mask])

    # for j in range(0, np.shape(temp_mat_rot)[1]):
    #     if np.mean(temp_mat_rot[:,j])==0.0 and j< 67:
    #             temp_mat_rot[:,j] == mean
    # temp_mat_rot[:,71] = towers['Tsoil_0_6cm_tse01'][0][0][0]
    # v1 = [];v2 = []; val1 =[]; val2 =[]; pos = []; z = np.array([])
    line_nw = []
    line_se = []
    range_nw = []
    range_se = []

    for key in towers:
        tmp = key[-5:]
        if tmp[0:3] == 'tnw':
            line_nw.append([towers[key][2][1],towers[key][0][0][0]])
            range_nw.append(towers[key][2][0])

        elif tmp[0:3] == 'tse':
            line_se.append([towers[key][2][1], towers[key][0][0][0]])
            range_se.append(towers[key][2][0])
    distance = np.abs(np.max(range_se) - np.min(range_nw))
    # a,b, c,d = np.polyfit(y, x, 3); fit2 = lambda z: a * z**3+ b*z**2+c**2 + d
    f_nw = main_fit(line_nw)
    f_se = main_fit(line_se)
    mat_nw = construct_mat(temp_mat_rot,line_nw,f_nw)
    mat_se = construct_mat(temp_mat_rot,line_se,f_se)
    temp_mat_rot2 = temp_mat_rot
    for j in range(0,np.shape(mat_nw)[1]):

        if np.mean(mat_nw[:, j]) == 0.0 and np.mean(mat_se[:, j]) == 0.0:
            pass
        elif np.mean(mat_nw[:, j]) != 0.0 and np.mean(mat_se[:, j]) == 0.0:
            temp_mat_rot[:, j] = mat_nw[:, j]
        elif np.mean(mat_se[:, j]) != 0.0 and np.mean(mat_nw[:, j]) == 0.0:
            temp_mat_rot[:, j] = mat_se[:, j]
        else:
            for i in range(0, np.shape(mat_nw)[0]):
                if i> np.max(range_se) and i< np.min(range_nw):
                    temp_mat_rot[i,j] = np.abs(i-np.min(range_nw))/distance*mat_se[i,j]+np.abs(i-np.max(range_se))/distance*mat_nw[i,j]

                elif i<= np.max(range_se):
                    temp_mat_rot[i,j] = mat_se[i,j]

                elif i>= np.min(range_nw):
                    temp_mat_rot[i,j] = mat_nw[i,j]

    for i in range(0, np.shape(mat_nw)[0]):
        for j in range(0, np.shape(mat_nw)[1]):
            if temp_mat_rot[i,j]>1.0:
                for key in towers:
                    distance  = np.sqrt((towers[key][2][0]-i)**2+(towers[key][2][1]-j)**2)/10
                    temp_mat_rot[i,j] = (1-np.exp(-distance**2))*temp_mat_rot[i,j]+np.exp(-distance**2)*towers[key][0][0][0]
                    if i == 55 and j == 107:
                        pop  =1


    # plotd.plot_matrix(temp_mat_rot)

 #   np.savetxt("surface_temp_mat.csv",temp_mat_rot,delimiter=',')
#    pd.DataFrame(np_array).to_csv("~/../calaf-group2/Zev_research2/perdigao_simulation_results/surface_temp_mat.csv")

    # temp_mat_rot2[i+1,j] = temp_mat_rot2[i,j];
                # temp_mat_rot2[i , j+1] = temp_mat_rot2[i, j];
                # temp_mat_rot2[i+1, j + 1] = temp_mat_rot2[i, j];
                # temp_mat_rot2[i - 1, j] = temp_mat_rot2[i, j];
                # temp_mat_rot2[i, j - 1] = temp_mat_rot2[i, j];
                # temp_mat_rot2[i - 1, j - 1] = temp_mat_rot2[i, j];
    pop = 1

    return temp_mat_rot









    # tic = 0; tiv = []; tic_loc = []
    # for j in range(72, 185):
    #     for i in range(26, 102):
    #         if temp_mat_rot[i,j] > 0.0:
    #             tic += 1
    #             tiv = temp_mat_rot[i,j]; tic_loc.append([i,j])
    #
    #         elif j>= 67 and j< 71:
    #             pass
    #         elif j==71:
    #             temp_mat_rot[:,j] ==
    #         elif j<89 and j>71:
    #             pass
    #         elif j==89:
    #             temp_mat_rot[:,89] == (towers['Tsoil_0_6cm_tse02'][0][0][0]+towers['Tsoil_0_6cm_tnw02'][0][0][0])/2
    #



def squeeze_grid(towers,iter):
    minx = 10000;
    maxx = 0;
    miny = 10000;
    maxy = 0;
    for key in towers:
        xit = towers[key][iter][0];
        yit = towers[key][iter][1];
        # else:
        #     tmp = towers[key][0]

        if minx > xit:
            minx = xit
        elif maxx < xit:
            maxx = xit
        if miny > yit:
            miny = yit
        elif maxy < yit:
            maxy = yit

    return minx,miny,maxx,maxy
def create_grid(towers):
    # x0 = [39.50,-7.735]; x1 = [39.50,-7.48]; x2 = [39.90, -7.735]; x3 = [39.90,-7.48]
    x0 = [12309.24,-18671.38]; x1 = [56176.32,-18477.50]; x2 = [12238.38,25740.19];
    # x0 = utm.from_latlon(x0[0],x0[1]); x1 = utm.from_latlon(x1[0], x1[1]);
    # x2 = utm.from_latlon(x2[0], x2[1]); x3 = utm.from_latlon(x3[0], x3[1])
    dist1 = distance_cartesian(x0[0],x0[1],x1[0],x1[1]); dist2 = distance_cartesian(x0[0], x0[1], x2[0], x2[1])
    x = np.arange(0,dist1,20); y = np.arange(0,dist2,20)
    [X,Y] = np.meshgrid(x,y); theta = np.pi/(5.025); xshift = 100; yshift= 100;
    temp_mat = np.zeros(np.shape(X)); temp_mat_rot =  np.zeros(np.shape(X));
    rotate_mat = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])

    for key in towers:
        xdist = np.abs(x0[0]-towers[key][1][0]);  ydist = np.abs(x0[1]-towers[key][1][1])
        xit = ceil(xdist/20); yit = ceil(ydist/20);
        tmp = xit; xit = yit; yit = tmp;
        temp_mat[xit,yit] = towers[key][0][0]
        # if len(key)>5:
        towers[key] = [towers[key],[xit,yit]];
        tmp = np.matmul(rotate_mat, np.asarray(towers[key][1], dtype=np.float32).T);
        tmp = np.ndarray.tolist(tmp); tmp[0]= int(tmp[0]); tmp[1]=int(tmp[1])
        towers[key].append(tmp)
        # temp_mat_rot [int(tmp[0]),int(tmp[1])] = towers[key][0][0][0]
        if tmp[1]+yshift<0:
            yshift += 100+abs(tmp[1])
        elif tmp[0]+xshift<0:
            xshift += 100+abs(tmp[0])
    for key in towers:
        towers[key][2][0]+=xshift; towers[key][2][1]+= yshift
        temp_mat_rot[towers[key][2][0],towers[key][2][1]] = towers[key][0][0][0]
    minx,miny,maxx,maxy = squeeze_grid(towers,1)
    minx_rot, miny_rot, maxx_rot, maxy_rot = squeeze_grid(towers, 2)
    temp_mat = temp_mat[minx-12 : maxx+12,miny-79 : maxy+80]
    temp_mat_rot = temp_mat_rot[minx_rot-(26-7) : maxx_rot+20 , miny_rot-(70-3) : maxy_rot+70 ]
    # temp_mat_rot = np.flip(temp_mat_rot)
    # fig = plt.figure(figsize=(10, 6))
    # c = plt.pcolor(temp_mat_rot, cmap='Reds')
    # fig.colorbar(c)

    # plt.show


    for key in towers:
        towers[key][1][0] = towers[key][1][0]-(minx-12)# +62
        towers[key][1][1] = towers[key][1][1] - (miny - 80)
        towers[key][2][0] = towers[key][2][0]-(minx_rot-(26-7))
        towers[key][2][1] = towers[key][2][1] - (miny_rot -(70-3) )
        tmp = towers[key][1];

        # xit = int(np.ceil(towers[key][1][0])); yit = int(np.ceil(towers[key][1][1]));
    fig = plt.figure(figsize=(10, 6))
    c = plt.pcolor(temp_mat_rot, cmap='Reds')
    fig.colorbar(c)


    line_mat = line_fit_method(temp_mat_rot, towers)

    return(temp_mat_rot,line_mat,towers)

def soil_temp_spatial(t1,t2):
    tower_loc = tower_location()
    tower_loc.pop('tse12.10m',None); tower_loc.pop('v07.2m',None)
    Tsoil_list = ad.name_list('Ts')
    temp_avg = {}
    temp_interp = fit_temperature('T_', t1, t2)
    for name in Tsoil_list:
        temp_avg[name] = [ds.avg_low_freq(name,t1,t2)]
        print('soiltemp ',name,' ',ds.avg_low_freq(name,t1,t2))
        for key in tower_loc:
            key = str(key)
            if key[0:5] == name[-5:] and len(temp_avg[name])<2 and str(tower_loc[key][0])!= 'nan':
                    # temp_avg[name] = [temp_avg[name],utm.from_latlon(tower_loc[key][0]*10**-3,tower_loc[key][1]*10**-3)]

                temp_avg[name] = [temp_avg[name],tower_loc[key] ]

            elif key[0:3] == name[-3:] and len(temp_avg[name])<2 and str(tower_loc[key][0])!= 'nan':
                    temp_avg[name] = [temp_avg[name],tower_loc[key]]
    list_of_crossover = []
    for key in temp_interp:
        inn = 0
        for key2 in temp_avg:
            if key == key2[-5:]:
                list_of_crossover.append(key)
                inn += 1

    for i in range(0,len(list_of_crossover)):
        temp_interp.pop(list_of_crossover[i])

    for key in tower_loc:
        if type(key) == str:
            if key[0:5] in temp_interp:
                temp_avg[key[0:5]] = [temp_interp[key[0:5]]];
                if str((tower_loc[key][0])) == 'nan' or str((tower_loc[key][0])) == 'Does not exist!':
                    key = key[0:6]+'10m'
                if str((tower_loc[key][0])) == 'nan' or str((tower_loc[key][0])) == 'Does not exist!':
                    key = key[0:6]+'20m'
                temp_avg[key[0:5]] = [temp_avg[key[0:5]],tower_loc[key]]
                temp_interp.pop(key[0:5])
    keys_TBD = []
    for key in temp_avg.keys():
        if temp_avg[key][0][0] == None:
            keys_TBD.append(key)
    [temp_avg.pop(keys_TBD[i]) for i in range(0,len(keys_TBD))]
    #x = distance_cartesian(temp_avg['Tsoil_0_6cm_tnw02'][1][0], temp_avg['Tsoil_0_6cm_tnw02'][1][1], temp_avg['Tsoil_0_6cm_tse01'][1][0],
    #    temp_avg['Tsoil_0_6cm_tnw05'][1][1])
    temp_mat,line_mat,towers = create_grid(temp_avg)
    temperature = interp_temp(towers,temp_mat)

def fit_temperature(var,t1,t2,rate = 'high_rate',deriv = False):

    organized = ad.orginize_by_tower(var,rate)
    tower_and_interp = {}
    for key in organized:
        avg = []; it = 0; y = np.array([]); x = np.array([]); ij = 0
        for keys in organized[key]:
            if key == 'tnw16':
                pop = 1
            av = ds.avg_high_freq(keys[0]);
            if not(math.isnan(av)):
                avg.append([av]); avg[it].append(keys[1]); it += 1
        for iters in range(1,len(avg)):
            for it in range(0,len(avg)-iters):
                if avg[it][1] > avg[it+1][1]:
                    tmp = avg[it]; avg[it] = avg[it+1]; avg[it+1] = tmp

        for j in range(0,len(avg)-ij):
            if avg[j][0] == None:
                pass
            else:
                y = np.append(y,avg[j].pop(1))
                x = np.append(x,avg[j].pop(0))
        # avg = np.asarray(avg,dtype=np.float32); y = np.asarray(y,dtype=np.float32);

        if len(x)>=3:
            a,b = np.polyfit(np.log(y),x,1); fit = lambda z: a*np.log(z) +b
            c,d = np.polyfit(y,x,1); fit2 = lambda z: c*z+ d
            temp_0 = .50*fit(.01)+.50*fit2(.01)
            tower_and_interp[key] = temp_0
        elif len(x) == 2:
            c, d = np.polyfit(y, x, 1);  fit2 = lambda z: c * z + d; temp_0 = fit2(.1)
            tower_and_interp[key] = temp_0
        else:
            pass
        #if key[0:3] == 'tse':
        xt = np.linspace(.01,100,200)
        plt.figure()
        plt.scatter(x+273, y)
        plt.plot(.50*fit(xt)+.50*fit2(xt)+273,xt)
        plt.title(key)
        if deriv and key == 'tse04':
            if len(x) == 0:
                return np.nan
            else:
                return np.gradient(x,y)
        print('interp temp ',key,' ', temp_0)
    return tower_and_interp
        # plotd.plot_variable(x,y,fit,fit2)
        # Temperature plots look really bad there might be an issue with order with y vs avg

