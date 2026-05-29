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
import data_statistics as ds
import plot_data as plotd
from scipy.stats import entropy
def init_high():
    global my_data
    my_data = data_high
    return my_data

def init_low():
    global my_data
    my_data = data_low
    return my_data


def single_hour_or_day(hr,date,rate):
    if rate == 'high_rate':
        path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_data/perdigao_data_' +rate+'/' + date + '/' \
           + 'isfs_qc_tiltcor_hr_2017' + date + '_' + str(hr) + '.nc'
        print(path)
        global data_high
        data_high = nc.Dataset(path, 'r', format="NETCDF4")
        data_high.set_auto_mask(False)
        data_high.set_auto_scale(False)
    else:
        path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_data/perdigao_data_' + rate + '/' \
               + 'isfs_qc_tiltcor_2017' + date + '.nc'
        global data_low
        data_low = nc.Dataset(path, 'r', format="NETCDF4")
        data_low.set_auto_mask(False)
        data_low.set_auto_scale(False)

def variable_over_time(date,rate,var, hrstart = None,hrend = None, plot = False, stat = False):

    hour_list = []
    if not(hrstart==None):
        for i in range(hrstart,hrend):
            if i<10:
                hour_list += ['_0'+str(i)]
            elif i>=10:
                hour_list += ['_'+str(i)]
    global data
    averages = np.array([])
    if rate == 'high_rate':
        for hr in hour_list:
            path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_data/perdigao_data_' + rate + '/' + date + '/' \
                       + 'isfs_qc_tiltcor_hr_2017' + date + hr + '.nc'
            print(path)
            data = nc.Dataset(path, 'r', format="NETCDF4")
            data.set_auto_mask(False)
            data.set_auto_scale(False)

            [avg1, avg2] = ds.avg_30min_high_freq(var)
            averages = np.append(averages, np.array([avg1, avg2]))
        if plot: plot_variable(averages)

        return averages
    else:
        path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_data/perdigao_data_' + rate + '/' \
               + 'isfs_qc_tiltcor_2017' + date + '.nc'
        data = nc.Dataset(path, 'r', format="NETCDF4"); data.set_auto_mask(False); data.set_auto_scale(False)
        try:
            mask = return_data_values(var)<10000;
        except:
            print('tse02 not here')
            var = 'Rsw_in_20m_tse12'
            mask = return_data_values(var) < 10000;
        out_var = return_data_values(var)[mask]
        mask = ~np.isnan(out_var); out_var = out_var[mask]
        mask = ~np.isinf(out_var); out_var = out_var[mask]
        if plot:
            plotd.plot_variable(out_var,var,date)
        if stat:
            try:
                tmp = np.std(np.gradient(out_var))
                return tmp
            except:
                return np.nan

def return_data_values(str):
    var = data[str][:]
    return var
def clean_data(var):
    # cleans data as Fill value, nan and inf
    mask = var < 10000;
    out_var = var[mask]
    mask = ~np.isnan(out_var);
    out_var = out_var[mask]
    mask = ~np.isinf(out_var);
    out_var = out_var[mask]
    return out_var
def name_list(str):
    # returns list of towers
    str_list = []
    for var in data_low.variables.values(): # at some point may need to differentiate between low and high with inputs
        if var._name[0] + var._name[1] == str and var._name[6] == '0':
            str_list.append(var._name)
    return str_list

def orginize_by_tower(var,rate):
    # function organizes variable for tower height and variable name
    ordered_var = {}
    ordered_var['fill'] = 'fill'
    if rate == 'high_rate':
        data = init_high()
    else:
        data = init_low()
    for variable in data.variables.values():
       if variable._name[0:len(var)] == var:
           if  variable._name[-5:] in ordered_var:
               tmp = ordered_var[variable._name[-5:]]
               tmp.append([variable._name])
               ordered_var[variable._name[-5:]]= tmp
               # ordered_var[variable._name[-5:]] = ordered_var[variable._name[-5:]].append(variable._name)
           else:
                ordered_var[variable._name[-5:]] = [[variable._name]]
    ordered_var.pop('fill', None)
    for key in ordered_var:
        i = 0
        for keys in ordered_var[key]:
            num = ''
            for it in range(0,len(keys[0])-3):
                if keys[0][it].isdigit():
                    num += keys[0][it]
            ordered_var[key][i].append(int(num))
            i+=1
    for key in ordered_var:
        for iters in range(1, len(ordered_var[key])):
            for it in range(0, len(ordered_var[key]) - iters):
                if ordered_var[key][it][1] > ordered_var[key][it + 1][1]:
                    tmp = ordered_var[key][it];
                    ordered_var[key][it] = ordered_var[key][it + 1];
                    ordered_var[key][it + 1] = tmp
    return ordered_var

def tower_to_profile(var,rate):
    organized = orginize_by_tower(var, rate)
    tower_and_profile = {}
    for key in organized:
        avg = [];
        it = 0;
        y = np.array([]);
        x = np.array([]);
        ij = 0
        for keys in organized[key]:

            av = ds.avg_high_freq(keys[0]);
            if not (math.isnan(av)):
                avg.append([av]);
                avg[it].append(keys[1]);
                it += 1
        for iters in range(1, len(avg)):
            for it in range(0, len(avg) - iters):
                if avg[it][1] > avg[it + 1][1]:
                    tmp = avg[it];
                    avg[it] = avg[it + 1];
                    avg[it + 1] = tmp

        for j in range(0, len(avg) - ij):
            if avg[j][0] == None:
                pass
            else:
                y = np.append(y, avg[j].pop(1))
                x = np.append(x, avg[j].pop(0))
        tower_and_profile[key] = [x,y]
    return tower_and_profile
        # avg = np.asarray(avg,dtype=np.float32); y = np.asarray(y,dtype=np.float32);


def topo_rough_read():
    path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/perdigao_simulation_results/validation/'
    # topo_path = 'newa_perdigao_map_topo.nc'
    rough_path = 'newa_perdigao_map_roug_summer.nc'

    # Currently topo data set is broken either email or redownload
    # topo = nc.Dataset(path + topo_path, 'r', format="NETCDF4")
    # topo.set_auto_mask(False)
    # topo.set_auto_scale(False)

    rough = nc.Dataset(path + rough_path, 'r', format="NETCDF4")
    rough.set_auto_mask(False)
    rough.set_auto_scale(False)

    lon = rough['lon'][:]
    lat = rough['lat'][:]
    z0 = rough['z0'][:]
    pop = 1

    return topo, rough

