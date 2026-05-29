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
from photutils.utils import ShepardIDWInterpolator as idw
from scipy.interpolate import Rbf
import random as rand


def avg_high_freq(str,div = 1):
    count = 0
    data = ad.init_high()
    vec_len = len(data[str][:])/div
    dat = ad.clean_data(data[str][:])
    data_shape = dat.shape
    vec_len = ceil(data_shape[0]/div)
    # avg_mat = np.zeros((vec_len, div))
    avg_list = []
    for j in range(0,div):
        if div == 1:
            tmp = dat[0:vec_len];
            return np.mean(tmp)
        else:
            tmp = dat[0:vec_len]; dat = dat[vec_len+1:]
        avg_list.append(np.mean(tmp))
    if data_shape[0] == 0:
        avg = np.nan
        return avg

    return avg_list

def avg_low_freq(dstr,t1,t2, div = 1):

    time1 = t1 * 12
    time2 = t2 * 12

    data = ad.init_low()
    tmp = ad.my_data[dstr][time1:time2]
    mask = tmp < 1000.0
    TBA = tmp[mask]
    mask = TBA != np.nan
    TBA = TBA[mask]
    if len(TBA) ==0:
        avg = np.nan
        return avg
    avg = np.sum(TBA)/len(TBA)
    return avg
def fix_length(var1_prime,var2_prime):
    avg_prime = []
    if len(var1_prime) > len(var2_prime):
        for i in range(1,len(var1_prime)):
            if i%20 == 0:
                avg_prime.append(np.mean(var1_prime[i-20:i]))
        if len(avg_prime) > len(var2_prime):
            while len(avg_prime) > len(var2_prime):
                del avg_prime[-1]

        else:
            while len(avg_prime) < len(var2_prime):
                var2_prime = var2_prime[:-1]

    else:
        for i in range(1,len(var2_prime)):
            if i%20 == 0:
                avg_prime.append(np.mean(var2_prime[i-20:i]))
        if len(avg_prime) > len(var1_prime):
            while len(avg_prime) > len(var1_prime):
                del avg_prime[-1]

        else:
            while len(avg_prime) < len(var1_prime):
                var1_prime = var1_prime[:-1]

    return var1_prime,var2_prime,np.asarray(avg_prime)
def calc_wind_angle(dirstr,t1,t2):
    # u_averages = avg_high_freq(ustr,2)
    # v_averages = avg_high_freq(vstr,2)
    # if np.any(np.isnan(u_averages)):
    #     return np.nan
    # theta1 = np.arctan(v_averages[0]/u_averages[0])*180/np.pi; theta2 = np.arctan(v_averages[1]/u_averages[1])*180/np.pi
    # if theta1 < 0.0:
    #     theta1 = 360+theta1
    # if theta2 < 0.0:
    #     theta2  = 360+theta2
    theta = avg_low_freq(dirstr,t1,t2)
    return theta

def calc_vel_mag(ustr,vstr,wstr):
    u_average = avg_high_freq(ustr, 1)
    v_average = avg_high_freq(vstr, 1)
    w_average = avg_high_freq(wstr, 1)
    if np.any(np.isnan(u_average)):
        return np.nan
    else:
        return np.sqrt(u_average**2+v_average**2+w_average**2)
def calc_fluctuations(str):
    data = ad.init_high()
    d1 = ad.clean_data(data[str][0:ceil(len(data[str][:])/2)]); d2 = ad.clean_data(data[str][ceil(len(data[str][:])/2)+1:])
    if len(d1) == 0 or len(d2) == 0:

        return 'Fill_value'
    d1_bar = np.mean(d1); d2_bar = np.mean(d2)
    d1_prime = d1-d1_bar; d2_prime = d2-d2_bar
    d1_prime = np.append(d1_prime,d2_prime)
    return d1_prime
def calc_flux(str1,str2):
    var1_prime = calc_fluctuations(str1)
    var2_prime = calc_fluctuations(str2)
    if len(var1_prime) != len(var2_prime):
        var1_prime, var2_prime,prime_avg = fix_length(var1_prime,var2_prime)
        if len(var1_prime) > len(var2_prime):
            var1_prime = prime_avg
        else:
            var2_prime = prime_avg
    if type(var1_prime) == str or type(var2_prime) == str:
        return np.nan
    return np.sum(np.multiply(var1_prime,var2_prime))/len(var1_prime)
def calc_u_star(ustr,vstr,wstr):
    ustar = sqrt(calc_flux(ustr,wstr)**2+calc_flux(vstr,wstr)**2)
    if type(ustar) == str:
        return 'Fill_value'
    else:
        return ustar
def calc_TKE(ustr,vstr,wstr):
    return np.sqrt(calc_fluctuations(ustr)**2+calc_fluctuations(vstr)**2+calc_fluctuations(wstr)**2)
def calc_obukov(ustr,vstr,wstr,Tstr):
    vonk = .4; g = 9.81;
    u_star = calc_u_star(ustr,vstr,wstr)

    q_flux = calc_flux(wstr,Tstr) # should be the Temperature at the surface
    if type(u_star) == str or type(q_flux) == str or type(u_star) == np.nan or type(q_flux) == np.nan:
        print(str, ' Lost to the fill value noooo!')
        return None
    T_avg = avg_high_freq(Tstr,div = 1)
    return -T_avg*u_star**3/(vonk*g*q_flux)