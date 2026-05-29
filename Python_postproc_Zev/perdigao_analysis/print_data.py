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
import data_statistics as ds


def print_names(str,rate,start = 'b'):
    if rate == 'high_rate':
        data = ad.init_high()
    else:
        data = ad.init_low()
    # for var in data.variables.values():
    #     if var._name[-2] + var._name[-1] == str:
    #         print(var._name)
    if start == 'b':
        for var in data.variables.values():

            if var._name[0:len(str)] == str:
                print(var._name) #,' avg ',ds.avg_low_freq(var._name,12,14))

    elif start == 'e':
        for var in data.variables.values():
            if var._name[(len(str)+1):] == str:
                print(var._name)






def print_all(rate):
    if rate == 'high_rate':
        data = ad.init_high()
    else:
        data = ad.init_low()
    for var in data.variables.values():
        print(var._name)
def print_data_values(str):
    data = ad.init()
    print(data[str])
    tmp = data[str][:]
    print(type(tmp))
    print(data[str][:])
def test_fill():
    data = ad.init()
    for var in data.variables:
        data_shape = data[var].shape
        if len(data_shape) > 0:
            if data_shape[0] == 3600 and len(data_shape) == 2:
                if np.mean(data[var][100]) < 1.e+30:
                    print('less than fill value', var)
                if np.mean(data[var][100]) == 0.0:
                    print('variable = 0', var)

            elif data_shape[0] == 3600 and len(data_shape) == 1:
                if np.mean(data[var][100:200]) < 1.e+30:
                    print('less than fill value', var)
                if np.mean(data[var][100:200]) == 0.0:
                    print('variable = 0', var)
        else:
            print(var)