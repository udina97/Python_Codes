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

def plot_variable(var,strr = None, date = None):

    y = np.linspace(0,len(var),len(var))

    plt.figure(figsize=(10,6))
    plt.scatter(y,var, color = 'b')
    plt.title(strr + ' ' + date)
    plt.xlabel('time')
    #plt.show()



    plt.show()
def plot_profile(x,y,strr = None):
    plt.figure(figsize=(10,6))
    plt.plot(x,y, color = 'b')
    plt.xlabel(strr)
    plt.ylabel('height')
    plt.show()
def plot_matrix(Z):
    fig = plt.figure(figsize=(10, 6))
    c = plt.pcolor(Z )#cmap = 'Reds')
    fig.colorbar(c)

    plt.show()
    return

def plot_custom(var,vary,fit,fit2):
    plt.figure(figsize=(10, 6))
    plt.scatter(var, vary, label=var)
    plt.plot(fit(np.arange(.1,100,.1)),np.arange(.1,100,.1))
    plt.plot(fit2(np.arange(.1, 100, .1)), np.arange(.1, 100, .1))
    plt.plot((.25*fit(np.arange(.1,100,.1))+.75*fit2(np.arange(.1, 100, .1))),np.arange(.1, 100, .1))
    plt.title('var')
    plt.xlabel('time')