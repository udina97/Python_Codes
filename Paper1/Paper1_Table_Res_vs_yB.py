#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:26:57 2026

@author: u1450851
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.transforms import ScaledTranslation
from sklearn.linear_model import LinearRegression

#%%Set path to the profiles

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/'
cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat']
# cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps']
Nz_SLayer = 200
canopyH = 39

#%%Load the data into a dictionary and compute the slopes of the regression line

prof = dict()

for i in range(len(cases)):
    
    prof[cases[i]] = np.load(path_to_data + 'ResTKEvsYB_' + cases[i] + '_Q.npy')

#%%Compute the the zero-crossing on the TKE median and IQ


def positive_to_negative_crossing_yB(tke, yB):
    """
    Returns yB where tke crosses from positive to negative.

    If the TKE value hits exactly zero before going negative,
    returns the first yB where TKE is zero.
    Otherwise, linearly interpolates between the last positive
    and first negative TKE value.
    """

    tke = np.asarray(tke)
    yB = np.asarray(yB)

    for i in range(len(tke) - 1):

        # Case 1: exact zero followed by negative values
        if tke[i] == 0 and tke[i + 1] < 0:
            return yB[i]

        # Case 2: positive followed by exact zero
        # Check whether that zero is followed by a negative later
        if tke[i] > 0 and tke[i + 1] == 0:
            j = i + 1

            while j < len(tke) and tke[j] == 0:
                j += 1

            if j < len(tke) and tke[j] < 0:
                return yB[i + 1]

        # Case 3: positive followed directly by negative
        if tke[i] > 0 and tke[i + 1] < 0:
            tke0, tke1 = tke[i], tke[i + 1]
            yB0, yB1 = yB[i], yB[i + 1]

            yB_cross = yB0 + (0 - tke0) * (yB1 - yB0) / (tke1 - tke0)

            return round(float(yB_cross), 3)

    return None


def crossings_for_array(arr):
    """
    arr shape: (4, 28)

    Row 0: TKE residual median
    Row 1: TKE residual IQ25
    Row 2: TKE residual IQ75
    Row 3: mean yB bin value
    """

    tke_median = arr[0]
    tke_iq25 = arr[1]
    tke_iq75 = arr[2]
    yB = arr[3]

    return {
        "median": positive_to_negative_crossing_yB(tke_median, yB),
        "IQ25": positive_to_negative_crossing_yB(tke_iq25, yB),
        "IQ75": positive_to_negative_crossing_yB(tke_iq75, yB),
    }

crossing_yB = {
    key: crossings_for_array(arr)
    for key, arr in prof.items()
}


#%%Compute the mean of the crossing values

tmp = np.zeros((len(cases)))
for i in range(len(cases)):
    tmp[i] = crossing_yB[cases[i]]['IQ75']
    
print(np.mean(tmp))


















































