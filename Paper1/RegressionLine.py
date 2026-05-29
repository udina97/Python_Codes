#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:26:37 2025

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

start = 7
end = 14

prof = dict()

for i in range(len(cases)):
    
    prof[cases[i]] = np.load(path_to_data + 'ResTKEvsYB_' + cases[i] + '_Q.npy')

y_TKE = np.concatenate([prof['Gap_12_9mps'][0,start:end],prof['Gap_8_9mps'][0,start:end],prof['Gap_4_9mps'][0,start:end],prof['Patch_12_9mps'][0,start:end],\
                        prof['Patch_8_9mps'][0,start:end],prof['Patch_4_9mps'][0,start:end],prof['ATTO'][0,start:end],prof['Sinusoidal'][0,start:end],\
                            prof['Flat'][0,start:end]])
# x_YB = np.tile(prof['Flat'][3,7:14],9).reshape(-1, 1)

# # Fit regression
# model = LinearRegression()
# model.fit(x_YB, y_TKE)

# slope = model.coef_[0]
# intercept = model.intercept_
# r2 = model.score(x_YB, y_TKE)

# slopes = np.zeros((len(cases)))
# intercepts = np.zeros((len(cases)))

# for i in range(len(cases)):
#     coeffs = np.polyfit(prof[cases[i]][2,7:14], prof[cases[i]][0,7:14], deg=1)  # degree=1 → linear fit
#     slopes[i], intercepts[i] = coeffs
    
# del prof

# sl1 = slopes[(abs(slopes) < 400)]
# sl2 = slopes[(abs(slopes) > 400)]
# median_slope = np.median(slopes)

import statsmodels.api as sm

x_YB = np.tile(prof['Flat'][3, start:end], 9)
X = sm.add_constant(x_YB)

ols_model = sm.OLS(y_TKE, X).fit()

intercept = ols_model.params[0]
slope = ols_model.params[1]
r2 = ols_model.rsquared

yB_zero = -intercept / slope

cov = ols_model.cov_params()

var_intercept = cov[0, 0]
var_slope = cov[1, 1]
cov_intercept_slope = cov[0, 1]

dyB0_db = -1 / slope
dyB0_dm = intercept / slope**2

var_yB_zero = (
    dyB0_db**2 * var_intercept
    + dyB0_dm**2 * var_slope
    + 2 * dyB0_db * dyB0_dm * cov_intercept_slope
)

err_yB_zero = np.sqrt(var_yB_zero)

#%%Load Giometto's data
import scipy.io

# Load the .mat file
data = scipy.io.loadmat('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/' + \
                        'urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/simulation_G/profiles.mat')

diss = data['ds_tw'].squeeze()[8:]
# prod = data['sp_xy'].squeeze()[8:] + data['spd_xy'].squeeze()[8:] - data['spm2_xy'].squeeze()[7:]
prod = data['sp_tw'].squeeze()[8:]

res_norm = ((prod + diss)/abs(diss))*100
res_norm[(prod < 7e-3)] = 0

#%%Compute anisotropy for Giometto's data

R11 = data['uu_tw'].squeeze()
R22 = data['vv_tw'].squeeze()
R33 = data['ww_tw'].squeeze()
R12 = data['uv_tw'].squeeze()
R13 = data['uw_tw'].squeeze()
R23 = data['vw_tw'].squeeze()

R11d = data['uu_xy'].squeeze() + data['uud_xy'].squeeze()
R22d = data['vv_xy'].squeeze() + data['vvd_xy'].squeeze()
R33d = data['ww_xy'].squeeze() + data['wwd_xy'].squeeze()
R12d = data['uv_xy'].squeeze() + data['uvd_xy'].squeeze()
R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

def Anisotropy1D(R11, R22, R33, R12, R13, R23):
   
    import numpy as np

    # TKE
    e = R11 + R22 + R33
    N = len(R11)

    # Build Reynolds stress tensor (N, 3, 3)
    R_all = np.zeros((N, 3, 3))
    R_all[:, 0, 0] = R11
    R_all[:, 1, 1] = R22
    R_all[:, 2, 2] = R33
    R_all[:, 0, 1] = R_all[:, 1, 0] = R12
    R_all[:, 0, 2] = R_all[:, 2, 0] = R13
    R_all[:, 1, 2] = R_all[:, 2, 1] = R23

    # Avoid division by zero
    e_safe = np.where(e == 0.0, 1e-12, e)

    # Identity matrix
    Id = np.eye(3)

    # Anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3 = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB = C1c + 0.5 * C3c
    yB = C3c * (np.sqrt(3) / 2)

    return xB, yB, lambda3

xB,yB,lambda3 = Anisotropy1D(R11[8:], R22[8:], R33[8:], R12[8:], R13[8:], R23[8:])


#%%Plot


colors = ['green','limegreen','lightgreen','beige','khaki','gold','peachpuff','sandybrown','saddlebrown']
labels = ['g1200','g800','g400','i1200','i800','i400','ATTO','Sinusoidal','Flat']

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(8,5))

for i in range(len(cases)):
    
    prof = np.load(path_to_data + 'ResTKEvsYB_' + cases[i] + '_Q.npy')
    
    axs.plot(prof[3,:], prof[0,:], c=colors[i], label=labels[i])
    axs.fill_between(prof[3,:], np.array(prof[1,:]), np.array(prof[2,:]), alpha=0.1, color=colors[i])
    # axs.fill_between(
    #     prof[2,:],
    #     np.array(prof[0,:]) - np.array(prof[1,:]),
    #     np.array(prof[0,:]) + np.array(prof[1,:]),
    #     alpha=0.1,color=colors[i]
    # )
    
    # x_line = np.linspace(prof[2,7],prof[2,13],100)
    # y_line = slopes[i]*x_line + intercepts[i]
    # axs.plot(x_line,y_line,c='k')

x_line = np.linspace(0.28,0.45,100)
y_line_1 = slope*x_line + intercept
# y_line_2 = np.mean(sl2)*x_line + 200
axs.plot(x_line,y_line_1,c='k',ls='--')
# axs.plot([], [], ' ', label=f'$R^2$ = {r2 :.02f}') 
# axs.plot(x_line,y_line_2,c='k',ls='--')

# axs.scatter(yB[15:100],res_norm[15:100],s=10,c='k')

axs.axhline(0, color='k', linestyle=':')
axs.axvline(0.2875,color='k',linestyle=':')
axs.axvline(0.4375,color='k',linestyle=':')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-\varepsilon}{|\varepsilon|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)
axs.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)

plt.tight_layout()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'TKEres_vs_YB_AllCases_IQR_legend.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()