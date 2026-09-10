#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created from Paper1_varPDF.py to plot normalized variance PDFs for all cases.

@author: u1450851
"""


# Libraries and Functions
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions import uvpnode2wnode, wnode2uvpnode


# %% Import data

cases = ['Gap_12_9mps', 'Gap_8_9mps', 'Gap_4_9mps', 'Patch_12_9mps', 'Patch_8_9mps', 'Patch_4_9mps', 'ATTO', 'Sinusoidal', 'Flat']
# cases = ['Flat']
labels = ['g1200', 'g800', 'g400', 'i1200', 'i800', 'i400', 'ATTO', 'Sinusoidal', 'Flat']
colors = ['green', 'limegreen', 'lightgreen', 'beige', 'khaki', 'gold', 'peachpuff', 'sandybrown', 'saddlebrown']

case_labels = dict(zip(cases, labels))
case_colors = dict(zip(cases, colors))
gap_patch_cases = ['Gap_12_9mps', 'Gap_8_9mps', 'Gap_4_9mps', 'Patch_12_9mps', 'Patch_8_9mps', 'Patch_4_9mps'] #set(cases[:6])

topo_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
gap_patch_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'

data = dict()
anisotropy = dict()
dist = dict()
TKE = dict()

for case_name in cases:
    if case_name in gap_patch_cases:
        data[case_name] = xr.open_dataarray(gap_patch_path + case_name + '/Data_Momentum_4TKE.nc').data
        anisotropy[case_name] = xr.open_dataarray(gap_patch_path + case_name + '/anisotropy.nc').data
        TKE[case_name] = xr.open_dataarray(gap_patch_path + case_name + '/TKE_terms.nc').data
    else:
        data[case_name] = xr.open_dataarray(topo_path + case_name + '/dataTKE.nc').data
        anisotropy[case_name] = xr.open_dataarray(topo_path + case_name + '/anisotropy.nc').data
        dist[case_name] = xr.open_dataarray(topo_path + case_name + '/dist.nc').data
        TKE[case_name] = xr.open_dataarray(topo_path + case_name + '/TKE_terms.nc').data


# %% Compute Reynolds stresses

uu = dict()
vv = dict()
ww = dict()
uv = dict()
uw = dict()
vw = dict()
tke = dict()

for case_name in cases:
    if case_name in gap_patch_cases:
        uu[case_name] = data[case_name][:, :, :, 4] - data[case_name][:, :, :, 0]*data[case_name][:, :, :, 0] - data[case_name][:, :, :, 19]
        vv[case_name] = data[case_name][:, :, :, 5] - data[case_name][:, :, :, 1]*data[case_name][:, :, :, 1] - data[case_name][:, :, :, 20]
        ww[case_name] = wnode2uvpnode(data[case_name][:, :, :, 6] - data[case_name][:, :, :, 2]*data[case_name][:, :, :, 2]) - data[case_name][:, :, :, 21]
        uv[case_name] = data[case_name][:, :, :, 7] - data[case_name][:, :, :, 0]*data[case_name][:, :, :, 1] - data[case_name][:, :, :, 22]
        uw[case_name] = wnode2uvpnode(data[case_name][:, :, :, 8] - uvpnode2wnode(data[case_name][:, :, :, 0])*data[case_name][:, :, :, 2]
            - data[case_name][:, :, :, 23])
        vw[case_name] = wnode2uvpnode(data[case_name][:, :, :, 9] - uvpnode2wnode(data[case_name][:, :, :, 1])*data[case_name][:, :, :, 2]
            - data[case_name][:, :, :, 24])
    else:
        uu[case_name] = data[case_name][:, :, :, 4] - data[case_name][:, :, :, 0]*data[case_name][:, :, :, 0] - data[case_name][:, :, :, 19]
        vv[case_name] = data[case_name][:, :, :, 5] - data[case_name][:, :, :, 1]*data[case_name][:, :, :, 1] - data[case_name][:, :, :, 20]
        ww[case_name] = data[case_name][:, :, :, 6] - data[case_name][:, :, :, 2]*data[case_name][:, :, :, 2] - data[case_name][:, :, :, 21]
        uv[case_name] = data[case_name][:, :, :, 7] - data[case_name][:, :, :, 0]*data[case_name][:, :, :, 1] - data[case_name][:, :, :, 22]
        uw[case_name] = data[case_name][:, :, :, 8] - data[case_name][:, :, :, 0]*data[case_name][:, :, :, 2] - data[case_name][:, :, :, 23]
        vw[case_name] = data[case_name][:, :, :, 9] - data[case_name][:, :, :, 1]*data[case_name][:, :, :, 2] - data[case_name][:, :, :, 24]

    tke[case_name] = 0.5*(uu[case_name] + vv[case_name] + ww[case_name])


# %% Residual normalization

ResNorm = dict()

for case_name in cases:
    if case_name in gap_patch_cases:
        tmpDIS = copy.deepcopy(TKE[case_name][:, :, :, 11])
        tmpRES = copy.deepcopy(TKE[case_name][:, :, :, -1] + TKE[case_name][:, :, :, 11])
        threshold = 0.05*np.nanmax(np.nanmedian(tmpRES, axis=(0, 1)))
        tmpRES[abs(tmpRES) < threshold] = 0
    else:
        tmpRES = copy.deepcopy(TKE[case_name][:, :, :, 14] - TKE[case_name][:, :, :, 11])
        tmpDIS = copy.deepcopy(TKE[case_name][:, :, :, 11])
        val = np.nanmedian(tmpRES[(dist[case_name][:, :, :, 0] > 38) & (dist[case_name][:, :, :, 0] < 44)])
        tmpRES[abs(tmpRES) < 0.05*val] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        ResNorm[case_name] = np.where(tmpDIS != 0, tmpRES/np.abs(tmpDIS)*100, np.nan)


# %% Plot the normalized PDF

med_cR = []
med_yB = []

tollerance = 10

layout = [['a)', 'b)', 'c)'],
          ['d)', 'e)', 'f)']]

fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained', figsize=(9, 6), sharey=False)

for label, ax in axs_dict.items():
    ax.text(0.0, 1.0, label, transform=(ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)), fontsize=15, va='bottom', fontfamily='serif')

axs = np.array([[axs_dict[label] for label in row] for row in layout])
rng = np.random.default_rng()
nsamples = 200000

def filtered_values(case_name, var_e):
    if case_name in gap_patch_cases:
        values = var_e[:, :, 10:][abs(ResNorm[case_name])[:, :, 10:] < tollerance]
    else:
        mask = ((abs(ResNorm[case_name]) < tollerance) & (dist[case_name][:, :, :, 0] > 1*40) & (dist[case_name][:, :, :, 0] < 40*15))
        values = var_e[mask]

    values = np.asarray(values)
    return values[np.isfinite(values)]

def sample_values(values):
    sample_size = min(nsamples, len(values))
    if sample_size < 2:
        return None
    return rng.choice(values, size=sample_size, replace=False)

def plot_pdf(ax, samples, color, label=None):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        return np.nan

    kde = gaussian_kde(samples)
    x_pdf = np.linspace(np.nanmin(samples), np.nanmax(samples), 1000)
    pdf = kde(x_pdf)
    ax.plot(x_pdf, pdf, c=color, label=label)

    median_val = np.nanmedian(samples)
    y_median = np.interp(median_val, x_pdf, pdf)
    ax.plot([median_val, median_val], [0, y_median], c=color, ls='-')
    return median_val


for case_name in cases:
    uu_e = uu[case_name]/tke[case_name]
    vv_e = vv[case_name]/tke[case_name]
    ww_e = ww[case_name]/tke[case_name]
    uw_e = abs(uw[case_name]/tke[case_name])

    uu_samp = sample_values(filtered_values(case_name, uu_e))
    vv_samp = sample_values(filtered_values(case_name, vv_e))
    ww_samp = sample_values(filtered_values(case_name, ww_e))
    uw_samp = sample_values(filtered_values(case_name, uw_e))

    if uu_samp is None or vv_samp is None or ww_samp is None or uw_samp is None:
        print(f"Skipping {case_name}: not enough finite filtered samples.")
        continue

    cR = 1/(1 - (3/2)*ww_samp)
    yB = (np.sqrt(3)/2)*(1 - 1/cR)

    color = case_colors[case_name]
    plot_label = case_labels[case_name]

    median_val = plot_pdf(axs[0, 0], uu_samp, color, label=plot_label)
    print(f"Median uu/e for {case_name} is: {median_val}")

    median_val = plot_pdf(axs[0, 1], vv_samp, color)
    print(f"Median vv/e for {case_name} is: {median_val}")

    median_val = plot_pdf(axs[0, 2], ww_samp, color)
    print(f"Median ww/e for {case_name} is: {median_val}")

    median_val = plot_pdf(axs[1, 0], uw_samp, color)
    print(f"Median |uw|/e for {case_name} is: {median_val}")

    median_val = plot_pdf(axs[1, 1], cR, color)
    med_cR.append(median_val)
    print(f"Median cR for {case_name} is: {median_val}")

    median_val = plot_pdf(axs[1, 2], yB, color)
    med_yB.append(median_val)
    print(f"Median yB for {case_name} is: {median_val}")

axs[0, 0].set_xlabel(r"$\frac{\overline{u'u'}}{\overline{e}}$", fontsize=16)
axs[0, 1].set_xlabel(r"$\frac{\overline{v'v'}}{\overline{e}}$", fontsize=16)
axs[0, 2].set_xlabel(r"$\frac{\overline{w'w'}}{\overline{e}}$", fontsize=16)
axs[1, 0].set_xlabel(r"$\frac{|\overline{u'w'}|}{\overline{e}}$", fontsize=16)
axs[1, 1].set_xlabel(r"$c_R$", fontsize=16)
axs[1, 2].set_xlabel(r"$y_B^{approx}$", fontsize=16)

axs[0, 0].set_ylabel(r"PDF", fontsize=15)
axs[1, 0].set_ylabel(r"PDF", fontsize=15)

axs[0, 0].set_xlim(0.65, 1.25)
axs[0, 1].set_xlim(0.35, 1)
axs[0, 2].set_xlim(0.25, 0.65)
axs[1, 0].set_xlim(0.1, 0.4)
axs[1, 1].set_xlim(1, 5)
axs[1, 2].set_xlim(0.25, 0.8)

for i in range(len(axs[0])):
    axs[0, i].set_ylim(0)
    axs[1, i].set_ylim(0)
    axs[0, i].tick_params(axis='both', which='major', labelsize=12)
    axs[1, i].tick_params(axis='both', which='major', labelsize=12)

# Retrieve legend entries from the first subplot
handles, labels = axs[0, 0].get_legend_handles_labels()

# Reserve space on the right for the legend
fig.get_layout_engine().set(rect=(0, 0, 0.82, 1))

# Center the legend vertically between the two rows
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.83, 0.5), frameon=False, fontsize=9)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Variances_RottaModel_cRyB_AllCases.png', dpi=300, edgecolor='white', facecolor='white', bbox_inches="tight")

plt.show()

#%%Compute mean of the median cR and yB

print("Mean of the median cR: ", np.mean(np.array(med_cR)), "+-", np.std(np.array(med_cR)))
print("Mean of the median yB: ", np.mean(np.array(med_yB)), "+-", np.std(np.array(med_yB)))








































