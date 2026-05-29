import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import detrend
from heapq import nlargest


def Lorentz_Curve(ftopo):
    spec_topo = np.abs(ftopo) ** 2
    total_variance = np.sum(spec_topo) / 2
    sz = np.shape(ftopo)
    nx = sz[0];
    ny = sz[1];
    total_modes = int(np.floor(nx * ny / 2))
    variance_acc = np.zeros(total_modes)
    variance_individual = np.zeros(total_modes)
    percent_modes = np.zeros(total_modes)
    it = 1
    for i in range(1, total_modes):
        valmax = nlargest(2 * i, spec_topo.flatten())
        variance_acc[i] = np.sum(valmax[::2])
        variance_individual[i] = np.sum(valmax[2 * i - 1])
        percent_modes[i] = it / total_modes * 100.0
        # ith_mode, sum_modes = generate_mode(valmax,spec_topo,i)
        it += 1

    variance_acc /= total_variance
    variance_acc *= 100.0

    dvar = np.gradient(variance_acc)
    ddvar = np.gradient(dvar)

    variance_curvature = abs(ddvar) / (1 + abs(dvar) ** 2) ** (3 / 2)
    var_curve_grad = np.gradient(variance_curvature)

    # Choose curvature based off of continuous derivative of the curvature
    var_curve_adj = np.array([])
    max_curve_it = 0
    max_curve_val = 0.0
    box = 4
    for i in range(0, int(np.size(var_curve_grad) / box) - box, box):
        values = var_curve_grad[i:i + box]
        sup = np.max(values)
        infm = np.min(values)
        if sup - infm < .00005:  # .00125-.002 # this value may need to be adjusted
            # Maybe choose this number based off the variance of the signal
            var_curve_adj = np.append(var_curve_adj, values)
            if np.max(values) > max_curve_val:
                max_curve_it = np.argmax(values) + i
                max_curve_val = values[np.argmax(values)]

    curve_max = nlargest(2, variance_curvature)

    optimal_index = np.argmax(variance_curvature)
    optimal_variance = variance_acc[max_curve_it]
    optimal_mode = percent_modes[max_curve_it]

    plt.figure()
    plt.plot(variance_acc, percent_modes)
    plt.plot(optimal_variance, optimal_mode, 'ro')
    plt.ylabel('Percent Coefficients Represented')
    plt.xlabel('Topographic Variance')

    plt.figure
    plt.plot(variance_individual)
    plt.plot(variance_individual, 'ro')
    plt.xlabel('Mode Number')
    plt.ylabel('percent variance contributed')

    return max_curve_it


def phase_similarity(ftopo):
    mode1 = np.zeros(ftopo.shape, dtype=np.complex64)
    mode2 = np.zeros(ftopo.shape, dtype=np.complex64)

    phases = np.imag(ftopo)
    phase_flat = phases.flatten()
    phases_sort = np.sort(phases_flat)

    mask1 = phase_flat > 210.0
    phase_masked = phase_flat[mask1]
    mask2 = phase_masked < 320.0
    phase_masked2 = phase_masked[mask2]

    point1 = list(np.where(phases == phase_masked2[0]))
    point2 = list(np.where(phases == phase_masked2[1]))

    mode1[point1[0], point1[-1]] = ftopo[point1[0], point1[-1]]
    mode2[point2[0], point2[-1]] = ftopo[point2[0], point2[-1]]

    topo_mode1 = np.fft.ifft2(mode1, norm="ortho")
    topo_mode1 = topo_mode1.real
    topo_mode1 -= topo_mode1.min()

    topo_mode2 = np.fft.ifft2(mode2, norm="ortho")
    topo_mode2 = topo_mode2.real
    topo_mode2 -= topo_mode2.min()

    plt.figure()
    plt.pcolormesh(topo_mode1)
    plt.figure()
    plt.pcolormesh(topo_mode2)


def generate_mode(valmax, nmax, spec_topo, ftopo):
    ftopo_acc = np.zeros(spec_topo.shape, dtype=np.complex64)
    ftopo_sin = np.zeros(spec_topo.shape, dtype=np.complex64)
    for inmax in range(np.size(valmax)):
        ind = list(np.where(spec_topo == valmax[inmax]))
        # print(ind)
        if (len(ind[0]) > 1):
            ind[0] = np.array([ind[0][np.mod(inmax, 2)], ])
            ind[1] = np.array([ind[1][np.mod(inmax, 2)], ])

        ftopo_acc[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
        if inmax >= 2 * nmax - 2:
            ftopo_sin[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]

    topo_acc = np.fft.ifft2(ftopo_acc, norm="ortho")
    topo_acc = topo_acc.real
    topo_acc -= topo_acc.min()
    topo_sin = np.fft.ifft2(ftopo_sin, norm="ortho")
    topo_sin = topo_sin.real
    topo_sin -= topo_sin.min()
    return ftopo_sin, ftopo_acc


def generate_mode_w_list(list_of_modes, spec_topo, ftopo):
    gen = np.zeros(spec_topo.shape, dtype=np.complex64)
    for i in range(len(list_of_modes)):
        ind = list(np.where(spec_topo == list_of_modes[i]))
        if (len(ind[0]) > 1):
            ind[0] = np.array([ind[0][np.mod(i, 2)], ])
            ind[1] = np.array([ind[1][np.mod(i, 2)], ])
        gen[ind[0], ind[-1]] = ftopo[ind[0], ind[-1]]
    return gen


def gen_mode_plots(mode_list, spec_topo, ftopo, box = 10, total_box = 240):
    ifig = 0
    fig = plt.figure(ifig, figsize=[10, 10])
    pl = int((total_box/box)/2)
    gs = gridspec.GridSpec(pl, pl)
    gs.update(**sz)
    cmap_topo = cm.get_cmap("rainbow")
    isub = 0
    levels = np.linspace(0, 60, 30)
    for i in range(0, total_box, box):
        print(i)
        group = valmax[i:i + box]
        fgroup_modes = generate_mode_w_list(group, spec_topo, ftopo)
        group_modes = np.fft.ifft2(fgroup_modes, norm="ortho")
        group_modes = group_modes.real
        group_modes -= group_modes.min()
        # plt.figure()
        # plt.pcolormesh(group_modes)
        # plt.colorbar()
        # plt.title('first ' + str(i+10) + ' ith modes')

        ax = plt.subplot(gs[isub])
        ax.set_aspect('equal', adjustable='box')
        cax = ax.contourf(x, y, group_modes, levels, cmap=cmap_topo, extend='both')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        isub += 1

    cbar_ax = fig.add_axes([0.7, 0.1, 0.02, 0.3])
    cbar = plt.colorbar(cax,
                        ticks=np.arange(levels[0], levels[-1], 10), cax=cbar_ax)
    cbar.ax.set_xlabel(r'hgt (m)')
    cbar.ax.xaxis.set_label_position('top')


def organize_by_wave_mag(ftopo):
    K_mag = np.array([])
    Pow = np.array([])
    indexes = []
    spec_topo = np.abs(ftopo) ** 2
    for i in range(np.shape(ftopo)[0]):
        for j in range(np.shape(ftopo)[0]):
            if ftopo[i, j] * np.conj(ftopo[i, j]) != 0.0:
                K_mag = np.append(K_mag, np.sqrt(i ^ 2 + j ^ 2))
                Pow = np.append(Pow, spec_topo[i, j])  # ftopo[i,j]*np.conj(ftopo[i,j]))
                indexes.append((i, j))

    plt.figure()
    plt.plot(K_mag, Pow, 'bo')
    plt.xlabel('Wave Number Magnitude')
    plt.ylabel('Spectra Inner Product')
    plt.title('minimum required |k| Organization')

    organize = []
    for i in range(len(Pow)):
        if K_mag[i] > .95 and K_mag[i] < 1.05:
            organize.append(indexes[i])

    mode1 = generate_mode_w_list([spec_topo[organize[0][0], organize[0][1]]], spec_topo, ftopo)
    mode2 = generate_mode_w_list([spec_topo[organize[1][0], organize[1][1]]], spec_topo, ftopo)

    x, y = zip(*indexes)

    plt.scatter(x, y, c=Pow)
    plt.tight_layout()