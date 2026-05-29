import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from scipy.io import savemat
import os

def read_cbd(file_path):
    '''
    read_cbd: reads the cbd file data
    Arguments:
    file_path-- path where the file is located
    Return--
    dic-- data_matrix: 3-D matrix of size (N_x, N_y, N_z)
          x-coord: 1-D array of  x-coordinates size (N_x,1)
          y-coord: 1-D array of  y-coordinates size (N_y,1)
          z-coord: 1-D array of  z-coordinates size (N_z,1)
          diagonal_points: end points of the main diagonal
    '''
    f = open(file_path, "rb")
    id = np.fromfile(f, dtype=np.int64, count=1)[0]
    if id == 288230376151834571:
        T = np.float32
    elif id == 576460752303546315:
        T = np.float64
    else:
        raise ValueError("Invalid ID")
    N = tuple(np.fromfile(f, dtype=np.int64, count=3))
    xmin = tuple(np.fromfile(f, dtype=np.float64, count=3))
    xmax = tuple(np.fromfile(f, dtype=np.float64, count=3))
    x1, x2, x3 = (np.fromfile(f, dtype=np.float64, count=n) for n in N)
    data1 = np.fromfile(f, dtype=T, count=np.prod(N)).reshape(N, order="F")
    dic = {"data":data1, "x-coord":x1, "y-coord":x2, "z-coord":x3, "diagonal_points":[xmin,xmax]}
    f.close()
    return dic



def read_var(strr, path, snaps, nx, ny, nz, print_rate = 50):
    var = np.zeros([nx, ny, nz, snaps])

    for i in range(0,snaps):
        if i < 10:
            #if os.path.exists(path + strr + '-00' + str(i) + '.cbd'):
            vart = read_cbd(path + strr + '-00' + str(i) + '.cbd')
            var[:, :, :, i] = vart['data']

        elif i < 100:
            #if os.path.exists(path + strr + '-0' + str(i) + '.cbd'):
            ut = read_cbd(path + strr + '-0' + str(i) + '.cbd')
            var[:, :, :, i] = vart['data']

        else:
            #if os.path.exists(path + strr + '-' + str(i) + '.cbd'):
            ut = read_cbd(path + strr + '-' + str(i) + '.cbd')
            var[:, :, :, i] = vart['data']


        if i % print_rate == 0:
            print("Reading in " + str(i) + "th snapshot of " + strr)

    return var



def read_variables_inst(path, timesteps, frequency, scalars = False, sgs = False):
    snaps = int(timesteps/frequency)
    print_rate = 10
    vars = dict()

    with open(path + 'immersed-boundary/grid.yaml', 'r') as dims:
        try:
            grid = yaml.safe_load(dims)
        except yaml.YAMLError as exc:
            print(exc)
    nx = grid['grid_points.x']; ny = grid['grid_points.y']; nz = grid['grid_points.z']-1;

    data_type1 = 'instantaneous-fields/'

    u = read_var('u', path+data_type1, snaps, nx, ny, nz)
    v = read_var('v', path+data_type1, snaps, nx, ny, nz)
    w = read_var('w', path+data_type1, snaps, nx, ny, nz)

   # p = read_var('p', path+data_type1, snaps, nx, ny, nz)
    if scalars: pcon = read_var('pcon', path+data_type1, snaps, nx, ny, nz)
    #if sgs: txx = read_var('txx', path+'instantaneous-fields/', snaps, nx, ny, nz)
    #if sgs: txy = read_var('txy', path+'instantaneous-fields/', snaps, nx, ny, nz)
    if sgs: txz = read_var('txz', path+data_type1, snaps, nx, ny, nz)
#    if sgs and scalars: ptxx = read_var('pxx', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: ptxy = read_var('pxy', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#   if sgs and scalars: ptxz = read_var('pxz', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: dep = read_var('dep', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: real_dep = read_var('real_dep', path + 'instantaneous-fields/', snaps, nx, ny, nz)

    vars['u'] = u
    vars['v'] = v
    vars['w'] = w
   # vars['p'] = p
    if scalars: vars['pcon'] = pcon
   # if sgs: vars['txx'] = txx
   # if sgs: vars['txy'] = txy
    if sgs: vars['txz'] = txz
#    if sgs and scalars: vars['ptxx'] = ptxx
#    if sgs and scalars: vars['ptxy'] = ptxy
#    if sgs and scalars: vars['ptxz'] = ptxz
#    if sgs and scalars: vars['dep'] = dep
#    if sgs and scalars: vars['real_dep'] = real_dep

    try:
        grid['wall_dist'] = read_cbd(path +'immersed-boundary/wall-distance.cbd')
    except:
        pass
    try:
        grid['wall_norm_x'] = read_cbd(path +'immersed-boundary/wall-normal-x.cbd')
    except:
        pass
    try:
        grid['wall_norm_y'] = read_cbd(path +'immersed-boundary/wall-normal-y.cbd')
    except:
        pass
    try:
        grid['wall_norm_z'] = read_cbd(path +'immersed-boundary/wall-normal-z.cbd')
    except:
        pass

    return vars, grid


def read_variables_avg(path, timesteps, frequency, scalars = False, sgs = False):
    snaps = int(timesteps/frequency)
    print_rate = 10
    vars = dict()

    with open(path + 'immersed-boundary/grid.yaml', 'r') as dims:
        try:
            grid = yaml.safe_load(dims)
        except yaml.YAMLError as exc:
            print(exc)
    nx = grid['grid_points.x']; ny = grid['grid_points.y']; nz = grid['grid_points.z']-1;

    data_type1 = 'time-averaged-fields/'

    u = read_var('u', path+data_type1, snaps, nx, ny, nz)
    v = read_var('v', path+data_type1, snaps, nx, ny, nz)
    w = read_var('w', path+data_type1, snaps, nx, ny, nz)
    uw = read_var('uw', path+data_type1, snaps, nx, ny, nz)
    uu = read_var('uu', path + data_type1, snaps, nx, ny, nz)
    ww = read_var('ww', path + data_type1, snaps, nx, ny, nz)
    if scalars: pcon = read_var('pcon', path+data_type1, snaps, nx, ny, nz)
    if scalars: pw = read_var('pw', path + data_type1, snaps, nx, ny, nz)
    if scalars: pp = read_var('pp', path + data_type1, snaps, nx, ny, nz)
    # if scalars: ptsz = read_var('ptxz', path + data_type1, snaps, nx, ny, nz)
    # if scalars: ptsy = read_var('ptxy', path + data_type1, snaps, nx, ny, nz)
    # if scalars: ptsx = read_var('ptxx', path + data_type1, snaps, nx, ny, nz)
#    if sgs: txx = read_var('txx', path+'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs: txy = read_var('txy', path+'instantaneous-fields/', snaps, nx, ny, nz)
    if sgs: txz = read_var('txz', path+data_type1, snaps, nx, ny, nz)
#    if sgs and scalars: ptxx = read_var('pxx', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: ptxy = read_var('pxy', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#   if sgs and scalars: ptxz = read_var('pxz', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: dep = read_var('dep', path + 'instantaneous-fields/', snaps, nx, ny, nz)
#    if sgs and scalars: real_dep = read_var('real_dep', path + 'instantaneous-fields/', snaps, nx, ny, nz)


    vars['u'] = u
    vars['v'] = v
    vars['w'] = w
    vars['uw'] = uw
    vars['uu'] = uu
    vars['ww'] = ww

#    vars['p'] = p
    if scalars: vars['pp'] = pp
    if scalars: vars['pcon'] = pcon
    if scalars: vars['pw'] = pw
    # if scalars: vars['ptsz'] = ptxz
    # if scalars: vars['ptsy'] = ptxy
    # if scalars: vars['ptsx'] = ptxx
#    if sgs: vars['txx'] = txx
#    if sgs: vars['txy'] = txy
    if sgs: vars['txz'] = txz
#    if sgs and scalars: vars['ptxx'] = ptxx
#    if sgs and scalars: vars['ptxy'] = ptxy
#    if sgs and scalars: vars['ptxz'] = ptxz
#    if sgs and scalars: vars['dep'] = dep
#    if sgs and scalars: vars['real_dep'] = real_dep

    try:
        grid['wall_dist'] = read_cbd(path +'immersed-boundary/wall-distance.cbd')
    except:
        pass
    try:
        grid['wall_norm_x'] = read_cbd(path +'immersed-boundary/wall-normal-x.cbd')
    except:
        pass
    try:
        grid['wall_norm_y'] = read_cbd(path +'immersed-boundary/wall-normal-y.cbd')
    except:
        pass
    try:
        grid['wall_norm_z'] = read_cbd(path +'immersed-boundary/wall-normal-z.cbd')
    except:
        pass
    try:
        grid['phi'] = read_cbd(path + 'immersed-boundary/phi.cbd')
    except:
        pass
    try:
        grid['wall_norm_z'] = read_cbd(path +'immersed-boundary/wall-normal-z.cbd')
    except:
        pass

    return vars, grid


def plot_1d(x,y,title):
    plt.figure(dpi = 100)
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.title(title)


def pcolor(x,y,z,title):
    X,Y = np.meshgrid(x,y)
    plt.figure(dpi = 100)
    plt.contourf(X,Y,z)
    plt.title(title)


def build_intf(phi, dz):
    # Copied into matlab from Marco Giometto build_intf function
    nx, ny, nz = np.shape(phi);
    intf = np.zeros([nx, ny]);
    iintf = np.zeros([nx, ny]);
    init = 0;
    for j in range(1, ny):
        for i in range(1,nx):
            for k in range(1,nz - 1):
                if phi[i, j, k] * phi[i, j, k + 1] <= 0 and  init == 0:
                    intf[i, j] = (k - 1) * dz - phi[i, j, k];
                    iintf[i, j] = k;
                    init = 1;
        init = 0
    return intf, iintf


def avg_4d(var):
    return np.squeeze(np.mean(np.mean(np.mean(var[:,:,:,:],axis = 3, keepdims= True ),axis = 0, keepdims= True ), axis = 1, keepdims= True ))


def avg_3d(var):
    return np.squeeze(np.mean(np.mean(var[:,:,:],axis = 0, keepdims= True ), axis = 1, keepdims= True ))


def avg_time(var,I1,I2):
    return np.squeeze(np.mean(var[:,:,:,I1:I2], axis = 3, keepdims= True))


def dispersive_flux(ui,uj):
    uiz = avg_4d(ui); ujz = avg_4d(uj)
    return np.squeeze(np.mean(np.mean(np.multiply(avg_time(ui,0, np.shape(ui)[3]),avg_time(uj,0, np.shape(ui)[3])), axis = 0 , keepdims = True), axis = 1, keepdims = True)) \
           -  np.multiply(uiz,ujz) 


def reynolds_stress(ui,uj):
    # Rij = uiuj - ui * uj
    uiuj = np.multiply(ui,uj)
    uiavg = avg_time(ui, 0, np.shape(ui)[3])
    ujavg = avg_time(uj, 0, np.shape(uj)[3])
    uiujavg = avg_time(uiuj, 0, np.shape(uiuj)[3])
    return uiujavg - np.multiply(uiavg,ujavg)


def shear_stress(ui, uj, tij):
    Rij = reynolds_stress(ui,uj)
    disp = dispersive_flux(ui,uj)
    tau = tij
    total = -avg_4d(tau) + avg_3d(Rij) + disp
    stresses = dict()
    stresses['Rij'] = Rij; stresses['disp'] = disp
    stresses['tau'] = tau; stresses['total'] = total
    return stresses

def plot_ke(file_path, sim):
    try:
        ke = pd.read_csv(file_path + sim + 'output/ke.txt')
        plt.figure()
        plt.plot(ke)
        plt.title('ke')
        return ke
    except:
        print('no ke outputted')
        return np.nan




def main_compare():
    ## Establish File Path
    sim = 'IBM_32x32_const_profile_w_surface_concentration/'
    sim2 = 'test2/'
    file_path = '/scratch/general/lustre/u0851921/'
    output_loc = 'output/instantaneous-fields/'
    path = file_path+sim+output_loc
    path2 = file_path+sim2+output_loc
    timesteps = 10000
    frequency = 200
    plots = True

    ke = pd.read_csv(file_path + sim + 'output/ke.txt')
    ke2 = pd.read_csv(file_path + sim2 + 'output/ke.txt')

    vara, grid = read_variables(path,timesteps,frequency)
    vars2, grid2 = read_variables(path2,timesteps,frequency)

    uavg = avg_4d(u)
    pconavg = avg_4d(pcon)
    Reynolds_Stress = avg_3d(compute_stress(u,w))

    uavg2 = avg_z(u2)
    pconavg2 = avg_z(pcon2)

    if plots:
        z = np.linspace(0,grid['domain_size.z'],grid['grid_points.x'])
        z2 = np.linspace(0, grid2['domain_size.z'], grid2['grid_points.x'])

        plt.figure()
        plt.plot(uavg,z)
        plt.scatter(uavg,z)
        plt.plot(uavg2,z2)
        plt.title('average u')


        plt.figure()
        plt.plot(pconavg, z)
        plt.scatter(pconavg, z)
        plt.plot(pconavg2,z2)
        plt.title('average pcon')

        plt.figure()
        plt.plot(ke)
        plt.plot(ke2)
        plt.title('ke')

        plt.show()

    # scipy.io.savemat('u.mat', mydict)


def main(sim):
    ## Establish File Path

    file_path = '/scratch/general/lustre/u0851921/'
    output_loc = 'output/' #instantaneous-fields/'
    path = file_path + sim + output_loc

    with open(file_path + sim + 'input/config.yaml', 'r') as dims:
        try:
            config = yaml.safe_load(dims)
        except yaml.YAMLError as exc:
            print(exc)

    dt = config['time_step_size']
    u_star = config['ic.log_law.u_star']
    total_timesteps = config['number_of_time_steps']
    instantaneous_flag = config['output.instantaneous.active']
    try :
        averaged_flag = config['output.time_average.active']
        if averaged_flag == 'T':
            timesteps_skipped  = config['output.time_average.skip']
            frequency = config['output.time_average.length']
        else:
            timesteps_skipped = config['output.instantaneous.skip']
            frequency = config['output.instantaneous.frequency']
    except:
        timesteps_skipped = config['output.instantaneous.skip']
        frequency = config['output.instantaneous.frequency']
    timesteps = total_timesteps - timesteps_skipped
    SCALARS = True if config['particles.active'] == 'T' else False
    IBM = True if config['bc.lower.type'] == 'IBM_DFA' else False
    SGS = True if config['output.instantaneous.write_stress_tensor'] == 'T' else False
    try:
        INST = False if config['output.time_average.active'] == 'T' else True
    except:
        INST = True
    params = dict()
    params['dt'] = dt; params['total_timesteps'] = total_timesteps; params['skip'] = timesteps_skipped
    params['p_cnt3'] = frequency; params['PCON_flag'] = SCALARS; params['IBM'] = IBM; params['SGS'] = SGS
    params['u_star'] = u_star;


    plots = False

    print('Simulation ran for ', dt*total_timesteps/(60.0*60.0), ' hours')

    ke = plot_ke(file_path, sim)

    if INST:
        if not(SGS) and not(SCALARS):
            vars, grid = read_variables_inst(path, timesteps, frequency)
        elif not(SGS) and SCALARS:
            vars, grid = read_variables_inst(path, timesteps, frequency, scalars = True)
        elif not (SCALARS) and SGS:
            vars, grid = read_variables_inst(path, timesteps, frequency, sgs = True)
        else:
            vars, grid = read_variables_inst(path, timesteps, frequency, sgs = True, scalars = True)
    else:
        if not(SGS) and not(SCALARS):
            vars, grid = read_variables_avg(path, timesteps, frequency)
        elif not(SGS) and SCALARS:
            vars, grid = read_variables_avg(path, timesteps, frequency, scalars = True)
        elif not (SCALARS) and SGS:
            vars, grid = read_variables_avg(path, timesteps, frequency, sgs = True)
        else:
            vars, grid = read_variables_avg(path, timesteps, frequency, sgs = True, scalars = True)
        vars['ke'] = ke
        z = grid['wall_norm_z']['z-coord'];
        dz = grid['domain_size.z'] / grid['grid_points.z']
        x = grid['wall_norm_x']['x-coord'];
        dx = grid['domain_size.x'] / grid['grid_points.x']
        y = grid['wall_norm_z']['y-coord'];
        dy = grid['domain_size.y'] / grid['grid_points.y']
        if IBM: params['phi'] = grid['phi']

        params['x'] = x; params['y'] = y; params['z'] = z
        params['dx'] = dx; params['dy'] = dy; params['dz'] = dz

        vars['params'] = params

        savemat(path+'vars.mat', vars)
        print('Move to matlab for analysis or convert script for averaged variables')
        return



    z = grid['wall_norm_z']['z-coord']; dz = grid['domain_size.z']/grid['grid_points.z']
    x = grid['wall_norm_x']['x-coord']; dx = grid['domain_size.x']/grid['grid_points.x']
    y = grid['wall_norm_z']['y-coord']; dy = grid['domain_size.y']/grid['grid_points.y']

    uavg = avg_4d(vars['u'])
    vavg = avg_4d(vars['v'])
    wavg = avg_4d(vars['w'])


    if IBM: intf, iintf = build_intf(grid['wall_dist']['data'], dz)
    if SCALARS: pconavg = avg_4d(vars['pcon'])
    if SGS: shear = shear_stress(vars['u'], vars['w'], vars['txz'])


    if plots:

        if not(IBM): plot_1d(uavg,z,'average u')
        if IBM: plot_1d(uavg[int(np.max(iintf)):],z[int(np.max(iintf)):],'average uavg')
        plot_1d(wavg,z,'average w')
        plot_1d(vavg,z,'average v')
        if SCALARS and IBM: plot_1d(pconavg[int(np.max(iintf)):],z[int(np.max(iintf)):],'average pcon')
        if SCALARS and not(IBM): plot_1d(pconavg, z, 'average pcon')
        if SGS:
            plot_1d(-shear['total'], z, 'Total Shear Stress')
            plt.plot(-avg_3d(shear['Rij']),z, 'r-')
            plt.plot(-shear['disp'],z,'g-')
            plt.plot(avg_4d(shear['tau']),z)
            if IBM: plt.hlines(np.max(intf), 0, np.max(-shear['total']),'k')
            #plt.legend(['total','Rij','disp','tau'])

        X,Y = np.meshgrid(x,y)
        Z = np.squeeze(avg_time(vars['pcon'],0,int(timesteps/frequency))[:,16,:]).T

        plt.figure()
        plt.contourf(X,Y,Z,30)
        # print('For Running In Debug')  # Should just use Jupyter
        # plt.show()
    return vars, grid, IBM, SCALARS, SGS
sim =  'ATTO_sims/ATTO1_Columbia_IBM_r_guest1/'
main(sim)
# plt.hlines(np.max(intf),0,35)
# scipy.io.savemat('u.mat', mydict)
