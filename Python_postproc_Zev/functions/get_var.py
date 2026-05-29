import numpy as np
import matplotlib.pyplot as plt

def build_phi(path,nx,ny,nz,mpiProc):
    phi = np.zeros((nx,ny,nz))
    for i in range(0,mpiProc):
        nzi = int(np.floor(nz/mpiProc))
        inputfilename = path + 'phi.c' + str(i)
        with open(inputfilename, 'rb') as fid:
            data = np.fromfile(fid)
        cvar = np.reshape(data,(nx,ny,nzi))
        phi[:,:,0+nzi*i:nzi+nzi*i] = cvar[:,:,0:nzi]
    return phi

def get_var(path,var,nx,ny,nzTot,nt,mpiproc,avgt):
    global_var = np.zeros((nx,ny,nzTot,avgt))
    nzi = int(np.floor(nzTot/mpiproc))

    for i in range(0,mpiproc):
        inputfilename = path + var + '.c' + str(i)
        with open(inputfilename, 'rb') as fid:
            data = np.fromfile(fid)
        data = data[nx*ny*nzi*(nt-avgt):nx*ny*nzi*nt]
        cvar = np.reshape(data,(nx,ny,nzi,avgt))
        global_var[:,:,0+nzi*i:nzi+nzi*i,:] = cvar[:,:,0:nzi,:]
        #print('done mpi layer ' + str(i))

    #global_var = global_var[:,:,:,nt-avgt+1:nt]
    print('done with '+ var)
    return global_var



def build_intf(phi,dz):
    nx,ny,nz = np.shape(phi)
    intf = np.zeros((nx,ny))
    iintf = np.zeros((nx,ny))
    init = 0
    for j in range(0,ny):
        for i in range(0,nx):
            for k in range(0,nz-1):
                if phi[i,j,k]*phi[i,j,k+1] <= 0.0 and init == 0:
                    intf[i,j] = (k-1)*dz-phi[i,j,k]
                    iintf[i,j] = k
                    init = 1
            init = 0
    return intf, iintf




# sim = 'ATTO1_v1'
# path = '/scratch/general/vast/u0851921/ATTOv2/sim'+sim +'/output/ta1_field/'
# path = '/scratch/general/vast/u0851921/ATTOv2/sim'+sim +'/output/phi_functions/'
# nx = 300
# ny = 300
# nzTot = 270
# nt = 36
# mpiproc = 270
# phi = build_phi(path,nx,ny,nzTot,mpiproc)
# intf, iintf = build_intf(phi,540.0/nzTot)
#
# plt.figure(dpi= 100)
# plt.contourf(intf)
# plt.show()
# var = 'u'
# avgt = 6
# u = get_var(path,var,nx,ny,nzTot,nt,mpiproc,avgt)
# uavg_z = np.mean(np.mean(np.mean(u[:,:,:,1:6],axis=3),axis=1),axis=0)
# plt.figure
# plt.plot(uavg_z)
# plt.show