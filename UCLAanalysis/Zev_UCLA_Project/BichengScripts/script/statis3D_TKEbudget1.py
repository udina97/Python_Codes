#!/usr/bin/env python3

"""
Program : statis3d
The 3D statistics of simulation including mean velocity and fluctuation.
"""

### Histories:
### 2019/03/06 -- Bicheng Chen (chabby@ucla.edu) -- First create



## Import Modules
import numpy as np
import lespy as lp
import os.path
import time
import sys



## User-specified Variables
# File
fmt_lespath = '/data/2/bzc/LES/amazon_3D/{case:s}'
case = 'amazon_canopy_real'
fmt_out = './data/statistics3D_{case:s}_tt{tts:08d}-{tte:08d}.npz'
fmt_tau = 'tau_jt{tt:08d}.sbin'
fmt_can = 'canforce_jt{tt:08d}.sbin'
fmt_dissip = 'dissip_jt{tt:08d}.sbin'

# Data range
#tt_all = (144100, 216000+1, 100)
tt_all = (144100, 324000+1, 100)
#tt_all = (144100, 180000+1, 100)
#tt_all = (144100, 145000+1, 100)
items = ('u', 'v', 'w', 'p','u2', 'v2', 'w2', 'uv', 'uw', 'vw', 'p2',
  'pu', 'pv', 'pw', 'pdudx', 'pdvdy', 'pdwdz', 'txx', 'txy', 'txz',
  'tyy', 'tyz', 'tzz', 'u3', 'v3', 'w3', 'u2v', 'u2w', 'uv2', 'v2w',
  'uw2', 'vw2', 'uvw',
  'utxx', 'utxy', 'utxz', 'vtxy', 'vtyy', 'vtyz',
  'wtxz', 'wtyz', 'wtzz', 'dissip', 'Fcx', 'Fcy', 'Fcz', 'uFcx', 'vFcy', 'wFcz')




## Functions
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-1, norm="ortho")
  dphidx_c = complex(0, 1) * wn[np.newaxis, :] * phi_c
  dphidx_c[:, :, -1] = 0
  return np.fft.irfft(dphidx_c, axis=-1, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=-2, norm="ortho")
  dphidy_c = complex(0, 1) * wn[:, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=-2, norm="ortho")

def get_dphidz(phi, dm):
  dphidz = np.zeros(phi.shape)
  dphidz[:-1, :] = (phi[1:, :]-phi[:-1, :]) / (dm.dz/dm.zi)
  dphidz[-1, :] = dphidz[-2, :]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[0, :, :] = 0
  phi_h[1:, :, :] = 0.5*(phi_c[:-1, :, :] + phi_c[1:, :, :])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:-1, :, :] = 0.5*(phi_h[:-1, :, :]+phi_h[1:, :, :])
  phi_c[-1, :, :] = phi_c[-2, :, :]
  return phi_c

def padding(phi):
  shape = phi.shape
  ns = len(shape)
  shapec_big = list(shape)
  shapec_big[-1] = (shapec_big[-1] * 3 // 2 ) // 2 + 1
  shapec_big[-2] = shapec_big[-2] * 3 // 2
  phic = np.fft.rfft2(phi, axes=(-2, -1), norm="ortho")
  phic_big = np.zeros(shapec_big, dtype=np.complex)
  if ns ==3:
    phic_big[:, :(shape[-2]+1)//2, :shape[-1]//2+1] =\
      phic[:, :(shape[-2]+1)//2, :]
    phic_big[:, -shape[-2]//2:, :shape[-1]//2+1] =\
      phic[:, -shape[-2]//2:, :]
  elif ns ==2:
    phic_big[:(shape[-2]+1)//2, :shape[-1]//2+1] =\
      phic[:(shape[-2]+1)//2, :]
    phic_big[-shape[-2]//2:, :shape[-1]//2+1] =\
      phic[-shape[-2]//2:, :]
  phi_big = np.fft.irfft2(phic_big, axes=(-2, -1), norm="ortho")
  phi_big *= np.sqrt(phi_big.shape[-2]*phi_big.shape[-1]
    / (shape[-2]*shape[-1]))

  return(phi_big)

def unpadding(phi_big, ny, nx):
  shape_big = phi_big.shape
  ns = len(shape_big)
  shapec = list(shape_big)
  shapec[-1] = nx//2 +1
  shapec[-2] = ny
  phic = np.zeros(shapec, dtype=np.complex)
  phic_big = np.fft.rfft2(phi_big, axes=(-2, -1), norm="ortho")
  if ns == 3:
    phic[:, :(ny+1)//2, :] = phic_big[:, :(ny+1)//2, :nx//2+1]
    phic[:, -ny//2:, :] = phic_big[:, -ny//2:, :nx//2+1]
  elif ns == 2:
    phic[(ny+1)//2, :] = phic_big[(ny+1)//2, :nx//2+1]
    phic[-ny//2:, :] = phic_big[-ny//2:, :nx//2+1]
  phi = np.fft.irfft2(phic, axes=(-2, -1), norm="ortho")
  phi *= np.sqrt(ny*nx/(shape_big[-2]*shape_big[-1]))

  return(phi)



## Initialize the reading process
# Read setup parameters of the simulation
start_time = time.time()
print('#'*80)
print('   Reading the setup parameters of the simulation...')
path_les = fmt_lespath.format(case=case)
param = lp.lesParam.lesClass.param(path_les)
dm = param.domain
dm.show()

# Define the variables and coordinates
print('   Initializing the data reading...')
ntt = len(range(*tt_all))
z, y, x = lp.domain.dmFun.xyzcoord(dm)
ibm = lp.domain.dmClass.immersedbdy(param)
z_tpg = ibm.z_phi0 * dm.zi
y_tpg, x_tpg = lp.domain.dmFun.xycoord(dm)

# Data
data = {item: 0 for item in items}
wn_x = 2*np.pi*np.fft.rfftfreq(dm.nx, dm.dx/dm.zi)
wn_y = 2*np.pi*np.fft.rfftfreq(dm.ny, dm.dy/dm.zi)

## Calculate the mean
print('#'*80)
for itt, tt in enumerate(range(*tt_all)):
  print('  Step {itt:4d} over {ntt:4d}.'.format(itt=itt+1, ntt=ntt), flush=True)
  # Velocity
  vel = lp.flow.flowClass.velocity(param, tt=tt)
  data['u'] += vel.u / ntt
  data['v'] += vel.v / ntt
  data['w'] += vel.w / ntt

  # Press
  press = lp.flow.flowClass.pressure(param, tt=tt)
  data['p'] += press.p / ntt

  # 2-nd order quantities with velocity and pressure (resolved part)
  data['u2'] += vel.u**2 / ntt
  data['v2'] += vel.v**2 / ntt
  data['w2'] += vel.w**2 / ntt
  data['uv'] += vel.u*vel.v / ntt
  u_w = uvpnode2wnode(vel.u)
  data['uw'] += u_w*vel.w / ntt
  v_w = uvpnode2wnode(vel.v)
  data['vw'] += v_w*vel.w / ntt

  data['p2'] += press.p**2 / ntt
  data['pu'] += press.p*vel.u / ntt
  data['pv'] += press.p*vel.v / ntt
  data['pw'] += uvpnode2wnode(press.p)*vel.w / ntt

  dudx_c = get_dphidx(vel.u, wn_x)
  data['pdudx'] += press.p*dudx_c / ntt
  dvdy_c = get_dphidy(vel.v, wn_y)
  data['pdvdy'] += press.p*dvdy_c / ntt
  dwdz_c = get_dphidz(vel.w, dm)
  data['pdwdz'] += press.p*dwdz_c / ntt

  # 2-nd order quantities with velocity and pressure (un-resolved part)
  tau = lp.io.io_instFile.read_3dFile(
    os.path.join(param.path, 'output', fmt_tau.format(tt=tt)),
    nx=dm.nx, ny=dm.ny, nz=dm.nz, nq_read=6)
  data['txx'] += tau[0] / ntt
  data['txy'] += tau[1] / ntt
  data['txz'] += tau[2] / ntt
  data['tyy'] += tau[3] / ntt
  data['tyz'] += tau[4] / ntt
  data['tzz'] += tau[5] / ntt

  # 3-rd order quantities (resolved part)
  w_uv = wnode2uvpnode(vel.w)
  data['u3'] += vel.u**3 / ntt
  data['v3'] += vel.v**3 / ntt
  data['w3'] += vel.w**3 / ntt
  data['u2v'] += vel.u**2*vel.v / ntt
  data['u2w'] += u_w**2*vel.w / ntt
  data['uv2'] += vel.u*vel.v**2 / ntt
  data['v2w'] += v_w**2*vel.w / ntt
  data['uw2'] += u_w*vel.w**2 / ntt
  data['vw2'] += v_w*vel.w**2 / ntt
  data['uvw'] += vel.u*vel.v*w_uv / ntt

  #u_pad = padding(vel.u)
  #v_pad = padding(vel.v)
  #w_pad = padding(w_uv)
  #tke_pad = 0.5*(u_pad**2 + v_pad**2 + w_pad**2)
  #tke = unpadding(tke_pad, vel.u.shape[-2], vel.u.shape[-1])
  #data['tke*u'] += tke*vel.u / ntt
  #data['tke*v'] += tke*vel.v / ntt

  #u_pad = padding(u_w)
  #v_pad = padding(v_w)
  #w_pad = padding(vel.w)
  #tke_pad = 0.5*(u_pad**2 + v_pad**2 + w_pad**2)
  #var = unpadding(tke_pad, vel.u.shape[-2], vel.u.shape[-1])
  #data['tke*w'] += tke*vel.w / ntt

  # 3-rd order quantities (un-resolved part)
  data['utxx'] += vel.u*tau[0] / ntt
  data['utxy'] += vel.u*tau[1] / ntt
  data['utxz'] += u_w*tau[2] / ntt
  data['vtxy'] += vel.v*tau[1] / ntt
  data['vtyy'] += vel.v*tau[3] / ntt
  data['vtyz'] += v_w*tau[4] / ntt
  data['wtxz'] += vel.w*tau[2] / ntt
  data['wtyz'] += vel.w*tau[4] / ntt
  data['wtzz'] += vel.w*uvpnode2wnode(tau[5]) / ntt

  # Dissipation rate
  dissip = lp.io.io_instFile.read_3dFile(
    os.path.join(param.path, 'output', fmt_dissip.format(tt=tt)),
    nx=dm.nx, ny=dm.ny, nz=dm.nz)
  data['dissip'] += dissip / ntt

  # Canopy related
  fc = lp.io.io_instFile.read_3dFile(
    os.path.join(param.path, 'output', fmt_can.format(tt=tt)),
    nx=dm.nx, ny=dm.ny, nz=dm.nz, nq_read=6)
  data['Fcx'] += fc[0] / ntt
  data['Fcy'] += fc[1] / ntt
  data['Fcz'] += fc[2] / ntt
  data['uFcx'] += fc[3] / ntt
  data['vFcy'] += fc[4] / ntt
  data['wFcz'] += fc[5] / ntt

middle_time = time.time()

## Save the averaged data
# Domain
fn = fmt_out.format(case=case, tts=tt_all[0], tte=tt_all[1]-1)
np.savez(fn, x=x, y=y, z=z, lx=dm.lx, ly=dm.ly, lz=dm.lz, x_tpg=x_tpg,
  y_tpg=y_tpg, z_tpg=z_tpg, data=data)
end_time = time.time()
print("{:f} second for loop".format(middle_time-start_time))
print("{:f} second for all".format(end_time-start_time))
