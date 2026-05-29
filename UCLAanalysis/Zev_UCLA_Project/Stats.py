"""
Program Name: Stats.py
Program purpose: This program computes different flow statistics.

Program Author: Marc Calaf.

Date created: 05 March 2020
Last date modified: 05 March 2020

"""

# ProjectOn_inter(): function that projects the w-node data on uvp-node data
##############################################################################

def ProjectOn_inter(Nx,Ny,Nz,var3D):
    "Calling the ReynoldsStress() function"

    import numpy as np

    var3D_inter = np.zeros((Nx,Ny,Nz),'d',order='F')

    #Projecting the 3D variable onto the uvp nodes.
    for k in range(0,Nz-1):
        var3D_inter[:,:,k]=0.5*(var3D[:,:,k] + var3D[:,:,k+1])

    var3D_inter[:,:,-1] = var3D[:,:,-1]

    return(var3D_inter)

# ProjectOn_uvp(): function that projects the w-node data on uvp-node data
##############################################################################

def ProjectOn_uvp(Nx,Ny,Nz,var3D):
    "Calling the ReynoldsStress() function"

    import numpy as np

    var3D_uvp = np.zeros((Nx,Ny,Nz),'d',order='F')

    #Projecting the 3D variable onto the uvp nodes.
    for k in range(0,Nz-1):
        var3D_uvp[:,:,k]=0.5*(var3D[:,:,k] + var3D[:,:,k+1])

    var3D_uvp[:,:,-1] = var3D[:,:,-1]

    return(var3D_uvp)

# ProjectOn_w(): function that projects the uvp-node data on w-node data
##############################################################################

def ProjectOn_w(Nx,Ny,Nz,var3D):
    "Calling the ReynoldsStress() function"

    import numpy as np

    var3D_w = np.zeros((Nx,Ny,Nz),'d',order='F')
    var3D_w[:,:,0] = 0

    #Projecting the 3D variable onto the w nodes.
    for k in range(1,Nz):
        var3D_w[:,:,k]=0.5*(var3D[:,:,k-1] + var3D[:,:,k])



    return(var3D_w)


# ReynoldsStress(): function that computes the 3D Reynolds Stress
##############################################################################

def ReynoldsStress(Nx,Ny,Nz,avgU,avgV,avgW,avgU2,avgV2,avgW2,avgUV,avgUW,avgVW):
    "Calling the ReynoldsStress() function"

    import numpy as np
    import xarray as xr
    from Stats import ProjectOn_w,ProjectOn_uvp

    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

    # Projection of u & v on w-nodes: -------------------------
    avgU_w = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgV_w = np.zeros((Nx,Ny,Nz),'d',order='F')

    #Computing the projection of the W-related variables:
    for k in range(1,Nz):
        avgU_w[:,:,k]=0.5*(avgU[:,:,k-1] + avgU[:,:,k])
        avgV_w[:,:,k]=0.5*(avgV[:,:,k-1] + avgV[:,:,k])

    avgU_w[:,:,0] = 0
    avgV_w[:,:,0] = 0

    # Projection of w on uvpT-nodes: -------------------------
    avgW_uvpT = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgW2_uvpT = np.zeros((Nx,Ny,Nz),'d',order='F')

    #Computing the projection of the W-related variables:
    for k in range(0,Nz-1):
        avgW_uvpT[:,:,k]=0.5*(avgW[:,:,k] + avgW[:,:,k+1])
        avgW2_uvpT[:,:,k]=0.5*(avgW2[:,:,k] + avgW2[:,:,k+1])

    avgW_uvpT[:,:,-1] = avgW[:,:,-1]
    avgW2_uvpT[:,:,-1] = avgW2[:,:,-1]


    # Computing the different terms
    Rstress[:,:,:,0] = avgU2 - avgU*avgU #Rxx at uvpT node.
    Rstress[:,:,:,1] = avgV2 - avgV*avgV #Ryy at uvpT node.
    Rstress[:,:,:,2] = avgW2_uvpT - avgW_uvpT*avgW_uvpT #Rzz at uvpT node.
    Rstress[:,:,:,3] = avgUV - avgU*avgV #Rxy; Changing the corresponding sign!  at uvpT node.
    Rstress[:,:,:,4] = avgUW - avgU_w*avgW #Rxz; Changing the corresponding sign!  at w node.
    Rstress[:,:,:,5] = avgVW - avgV_w*avgW #Ryz; Changing the corresponding sign!  at w node.

    #(see Albertson's Thesis for reference in which grid points terms are computed)



    return(Rstress)



##############################################################################
##############################################################################
##############################################################################



# DispersiveStress(): function that computes the 3D Reynolds Stress
##############################################################################

def DispersiveStress(Nx,Ny,Nz,avgU,avgV,avgW,avgU2,avgV2,avgW2,avgUV,avgUW,avgVW):
    "Calling the DispersiveStress() function"

    import numpy as np
    import xarray as xr
    from Stats import ProjectOn_w,ProjectOn_uvp

    Dstress = xr.DataArray(np.ones(shape = (Nz,6),order='F'),\
                        dims=('z','variable'), coords = {'variable':['Dxx','Dyy',\
                        'Dzz','Dxy','Dxz','Dyz']})

    avgU_w = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgV_w = np.zeros((Nx,Ny,Nz),'d',order='F')

    #Computing the projection of the W-related variables:
    for k in range(1,Nz):
        avgU_w[:,:,k]=0.5*(avgU[:,:,k-1] + avgU[:,:,k])
        avgV_w[:,:,k]=0.5*(avgV[:,:,k-1] + avgV[:,:,k])

    avgU_w[:,:,0] = 0
    avgV_w[:,:,0] = 0

    dispU_w = avgU_w - np.mean(np.mean(avgU_w,axis = 0),axis=0)
    dispV_w = avgV_w - np.mean(np.mean(avgV_w,axis = 0),axis=0)
    dispU = avgU - np.mean(np.mean(avgU,axis = 0),axis=0)
    dispV = avgV - np.mean(np.mean(avgV,axis = 0),axis=0)
    dispW = avgW - np.mean(np.mean(avgW,axis = 0),axis=0)

    Dstress[:,0] = np.mean(np.mean(np.multiply(dispU,dispU),axis = 0),axis = 0) #Dxx
    Dstress[:,1] = np.mean(np.mean(np.multiply(dispV,dispV),axis = 0),axis = 0) #Dyy
    Dstress[:,2] = np.mean(np.mean(np.multiply(dispW,dispW),axis = 0),axis = 0) #Dzz
    Dstress[:,3] = np.mean(np.mean(np.multiply(dispU,dispV),axis = 0),axis = 0) #Dxy
    Dstress[:,4] = np.mean(np.mean(np.multiply(dispU_w,dispW),axis = 0),axis = 0) #Dxz
    Dstress[:,5] = np.mean(np.mean(np.multiply(dispV_w,dispW),axis = 0),axis = 0) #Dyz

    return(Dstress)

##############################################################################
##############################################################################
##############################################################################

# ReynoldsFlux(): function that computes the 3D Reynolds Stress
##############################################################################

def ReynoldsFlux(Nx,Ny,Nz,avgT,avgT2,avgUT,avgVT,avgWT,avgU,avgV,avgW):
    "Calling the ReynoldsFlux() function"

    import numpy as np
    import xarray as xr
    from Stats import ProjectOn_w,ProjectOn_uvp

    Rflux = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,5),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['FTT','FuT',\
                        'FvT','FwT_uvp','FwT_w']})

    avgW_uvp = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgWT_uvp = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgT_w = np.zeros((Nx,Ny,Nz),'d',order='F')

    avgW_uvp = ProjectOn_uvp(Nx,Ny,Nz,avgW)
    avgWT_uvp = ProjectOn_uvp(Nx,Ny,Nz,avgWT)

    #avgT_w = ProjectOn_w(Nx,Ny,Nz,avgT) --> We can't use this because at w-node, nz=1, T is not zero, only the velocity fields.

    #Computing the projection of the mean temperature on the W-related variables:
    for k in range(1,Nz):
        avgT_w[:,:,k]=0.5*(avgT[:,:,k-1] + avgT[:,:,k])

    avgT_w [:,:,0] = avgT [:,:,0]


    # Computing the different terms:
    Rflux[:,:,:,0] = avgT2 - avgT*avgT         #FTT on uvpT nodes
    Rflux[:,:,:,1] = avgUT - avgU*avgT         #FuT on uvpT nodes
    Rflux[:,:,:,2] = avgVT - avgV*avgT         #FvT on uvpT nodes
    Rflux[:,:,:,3] = avgWT_uvp - avgW_uvp*avgT #FwT  on uvp nodes
    Rflux[:,:,:,4] = avgWT - avgW*avgT_w       #FwT  on w nodes


    return(Rflux)


##############################################################################
##############################################################################
##############################################################################



# DispersiveFlux(): function that computes the 3D Reynolds Stress
##############################################################################

def DispersiveFlux(Nx,Ny,Nz,avgT,avgU,avgV,avgW):
    "Calling the DispersiveFlux() function"

    import numpy as np
    import xarray as xr
    from Stats import ProjectOn_w,ProjectOn_uvp

    Dflux = xr.DataArray(np.ones(shape = (Nz,5),order='F'),\
                        dims=('z','variable'), coords = {'variable':['DTT','DuT',\
                        'DvT','DwT_uvp','DwT_w']})

    avgW_uvp = np.zeros((Nx,Ny,Nz),'d',order='F')
    avgT_w = np.zeros((Nx,Ny,Nz),'d',order='F')

    avgW_uvp = ProjectOn_uvp(Nx,Ny,Nz,avgW)
    avgT_w = ProjectOn_w(Nx,Ny,Nz,avgT)



    dispT = avgT - np.mean(np.mean(avgT,axis = 0),axis=0)
    dispT_w = avgT_w - np.mean(np.mean(avgT_w,axis = 0),axis=0)
    dispU = avgU - np.mean(np.mean(avgU,axis = 0),axis=0)
    dispV = avgV - np.mean(np.mean(avgV,axis = 0),axis=0)
    dispW = avgW - np.mean(np.mean(avgW,axis = 0),axis=0)
    dispW_uvp = avgW_uvp - np.mean(np.mean(avgW_uvp,axis = 0),axis=0)

    Dflux[:,0] = np.mean(np.mean(np.multiply(dispT,dispT),axis = 0),axis = 0) #DTT
    Dflux[:,1] = np.mean(np.mean(np.multiply(dispU,dispT),axis = 0),axis = 0) #DuT
    Dflux[:,2] = np.mean(np.mean(np.multiply(dispV,dispT),axis = 0),axis = 0) #DvT
    Dflux[:,3] = np.mean(np.mean(np.multiply(dispW_uvp,dispT),axis = 0),axis = 0) #DwT_uvp nodes
    Dflux[:,4] = np.mean(np.mean(np.multiply(dispW,dispT_w),axis = 0),axis = 0) #DwT_w nodes


    return(Dflux)

##############################################################################
##############################################################################
##############################################################################

# DispFluct(): function that computes the 3D Dispersive Fluctuations
##############################################################################

def DispFluct(Nx,Ny,Nz,avgVar):
    "Calling the DispersiveFlux() function"

    import numpy as np
    import xarray as xr

    dispFluct = np.ones(shape = (Nx,Ny,Nz),order='F')

    dispFluct = avgVar - np.mean(np.mean(avgVar,axis = 0),axis=0)


    return(dispFluct)

##############################################################################
##############################################################################
##############################################################################


# VertProfile(): function that computes the 3D Reynolds Stress
##############################################################################

def mean_xy(var3D):
    "Calling the VertProfile() function"

    import numpy as np

    var1D = np.mean(np.mean(var3D,axis = 0),axis=0)

    return(var1D)

##############################################################################
##############################################################################
##############################################################################


# Dispersive Stress on the Reduced/Coarser LES grid():
##############################################################################
def DispersiveR(avgGridsXY,avgGridsZ,Nx,Ny,Nz,Nx_new,Ny_new,Nz_new,\
                      var3D_u,var3D_v,var3D_w,var3DR_u,var3DR_v,var3DR_w,var3DS_T,var3DSR_T):
    "Calling the DispersiveStress() function"

    import numpy as np
    import xarray as xr
    from Stats import ProjectOn_uvp

    #Projecting variables u,v & T on the w grid. ----------------------------
    var3D_u_wNodes = np.empty((Nx,Ny,Nz),'d',order='F')
    var3D_v_wNodes = np.empty((Nx,Ny,Nz),'d',order='F')
    var3DS_T_wNodes = np.empty((Nx,Ny,Nz),'d',order='F')

    var3D_u_wNodes = ProjectOn_w(Nx,Ny,Nz,var3D_u)
    var3D_v_wNodes = ProjectOn_w(Nx,Ny,Nz,var3D_v)
    var3DS_T_wNodes = ProjectOn_w(Nx,Ny,Nz,var3DS_T)
    #------------------------------------------------------------------------

    var3DR_u_wNodes = np.empty((Nx_new,Ny_new,Nz_new),'d',order='F')
    var3DR_v_wNodes = np.empty((Nx_new,Ny_new,Nz_new),'d',order='F')
    var3DSR_T_wNodes = np.empty((Nx_new,Ny_new,Nz_new),'d',order='F')

    #Compute the interpolated to w-node variables on the reduced grid:
    for k in range(0,Nz_new):
        z0 = k*avgGridsZ
        zf = (k+1)*avgGridsZ

        for j in range(0,Ny_new):
            y0 = j*avgGridsXY
            yf = (j+1)*avgGridsXY

            for i in range(0,Nx_new):
                x0 = i*avgGridsXY
                xf = (i+1)*avgGridsXY

                var3DR_u_wNodes[i,j,k] = np.mean(var3D_u_wNodes[x0:xf,y0:yf,z0:zf],axis =(0,1,2))
                var3DR_v_wNodes[i,j,k] = np.mean(var3D_v_wNodes[x0:xf,y0:yf,z0:zf],axis =(0,1,2))
                var3DSR_T_wNodes[i,j,k] = np.mean(var3DS_T_wNodes[x0:xf,y0:yf,z0:zf],axis =(0,1,2))




    DFluct = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,7),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['U','V',\
                        'W','T','U_wNode','V_wNode','T_wNode']})

    DstressR = xr.DataArray(np.ones(shape = (Nx_new,Ny_new,Nz_new,10),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Dxx','Dyy',\
                        'Dzz','Dxy','Dxz_wNode','Dyz_wNode','DTT','DuT','DvT','DwT_wNode']})

    for k in range(0,Nz_new):
        z0 = k*avgGridsZ
        zf = (k+1)*avgGridsZ

        for j in range(0,Ny_new):
            y0 = j*avgGridsXY
            yf = (j+1)*avgGridsXY

            for i in range(0,Nx_new):
                x0 = i*avgGridsXY
                xf = (i+1)*avgGridsXY

                #Compute the Dispersive fluctuations:
                DFluct.data[x0:xf,y0:yf,z0:zf,0] = var3D_u[x0:xf,y0:yf,z0:zf] - var3DR_u[i,j,k]
                DFluct.data[x0:xf,y0:yf,z0:zf,1] = var3D_v[x0:xf,y0:yf,z0:zf] - var3DR_v[i,j,k]
                DFluct.data[x0:xf,y0:yf,z0:zf,2] = var3D_w[x0:xf,y0:yf,z0:zf] - var3DR_w[i,j,k]
                DFluct.data[x0:xf,y0:yf,z0:zf,3] = var3DS_T[x0:xf,y0:yf,z0:zf] - var3DSR_T[i,j,k]

                #Compute the U'', V'' & T'' at the wNodes:
                DFluct.data[x0:xf,y0:yf,z0:zf,4] = var3D_u_wNodes[x0:xf,y0:yf,z0:zf] - var3DR_u_wNodes[i,j,k]
                DFluct.data[x0:xf,y0:yf,z0:zf,5] = var3D_v_wNodes[x0:xf,y0:yf,z0:zf] - var3DR_v_wNodes[i,j,k]
                DFluct.data[x0:xf,y0:yf,z0:zf,6] = var3DS_T_wNodes[x0:xf,y0:yf,z0:zf] - var3DSR_T_wNodes[i,j,k]

                #Compute the dispersive stresses on the new reduced grid:
                DstressR[i,j,k,0] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,0]*DFluct.data[x0:xf,y0:yf,z0:zf,0]) #<\overline{u}''\overline{u}''> --> uvpT node
                DstressR[i,j,k,1] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,1]*DFluct.data[x0:xf,y0:yf,z0:zf,1]) #<\overline{v}''\overline{v}''> --> uvpT node
                DstressR[i,j,k,2] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,2]*DFluct.data[x0:xf,y0:yf,z0:zf,2]) #<\overline{w}''\overline{w}''> --> w node
                DstressR[i,j,k,3] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,0]*DFluct.data[x0:xf,y0:yf,z0:zf,1]) #<\overline{u}''\overline{v}''> --> uvpT node
                DstressR[i,j,k,4] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,4]*DFluct.data[x0:xf,y0:yf,z0:zf,2]) #<\overline{u}''\overline{w}''> --> w node
                DstressR[i,j,k,5] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,5]*DFluct.data[x0:xf,y0:yf,z0:zf,2]) #<\overline{v}''\overline{w}''> --> w node

                DstressR[i,j,k,6] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,3]*DFluct.data[x0:xf,y0:yf,z0:zf,3]) #<\overline{T}''\overline{T}''> --> uvpT node
                DstressR[i,j,k,7] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,0]*DFluct.data[x0:xf,y0:yf,z0:zf,3]) #<\overline{u}''\overline{T}''> --> uvpT node
                DstressR[i,j,k,8] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,1]*DFluct.data[x0:xf,y0:yf,z0:zf,3]) #<\overline{v}''\overline{T}''> --> uvpT node
                DstressR[i,j,k,9] =  np.mean(DFluct.data[x0:xf,y0:yf,z0:zf,2]*DFluct.data[x0:xf,y0:yf,z0:zf,6]) #<\overline{w}''\overline{T}''> --> w node


    return(DstressR)


# DispFluctR(): computes the 3D Dispersive Fluctuations based on the Reduced LES grid
##############################################################################

def DispFluctR(avgGridsXY,avgGridsZ,Nx,Ny,Nz,Nx_new,Ny_new,Nz_new,Var3D,Var3DR):
    "Calling the DispersiveFlux() function"

    import numpy as np
    import xarray as xr

    dispFluct = np.ones(shape = (Nx,Ny,Nz),order='F')

    for k in range(0,Nz_new):
        z0 = k*avgGridsZ
        zf = (k+1)*avgGridsZ

        for j in range(0,Ny_new):
            y0 = j*avgGridsXY
            yf = (j+1)*avgGridsXY

            for i in range(0,Nx_new):
                x0 = i*avgGridsXY
                xf = (i+1)*avgGridsXY

                #Compute the Dispersive fluctuations:
                dispFluct[x0:xf,y0:yf,z0:zf] = Var3D[x0:xf,y0:yf,z0:zf] - Var3DR[i,j,k]



    return(dispFluct)

#def DispFluctR(avgGridsXY,avgGridsZ,Nx,Ny,Nz,Nx_new,Ny_new,Var3D,Var3DR):
#    "Calling the DispersiveFlux() function"

#    import numpy as np
#    import xarray as xr

#    dispFluct = np.ones(shape = (Nx,Ny,Nz),order='F')

#    for k in range(0,Nz):

#        for j in range(0,Ny_new):
#            y0 = j*avgGridsXY
#            yf = (j+1)*avgGridsXY

#            for i in range(0,Nx_new):
#                x0 = i*avgGridsXY
#                xf = (i+1)*avgGridsXY

                #Compute the Dispersive fluctuations:
#                dispFluct[x0:xf,y0:yf,k] = Var3D[x0:xf,y0:yf,k] - Var3DR[i,j,k]



#    return(dispFluct)


##############################################################################
##############################################################################
##############################################################################


# mean_xyz(): Computes the CVolume average on the reduced LES grid
##############################################################################

def mean_xyz(avgGridsXY,avgGridsZ,Nx_new,Ny_new,Nz_new,var3D):
    "Calling the VertProfile() function"

    import numpy as np

    var3D_avgCV = np.ones(shape = (Nx_new,Ny_new,Nz_new),order='F')

    for k in range(0,Nz_new):
        z0 = k*avgGridsZ
        zf = (k+1)*avgGridsZ
        #if k == 0:
        #    z0 = 1

        for j in range(0,Ny_new):
            y0 = j*avgGridsXY
            yf = (j+1)*avgGridsXY

            for i in range(0,Nx_new):
                x0 = i*avgGridsXY
                xf = (i+1)*avgGridsXY

                #Compute the dispersive stresses on the new reduced grid:
                var3D_avgCV[i,j,k] =  np.mean(var3D[x0:xf,y0:yf,z0:zf])


    return(var3D_avgCV)

##############################################################################
##############################################################################
##############################################################################


# TKE_Budget(): Computes the TKE Budget terms
##############################################################################

def TKE_Budget(Nx,Ny,Nz,Lx,Ly,Lz,dx,dy,dz,data,dataS):
    "Calling the TKE_Budget() function"

    
    import numpy as np
    import os
    import xarray as xr
    
    os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")
    
    from Stats import  ReynoldsStress, ReynoldsFlux
    from Analysis import Fddx, Fddy, ddz3_w, ddz3_uv

    
#Compute the Reynolds Stresses:

    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                         data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])



    Rflux = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,5),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['FTT','FuT',\
                        'FvT','FwT_uvp','FwT_w']})


    Rflux = ReynoldsFlux(Nx,Ny,Nz,dataS.data[:,:,:,0],dataS.data[:,:,:,1],dataS.data[:,:,:,2],dataS.data[:,:,:,3],\
               dataS.data[:,:,:,4],data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2])


    #Compute the TKE

    e = 0.5*((Rstress.data[:,:,:,0] + Rstress.data[:,:,:,1] + Rstress.data[:,:,:,2])+ \
        (data.data[:,:,:,16] + data.data[:,:,:,17] + data.data[:,:,:,18]))
        
    #Note that according to Albertson's thesis all three terms are computed at the same grid point. No need to shift.

    
    #Compute the Advection Terms of the TKE Equation:
    #
    # Adevc = Ui*dedxi

    #Note: we proceed without dealiasing the advection terms below. 

    #Term 1 TKE: U dedx + Vdedy + Wdedz

    U = data.data[:,:,:,0] 
    V = data.data[:,:,:,1]
    W = data.data[:,:,:,2]

    order = 1
    dedx = Fddx(Nx,Ny,Nz,e,order,0,Lx,Nx)
    dedy = Fddy(Nx,Ny,Nz,e,order,0,Ly,Ny)
    dedz = ddz3_uv(Nx,Ny,Nz,dz,e)

    tmp = W*dedz
    Advec_z = np.zeros((Nx,Ny,Nz),'d',order='F')

    for k in range(0,Nz-1): #Takes a value from the w-node and places it in the uvpT-node.
            Advec_z[:,:,k] = (tmp[:,:,k] + tmp[:,:,k+1])/2

    Advec = U*dedx + V*dedy + Advec_z


    #Compute the Buoyancy Production/Destruction Term of the TKE Equation:
    #
    # Buoyancy = (g/T)*(w'T')

    g = 9.81 #[m/s^2]
    g_star = g * (1000/(0.45**2)) #Non-dimensional g.

    T = dataS.data[:,:,:,0]

    Buoyancy = (g_star/T)*(Rflux.data[:,:,:,3] - dataS.data[:,:,:,7])
    
    
    #Compute the Mechanical shear production/destruction:
    #
    # Prod = -(u_i'u_j')dU_idx_j = 
    #      = - [ (u'u')dUdx + (u'v')dUdy + (u'w')dUdz +
    #            (v'u')dVdx + (v'v')dVdy + (v'w')dVdz +
    #            (w'u')dWdx + (w'v')dWdy + (w'w')dWdz]

    #Derivatives:
    dUdx = Fddx(Nx,Ny,Nz,U,order,0,Lx,Nx)
    dVdx = Fddx(Nx,Ny,Nz,V,order,0,Lx,Nx)
    dWdx = Fddx(Nx,Ny,Nz,W,order,0,Lx,Nx)

    dUdy = Fddy(Nx,Ny,Nz,U,order,0,Ly,Ny)
    dVdy = Fddy(Nx,Ny,Nz,V,order,0,Ly,Ny)
    dWdy = Fddy(Nx,Ny,Nz,W,order,0,Ly,Ny)

    tmp_dUdz = ddz3_uv(Nx,Ny,Nz,dz,U) #This is now on the w-nodes.
    tmp_dVdz = ddz3_uv(Nx,Ny,Nz,dz,V) #This is now on the w-nodes.
    dUdz = np.zeros((Nx,Ny,Nz),'d',order='F')
    dVdz = np.zeros((Nx,Ny,Nz),'d',order='F')
    for k in range(0,Nz-1): #Takes a value from the w-node and places it in the uvpT-node.
            dUdz[:,:,k] = (tmp_dUdz[:,:,k] + tmp_dUdz[:,:,k+1])/2
            dVdz[:,:,k] = (tmp_dVdz[:,:,k] + tmp_dVdz[:,:,k+1])/2

    dWdz = ddz3_w(Nx,Ny,Nz,dz,W) #This is now on the uvpT-nodes.   
        
        
    #Stresses:

    uu = Rstress.data[:,:,:,0] + data.data[:,:,:,16]    
    vv = Rstress.data[:,:,:,1] + data.data[:,:,:,17]
    ww = Rstress.data[:,:,:,2] + data.data[:,:,:,18]

    uv = Rstress.data[:,:,:,3] + data.data[:,:,:,19]
    uw = Rstress.data[:,:,:,4] - data.data[:,:,:,20]
    vw = Rstress.data[:,:,:,5] - data.data[:,:,:,21]


    Prod = (uu*dUdx) + (uv*dUdy) + (uw*dUdz) + \
           (uv*dVdx) + (vv*dVdy) + (vw*dVdz) + \
           (uw*dWdx) + (vw*dWdy) + (ww*dWdz)


    #Compute the budget (without the missing terms that we don't have on Fabien's data)

    Budget = -Advec + Buoyancy + Prod

    return(Advec,Buoyancy,Prod,Budget)

##############################################################################
##############################################################################
##############################################################################


# Analysis_Sigma(): Computes the TKE Budget terms
##############################################################################

def Analysis_Sigma(Nx,Ny,Nz,Nz_SLayer,Lx,Ly,Lz,dx,dy,dz,path):
    "Calling the Analysis_Sigma() function"

    
    import numpy as np
    import os
    import xarray as xr
    
    os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")
    
    from Stats import  ReynoldsStress, ReynoldsFlux
    
    
    #data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
    #                        dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
    #                        'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
    #                        'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
    #                        'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
    #                        'avgdudz','avgdvdz','avgNut','avgCs']})


    #dataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
    #                        dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
    #                        'avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
    #                        'avg_ds']})

    #dataS_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
    #                dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
    #                'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

    #data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
    #                dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})

        

    print(f'Reading from file {path}')

    
    data = xr.open_dataarray(path + 'Data_Momentum.nc')
    dataS = xr.open_dataarray(path + 'Data_Scalar.nc')
    dataS_2D = xr.open_dataarray(path + 'Data_Scalar_2D.nc')
    data_2D = xr.open_dataarray(path + 'Data_Momentum_2D.nc')
    
    #----------------------------------------------------------------------
    #Compute SigmaT and Tstar using surface based values.
    varT = dataS.data[:,:,0:Nz_SLayer,1] - (dataS.data[:,:,0:Nz_SLayer,0]**2)
    varU = data.data[:,:,0:Nz_SLayer,4] - (data.data[:,:,0:Nz_SLayer,0]**2)
    varV = data.data[:,:,0:Nz_SLayer,5] - (data.data[:,:,0:Nz_SLayer,1]**2)
    varW = data.data[:,:,0:Nz_SLayer,6] - (data.data[:,:,0:Nz_SLayer,2]**2)
    
    
    sigmaT = np.array(np.sqrt(varT),order = 'F')
    sigmaU = np.array(np.sqrt(varU),order = 'F')
    sigmaV = np.array(np.sqrt(varV),order = 'F')
    sigmaW = np.array(np.sqrt(varW),order = 'F')
    
    
    #----------------------------------------------------------------------        
    #Alternatively, from the 3D LES files: compute u_star and Heat flux local at each height.
    # Computing the Reynolds Stress Tensor and Sensible Heat fluxes:

    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                                dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                                'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                                 data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])



    Rflux = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,5),order='F'),\
                                dims=('x','y','z','variable'), coords = {'variable':['FTT','FuT',\
                                'FvT','FwT_uvp','FwT_w']})


    Rflux = ReynoldsFlux(Nx,Ny,Nz,dataS.data[:,:,:,0],dataS.data[:,:,:,1],dataS.data[:,:,:,2],dataS.data[:,:,:,3],\
                       dataS.data[:,:,:,4],data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2])
    
    #---------------------------------------------------------------------
    
    ustar_local = np.array((((-Rstress.data[:,:,0:Nz_SLayer,4] + data.data[:,:,0:Nz_SLayer,20])**2 + \
                            (-Rstress.data[:,:,0:Nz_SLayer,5] + data.data[:,:,0:Nz_SLayer,21])**2)**(1/4)),order = 'F')
    
    wT_local = np.array((Rflux[:,:,0:Nz_SLayer,3]-dataS.data[:,:,0:Nz_SLayer,7]),order = 'F')
        
    
    Ts = np.array(dataS.data[:,:,0:Nz_SLayer,0],order = 'F')
    
    return(sigmaT,sigmaU,sigmaV,sigmaW,ustar_local,wT_local,Ts)






