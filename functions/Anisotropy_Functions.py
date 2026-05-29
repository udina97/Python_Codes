"""
Program Name: Anisotropy_Functions.py
Program purpose: This program computes the turbulence anisotropy of any datset

Program Author: Marc Calaf.

Date created: 19 October 2022
Last date modified: 19 October 2022

To Do: ...

"""

# Anisotropy(): function that computes the turbulence anisotropy and returns (xB,yB)
####################################################################################

def Anisotropy(Nx,Ny,Nz,R11,R22,R33,R12,R13,R23):
    "Calling the Anisotorpy() function"
    
    import numpy as np
    
    # Calculate the TKE
    e = R11[:,:,:] + R22[:,:,:] + R33[:,:,:]

    # Define identity matrix
    Id = np.eye(3,3) 

    xB = np.zeros((Nx,Ny,Nz),order='F')
    yB = np.zeros((Nx,Ny,Nz),order = 'F')
    lambda3 = np.zeros((Nx,Ny,Nz),order='F')


    #... Loop over each point of the LES domain.
    for k in range(0,Nz):
        print(f"iteration = {k}")
        for i in range(0,Nx):
            for j in range(0,Ny):
           
                # ... Reynolds Matrix
                R = np.matrix([[R11[i,j,k], R12[i,j,k], R13[i,j,k]],
                               [R12[i,j,k], R22[i,j,k], R23[i,j,k]],
                               [R13[i,j,k], R23[i,j,k], R33[i,j,k]]])
       
       

                # .. calculate the anisotropy tensor
                B = R/e[i,j,k] -1/3*Id
            
                # .. calculate the eigenvalues values
                [eigenVal,eigenVec] = np.linalg.eig(B)
       
                #Sort Eigenvalues in decreasing order
                SortedeigenVal = np.sort(eigenVal)[::-1] 
                lambda3[i,j,k] = SortedeigenVal[2]
           
       
                # .. compute the C coefficients for Barycentric map
                C1c = SortedeigenVal[0] - SortedeigenVal[1]
                C2c = 2*(SortedeigenVal[1] - SortedeigenVal[2])
                C3c = 3*SortedeigenVal[2] + 1
       
                # .. compute the barycentric invariants
                xB[i,j,k] = C1c + C3c*1/2
                yB[i,j,k] = C3c*np.sqrt(3)/2
       
    
    
    return(xB,yB,lambda3)

def Anisotropy2D(Nx,Nz,R11,R22,R33,R12,R13,R23):
    "Calling the Anisotorpy() function"
    
    import numpy as np
    
    # Calculate the TKE
    e = R11[:,:] + R22[:,:] + R33[:,:]

    # Define identity matrix
    Id = np.eye(3,3) 

    xB = np.zeros((Nx,Nz),order='F')
    yB = np.zeros((Nx,Nz),order = 'F')


    #... Loop over each point of the LES domain.
    for k in range(0,Nz):
        print(f"iteration = {k}")
        for i in range(0,Nx):
           
                # ... Reynolds Matrix
                R = np.matrix([[R11[i,k], R12[i,k], R13[i,k]],
                               [R12[i,k], R22[i,k], R23[i,k]],
                               [R13[i,k], R23[i,k], R33[i,k]]])
       
       

                # .. calculate the anisotropy tensor
                B = R/e[i,k] -1/3*Id
            
                # .. calculate the eigenvalues values
                [eigenVal,eigenVec] = np.linalg.eig(B)
       
                #Sort Eigenvalues in decreasing order
                SortedeigenVal = np.sort(eigenVal)[::-1] 
           
       
                # .. compute the C coefficients for Barycentric map
                C1c = SortedeigenVal[0] - SortedeigenVal[1]
                C2c = 2*(SortedeigenVal[1] - SortedeigenVal[2])
                C3c = 3*SortedeigenVal[2] + 1
       
                # .. compute the barycentric invariants
                xB[i,k] = C1c + C3c*1/2
                yB[i,k] = C3c*np.sqrt(3)/2
       
    
    
    return(xB,yB)

####################################################################################
####################################################################################
####################################################################################


def ColorAnisotropy():
    "Calling the ColorAnisotorpy() function"
    
    import numpy as np
    
    iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
                  "#b29f74","#7f816b","#587571","#596c72","#454f51"]

    #Transform the HEX colors to RGB.
    from PIL import ImageColor

    iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')

    for i in range(0,np.size(iva_colors_HEX)):
        iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")

    iva_colors_RGB = iva_colors_RGB[:,:]/(256)

    #Transform the array of colors to a list of values. 
    colors = iva_colors_RGB.tolist()
    #----------------------------------------------------

    #The next few lines create a new colormap using IVA's colors:
    from matplotlib.colors import LinearSegmentedColormap,ListedColormap

    inbetween_color_amount = 10

    # the 10 is from the original 10 colors, the 4 is for R, G, B, A
    newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))

    # add first one already
    newcolvals[0] = colors[0]

    for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
        for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
            flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
            # discard first 1 since we already have it from previous iteration
            flow = flow[1:]
            newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
        
    newcolvals

    cmap = ListedColormap(newcolvals, name='from_list', N=None)
    
    return(cmap)

####################################################################################
####################################################################################
####################################################################################


def phi_m(Nx,Ny,Nz_SLayer,z_d,u,v,avgdUdz,avgdVdz,ustar):
    
    import numpy as np
    
    kappa = 0.4
    
    #Initialization of a couple variables:
    meandUdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
    phi_m_3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')


    for k in range(0,Nz_SLayer): 
        meandUdz[:,:,k] = ((u[:,:,k]*avgdUdz[:,:,k]) + (v[:,:,k]*avgdVdz[:,:,k]))/(np.sqrt(u[:,:,k]**2 + v[:,:,k]**2))
        phi_m_3D[:,:,k] =  (kappa*z_d[k]/ustar)*meandUdz[:,:,k]


    return(phi_m_3D)

def phi_m_loc(Nx,Ny,Nz_SLayer,z_d,u,v,avgdUdz,avgdVdz,ustar):
    
    import numpy as np
    
    kappa = 0.4
    
    #Initialization of a couple variables:
    meandUdz = np.zeros((Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
    phi_m_1d = np.zeros((Nz_SLayer),'d',order='F')


    for k in range(0,Nz_SLayer): 
        meandUdz[k] = ((u[k]*avgdUdz[k]) + (v[k]*avgdVdz[k]))/(np.sqrt(u[k]**2 + v[k]**2))
        phi_m_1d[k] =  (kappa*z_d[k]/ustar)*meandUdz[k]


    return(phi_m_1d)


####################################################################################
####################################################################################
####################################################################################


def Anisotropy_Clustering(Nx,Ny,Nz_SLayer,Rstress):
    
    import numpy as np
    import xarray as xr
    
    # Analyzing turbulence anisotropy
    
    xB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    yB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    lambda3 = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')


    [xB,yB,lambda3] = Anisotropy(Nx,Ny,Nz_SLayer,Rstress.data[:,:,:,0],Rstress.data[:,:,:,1],Rstress.data[:,:,:,2],
                         Rstress.data[:,:,:,3],Rstress.data[:,:,:,4],Rstress.data[:,:,:,5])
    
    
    # Clustering data based on anisotropy:

    AnisType = np.zeros((Nx,Ny,Nz_SLayer),dtype='int',order='F')

    for k in range(0,Nz_SLayer):
        for i in range(0,Nx):
            for j in range(0,Ny):

                if (yB[i,j,k] < 0.1):
                    
                   AnisType[i,j,k] = 1
                
                elif (yB[i,j,k] >= 0.1 and yB[i,j,k] < 0.2):
        
                    AnisType[i,j,k] = 2
                
                elif (yB[i,j,k] >= 0.2 and yB[i,j,k] < 0.3):
                    
                    AnisType[i,j,k] = 3
                
                elif (yB[i,j,k] >= 0.3 and yB[i,j,k] < 0.4):
                    
                    AnisType[i,j,k] = 4
                
                elif (yB[i,j,k] >= 0.4 and yB[i,j,k] < 0.5):
                    
                    AnisType[i,j,k] = 5
                
                elif (yB[i,j,k] >= 0.5 and yB[i,j,k] < 0.6):
        
                    AnisType[i,j,k] = 6
                
                elif (yB[i,j,k] >= 0.6 and yB[i,j,k] < 0.7):
                    
                    AnisType[i,j,k] = 7
                
                elif (yB[i,j,k] >= 0.7 and yB[i,j,k] < 0.8):
                    
                    AnisType[i,j,k] = 8
                
                else: 
                    
                    AnisType[i,j,k] = 9
    
    
    AnisType_1D = np.ndarray.flatten(AnisType)
    
    
    return(xB,yB,AnisType_1D,lambda3)


def Anisotropy_Clustering2D(Nx,Nz_SLayer,Rstress):
    
    import numpy as np
    import xarray as xr
    
    # Analyzing turbulence anisotropy
    
    xB = np.zeros((Nx,Nz_SLayer),'d',order='F')
    yB = np.zeros((Nx,Nz_SLayer),'d',order='F')


    [xB,yB] = Anisotropy2D(Nx,Nz_SLayer,Rstress.data[:,:,0],Rstress.data[:,:,1],Rstress.data[:,:,2],
                         Rstress.data[:,:,3],Rstress.data[:,:,4],Rstress.data[:,:,5])
    
    
    # Clustering data based on anisotropy:

    AnisType = np.zeros((Nx,Nz_SLayer),dtype='int',order='F')

    for k in range(0,Nz_SLayer):
        for i in range(0,Nx):

                if (yB[i,k] < 0.1):
                    
                   AnisType[i,k] = 1
                
                elif (yB[i,k] >= 0.1 and yB[i,k] < 0.2):
        
                    AnisType[i,k] = 2
                
                elif (yB[i,k] >= 0.2 and yB[i,k] < 0.3):
                    
                    AnisType[i,k] = 3
                
                elif (yB[i,k] >= 0.3 and yB[i,k] < 0.4):
                    
                    AnisType[i,k] = 4
                
                elif (yB[i,k] >= 0.4 and yB[i,k] < 0.5):
                    
                    AnisType[i,k] = 5
                
                elif (yB[i,k] >= 0.5 and yB[i,k] < 0.6):
        
                    AnisType[i,k] = 6
                
                elif (yB[i,k] >= 0.6 and yB[i,k] < 0.7):
                    
                    AnisType[i,k] = 7
                
                elif (yB[i,k] >= 0.7 and yB[i,k] < 0.8):
                    
                    AnisType[i,k] = 8
                
                else: 
                    
                    AnisType[i,k] = 9
    
    
    AnisType_1D = np.ndarray.flatten(AnisType)
    
    
    return(xB,yB,AnisType_1D)



####################################################################################
####################################################################################
####################################################################################

def PhiM_Data(Nx,Ny,Nz,Nz_SLayer,dz,zi,NumVariables,NumVariablesSC,data,dataS,data_2D,dataS_2D):
    
    
    #Libraries and Functions
    import numpy as np
    import os
    import xarray as xr


    os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


    from Stats import  ReynoldsStress, ReynoldsFlux
    from Anisotropy_Functions import Anisotropy
    
    #Defining a couple simulation parameters:
    u_scale = 0.45 #[m/s]
    step = 8 #Step used to run the for loops (we take 1 data point every 8 points in the Nx, ans Ny directions)

   
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

    
    # Computing the Surface Variables (ustar & L) using the 3D LES output, so these can be height dependent:

    kappa = 0.4
    g_star = 9.81/(u_scale**2/zi) 
    z = np.arange(1,Nz+1)*(dz/2)
    
    #From the 3D LES files:
    ustar = ((-Rstress.data[:,:,0:Nz_SLayer,4]+data.data[:,:,0:Nz_SLayer,20])**2 + (-Rstress.data[:,:,0:Nz_SLayer,5] + data.data[:,:,0:Nz_SLayer,21])**2)**(1/4)
    Ts = dataS.data[:,:,0:Nz_SLayer,0]
    wT = Rflux[:,:,0:Nz_SLayer,3]-dataS.data[:,:,0:Nz_SLayer,7]

    #To compute the spatially distributed Obukhov length 
    L = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    zoL = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    
    
    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):
                L[i,j,k] = - (Ts[i,j,k]*ustar[i,j,k]**3)/(kappa*g_star*wT[i,j,k])
                zoL[i,j,k] = z[k]/L[i,j,k]




    # Analyzing turbulence anisotropy
    
    xB = np.zeros((Nx,Ny,Nz),'d',order='F')
    yB = np.zeros((Nx,Ny,Nz),'d',order='F')


    [xB,yB] = Anisotropy(Nx,Ny,Nz_SLayer,Rstress.data[:,:,:,0],Rstress.data[:,:,:,1],Rstress.data[:,:,:,2],
                         Rstress.data[:,:,:,3],Rstress.data[:,:,:,4],Rstress.data[:,:,:,5])



    # Clustering data based on anisotropy:

    AnisType = np.zeros((Nx,Ny,Nz_SLayer),dtype='int',order='F')

    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):

                if (yB[i,j,k] < 0.1):
                    
                   AnisType[i,j,k] = 1
                
                elif (yB[i,j,k] >= 0.1 and yB[i,j,k] < 0.2):
        
                    AnisType[i,j,k] = 2
                
                elif (yB[i,j,k] >= 0.2 and yB[i,j,k] < 0.3):
                    
                    AnisType[i,j,k] = 3
                
                elif (yB[i,j,k] >= 0.3 and yB[i,j,k] < 0.4):
                    
                    AnisType[i,j,k] = 4
                
                elif (yB[i,j,k] >= 0.4 and yB[i,j,k] < 0.5):
                    
                    AnisType[i,j,k] = 5
                
                elif (yB[i,j,k] >= 0.5 and yB[i,j,k] < 0.6):
        
                    AnisType[i,j,k] = 6
                
                elif (yB[i,j,k] >= 0.6 and yB[i,j,k] < 0.7):
                    
                    AnisType[i,j,k] = 7
                
                elif (yB[i,j,k] >= 0.7 and yB[i,j,k] < 0.8):
                    
                    AnisType[i,j,k] = 8
                
                else: 
                    
                    AnisType[i,j,k] = 9


    # Analyzing the Gradients:
    '''
    #From the LES datasets:
    avgdUdz = data.data[:,:,:,22]
    avgdVdz = data.data[:,:,:,23]

    phi_u = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    phi_v = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    for k in range(0,Nz_SLayer):
        phi_u[:,:,k] =  (kappa*z[k]/ustar[:,:,k])*avgdUdz[:,:,k]
        phi_v[:,:,k] =  (kappa*z[k]/ustar[:,:,k])*avgdVdz[:,:,k]
    '''

    # Analyzing the vertical Gradients on the mean wind direction:

        #From the LES datasets:
        
    u = data.data[:,:,:,0]
    v = data.data[:,:,:,1]
    avgdUdz = data.data[:,:,:,22]
    avgdVdz = data.data[:,:,:,23]

    meandUdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
    phi_u = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    for k in range(0,Nz_SLayer): 
        meandUdz[:,:,k] = ((u[:,:,k]*avgdUdz[:,:,k]) + (v[:,:,k]*avgdVdz[:,:,k]))/(np.sqrt(u[:,:,k]**2 + v[:,:,k]**2))
        phi_u[:,:,k] =  (kappa*z[k]/ustar[:,:,k])*meandUdz[:,:,k]


    # Reshape the relevant variables into a single column vector:
        
    phi_M_1D = np.ndarray.flatten(phi_u)
    zoL_1D = np.ndarray.flatten(zoL)
    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)
    AnisType_1D = np.ndarray.flatten(AnisType)

    PhiM_AnisData = xr.DataArray(np.ones(shape = (Nx*Ny*Nz_SLayer,5),order='F'),\
                        dims=('space','variable'), coords = {'variable':['phi_M_1D','zoL_1D',\
                        'yB_1D','xB_1D','AnisType_1D']})
        
    PhiM_AnisData[:,0] = phi_M_1D; PhiM_AnisData[:,1] = zoL_1D; PhiM_AnisData[:,2] = yB_1D
    PhiM_AnisData[:,3] = xB_1D; PhiM_AnisData[:,4] = AnisType_1D    
    
    
    
    
    return(PhiM_AnisData,yB,xB)

####################################################################################
####################################################################################
####################################################################################



def PhiM_Data_Neutral(Nx,Ny,Nz,Nz_SLayer,dz,zi,NumVariables,data,data_2D):
    
    
    #Libraries and Functions
    import numpy as np
    import os
    import xarray as xr


    os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


    from Stats import  ReynoldsStress
    from Anisotropy_Functions import Anisotropy
    
    #Defining a couple simulation parameters:

    step = 1 #Step used to run the for loops (we take 1 data point every 8 points in the Nx, ans Ny directions)

   
    # Computing the Reynolds Stress Tensor and Sensible Heat fluxes:

    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                             data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])



    
    # Computing the Surface Variables (ustar & L) using the 3D LES output, so these can be height dependent:

    kappa = 0.4
    z = np.arange(1,Nz+1)*(dz/2)
    
    #From the 3D LES files:
    ustar = ((-Rstress.data[:,:,0:Nz_SLayer,4]+data.data[:,:,0:Nz_SLayer,20])**2 + (-Rstress.data[:,:,0:Nz_SLayer,5] + data.data[:,:,0:Nz_SLayer,21])**2)**(1/4)

    
    #Creating 3D height vector:
    z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    z = (np.arange(0,Nz_SLayer)*dz) + dz/2
    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):
                z3D[i,j,k] = z[k]



    # Analyzing turbulence anisotropy
    
    xB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    yB = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')


    [xB,yB] = Anisotropy(Nx,Ny,Nz_SLayer,Rstress.data[:,:,:,0],Rstress.data[:,:,:,1],Rstress.data[:,:,:,2],
                         Rstress.data[:,:,:,3],Rstress.data[:,:,:,4],Rstress.data[:,:,:,5])



    # Clustering data based on anisotropy:

    AnisType = np.zeros((Nx,Ny,Nz_SLayer),dtype='int',order='F')

    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):

                if (yB[i,j,k] < 0.1):
                    
                   AnisType[i,j,k] = 1
                
                elif (yB[i,j,k] >= 0.1 and yB[i,j,k] < 0.2):
        
                    AnisType[i,j,k] = 2
                
                elif (yB[i,j,k] >= 0.2 and yB[i,j,k] < 0.3):
                    
                    AnisType[i,j,k] = 3
                
                elif (yB[i,j,k] >= 0.3 and yB[i,j,k] < 0.4):
                    
                    AnisType[i,j,k] = 4
                
                elif (yB[i,j,k] >= 0.4 and yB[i,j,k] < 0.5):
                    
                    AnisType[i,j,k] = 5
                
                elif (yB[i,j,k] >= 0.5 and yB[i,j,k] < 0.6):
        
                    AnisType[i,j,k] = 6
                
                elif (yB[i,j,k] >= 0.6 and yB[i,j,k] < 0.7):
                    
                    AnisType[i,j,k] = 7
                
                elif (yB[i,j,k] >= 0.7 and yB[i,j,k] < 0.8):
                    
                    AnisType[i,j,k] = 8
                
                else: 
                    
                    AnisType[i,j,k] = 9



    # Analyzing the vertical Gradients on the mean wind direction:

        #From the LES datasets:
        
    u = data.data[:,:,:,0]
    v = data.data[:,:,:,1]
    avgdUdz = data.data[:,:,:,22]
    avgdVdz = data.data[:,:,:,23]

    meandUdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
    phi_u = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    h_canopyTop = 10 + 1 #One grid point above the top of the canopy. 

    for k in range(0,Nz_SLayer): 
        meandUdz[:,:,k] = ((u[:,:,k]*avgdUdz[:,:,k]) + (v[:,:,k]*avgdVdz[:,:,k]))/(np.sqrt(u[:,:,k]**2 + v[:,:,k]**2))
        phi_u[:,:,k] =  (kappa*z[k]/ustar[:,:,h_canopyTop])*meandUdz[:,:,k]


    # Reshape the relevant variables into a single column vector:
        
    phi_M_1D = np.ndarray.flatten(phi_u)
    z3D_1D = np.ndarray.flatten(z3D)
    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)
    AnisType_1D = np.ndarray.flatten(AnisType)

    PhiM_AnisData = xr.DataArray(np.ones(shape = (Nx*Ny*Nz_SLayer,5),order='F'),\
                        dims=('space','variable'), coords = {'variable':['phi_M_1D','yB_1D','xB_1D','AnisType_1D','z3D_1D']})
        
    PhiM_AnisData[:,0] = phi_M_1D; PhiM_AnisData[:,1] = yB_1D; PhiM_AnisData[:,2] = xB_1D; 
    PhiM_AnisData[:,3] = AnisType_1D; PhiM_AnisData[:,4] = z3D_1D;   
    
    
    
    
    return(PhiM_AnisData,yB,xB)

####################################################################################
####################################################################################
####################################################################################


def Phi_Data(Nx,Ny,Nz,Nz_SLayer,dz,zi,NumVariables,NumVariablesSC,data,dataS,data_2D,dataS_2D):
    
    
    #Libraries and Functions
    import numpy as np
    import os
    import xarray as xr


    os.chdir("/Users/mcalaf/Documents/Utah/PythonCodes/")


    from Stats import  ReynoldsStress, ReynoldsFlux
    from Anisotropy_Functions import Anisotropy
    
    #Defining a couple simulation parameters:
    u_scale = 0.45 #[m/s]
    step = 8 #Step used to run the for loops (we take 1 data point every 8 points in the Nx, ans Ny directions)

   
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

    
    # Computing the Surface Variables (ustar & L) using the 3D LES output, so these can be height dependent:

    kappa = 0.4
    g_star = 9.81/(u_scale**2/zi) 
    z = np.arange(1,Nz+1)*(dz/2)
    
    #From the 3D LES files:
    ustar = ((-Rstress.data[:,:,0:Nz_SLayer,4]+data.data[:,:,0:Nz_SLayer,20])**2 + (-Rstress.data[:,:,0:Nz_SLayer,5] + data.data[:,:,0:Nz_SLayer,21])**2)**(1/4)
    Ts = dataS.data[:,:,0:Nz_SLayer,0]
    wT = Rflux[:,:,0:Nz_SLayer,3]-dataS.data[:,:,0:Nz_SLayer,7]
    Tstar = wT/ustar

    #To compute the spatially distributed Obukhov length 
    L = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    zoL = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    
    
    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):
                L[i,j,k] = - (Ts[i,j,k]*ustar[i,j,k]**3)/(kappa*g_star*wT[i,j,k])
                zoL[i,j,k] = z[k]/L[i,j,k]




    # Analyzing turbulence anisotropy
    
    xB = np.zeros((Nx,Ny,Nz),'d',order='F')
    yB = np.zeros((Nx,Ny,Nz),'d',order='F')


    [xB,yB] = Anisotropy(Nx,Ny,Nz_SLayer,Rstress.data[:,:,:,0],Rstress.data[:,:,:,1],Rstress.data[:,:,:,2],
                         Rstress.data[:,:,:,3],Rstress.data[:,:,:,4],Rstress.data[:,:,:,5])



    # Clustering data based on anisotropy:

    AnisType = np.zeros((Nx,Ny,Nz_SLayer),dtype='int',order='F')

    for k in range(0,Nz_SLayer):
        for i in range(0,Nx,step):
            for j in range(0,Ny,step):

                if (yB[i,j,k] < 0.1):
                    
                   AnisType[i,j,k] = 1
                
                elif (yB[i,j,k] >= 0.1 and yB[i,j,k] < 0.2):
        
                    AnisType[i,j,k] = 2
                
                elif (yB[i,j,k] >= 0.2 and yB[i,j,k] < 0.3):
                    
                    AnisType[i,j,k] = 3
                
                elif (yB[i,j,k] >= 0.3 and yB[i,j,k] < 0.4):
                    
                    AnisType[i,j,k] = 4
                
                elif (yB[i,j,k] >= 0.4 and yB[i,j,k] < 0.5):
                    
                    AnisType[i,j,k] = 5
                
                elif (yB[i,j,k] >= 0.5 and yB[i,j,k] < 0.6):
        
                    AnisType[i,j,k] = 6
                
                elif (yB[i,j,k] >= 0.6 and yB[i,j,k] < 0.7):
                    
                    AnisType[i,j,k] = 7
                
                elif (yB[i,j,k] >= 0.7 and yB[i,j,k] < 0.8):
                    
                    AnisType[i,j,k] = 8
                
                else: 
                    
                    AnisType[i,j,k] = 9



    # Analyzing the vertical Gradients on the mean wind direction:

        #From the LES datasets:
        
    u = data.data[:,:,:,0]
    v = data.data[:,:,:,1]
    avgdUdz = data.data[:,:,:,22]
    avgdVdz = data.data[:,:,:,23]

    meandUdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F') # THis is for the Mean velocity gradient.
    phi_u = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

    for k in range(0,Nz_SLayer): 
        meandUdz[:,:,k] = ((u[:,:,k]*avgdUdz[:,:,k]) + (v[:,:,k]*avgdVdz[:,:,k]))/(np.sqrt(u[:,:,k]**2 + v[:,:,k]**2))
        phi_u[:,:,k] =  (kappa*z[k]/ustar[:,:,k])*meandUdz[:,:,k]


    #Temperature Gradient (needs to be computed since it is not an output from the LES)
    dTdz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    phi_T = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
    
    #@ the surface:
    phi_h = dataS_2D.data[:,:,4]
    dTdz[:,:,0] = - phi_h[:,:]*wT[:,:,0]/(ustar[:,:,0]*kappa*dz/2) #As it is written in the LES  
    phi_T[:,:,0] =  phi_h 
  
    #@ above the first LES grid point:
    for k in range(1,Nz_SLayer):
        dTdz[:,:,k] = (dataS.data[:,:,k,0]-dataS.data[:,:,k-1,0])/dz
        phi_T[:,:,k] =  (kappa*z[k]/Tstar[:,:,k])*dTdz[:,:,k]    

    # Reshape the relevant variables into a single column vector:
        
    phi_M_1D = np.ndarray.flatten(phi_u)
    zoL_1D = np.ndarray.flatten(zoL)
    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)
    AnisType_1D = np.ndarray.flatten(AnisType)
    phi_T_1D = np.ndarray.flatten(phi_T)

    PhiM_AnisData = xr.DataArray(np.ones(shape = (Nx*Ny*Nz_SLayer,6),order='F'),\
                        dims=('space','variable'), coords = {'variable':['phi_M_1D','zoL_1D',\
                        'yB_1D','xB_1D','AnisType_1D','phi_T_1D']})
        
    PhiM_AnisData[:,0] = phi_M_1D; PhiM_AnisData[:,1] = zoL_1D; PhiM_AnisData[:,2] = yB_1D
    PhiM_AnisData[:,3] = xB_1D; PhiM_AnisData[:,4] = AnisType_1D; PhiM_AnisData[:,5] = phi_T_1D   
    
    
    
    
    return(PhiM_AnisData,yB,xB)



####################################################################################
####################################################################################
####################################################################################




