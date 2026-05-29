"""
Program Name: Analysis.py
Program purpose: This program contains several analysis funcitons.
                 
Program Author: Marc Calaf.

Date created: 18 March 2020
Last date modified: 14 October 2020

"""


# ddz(): function that computes the 1D Vertical derivative
##############################################################################

def ddz(Nz,dz,var1D):
    "Calling the ddz() function"
    
    import numpy as np

    dF = np.empty(Nz,'d',order='F')  #This variable has the corresponding values of the derivative.


    for k in range(1,Nz):
        dF[k] = (var1D[k] - var1D[k-1])/dz
    
    dF[0] = 0.5*var1D[1]
        
        
    return(dF)


##############################################################################
##############################################################################
##############################################################################

# ddz3_w(): function that computes the 1D Vertical derivative
##############################################################################

def ddz3_w(Nx,Ny,Nz,dz,var3D):
    "Calling the ddz3_w() function. This takes a variable on the w nodes and places the derivative on the uvpT node"
    
    import numpy as np


    dF3 = np.empty((Nx,Ny,Nz),dtype='float',order='F')  #This variable has the corresponding values of the derivative.

    
    for k in range(0,Nz-1):
        
        dF3[:,:,k] = (var3D[:,:,k+1] - var3D[:,:,k])/dz
    
    dF3[:,:,Nz-1] = (var3D[:,:,Nz-1] - var3D[:,:,Nz-2])/dz
        
        
        
    return(dF3)


##############################################################################
##############################################################################
##############################################################################

# ddz3_uv(): function that computes the 1D Vertical derivative
##############################################################################

def ddz3_uv(Nx,Ny,Nz,dz,var3D):
    "Calling the ddz3_uv() function. This takes a variable on the uvpT nodes and places the derivative on the w node"
    
    import numpy as np


    dF3 = np.empty((Nx,Ny,Nz),'d',order='F')  #This variable has the corresponding values of the derivative.

    
    for k in range(1,Nz-1):
        
        dF3[:,:,k] = (var3D[:,:,k] - var3D[:,:,k-1])/dz
    
    # dF3[:,:,0] = 0.0 #This fixes the derivative at the surface. These are normally parametrized through MO.       
    dF3[:,:,0] = var3D[:,:,0]/(dz/2)
        
    return(dF3)


##############################################################################
##############################################################################
##############################################################################



# ddz3(): function that computes the 1D Vertical derivative
##############################################################################

def ddz3(Nx,Ny,Nz,dz,var3D):
    "Calling the ddz3() function"
    
    import numpy as np


    dF3 = np.empty((Nx,Ny,Nz),'d',order='F')  #This variable has the corresponding values of the derivative.

    #for k in range(1,Nz):
        
    #    dF3[:,:,k] = (var3D[:,:,k] - var3D[:,:,k-1])/dz
    
    #dF3[:,:,0] = 0.5*dF3[:,:,1]
    
    for k in range(0,Nz-2):
        
        dF3[:,:,k] = (var3D[:,:,k+1] - var3D[:,:,k])/dz
    
    dF3[:,:,Nz-1] = (var3D[:,:,Nz-1] - var3D[:,:,Nz-2])/dz
        
        
        
    return(dF3)


##############################################################################
##############################################################################
##############################################################################

# ddz3_2ndOrder(): function that computes the 1D Vertical derivative
##############################################################################

def ddz3HOrder(Nx,Ny,Nz,dz,var3D):
    "Calling the ddz3() function"
    
    import numpy as np


    dF3 = np.empty((Nx,Ny,Nz),'d',order='F')  #This variable has the corresponding values of the derivative.

    
    dF3[:,:,0] = (var3D[:,:,1] - var3D[:,:,0])/dz
    
    for k in range(1,Nz-2):
        
        dF3[:,:,k] = (var3D[:,:,k+1] - var3D[:,:,k-1])/(2*dz)
    
    dF3[:,:,Nz-1] = (var3D[:,:,Nz-1] - var3D[:,:,Nz-2])/dz
        
        
        
    return(dF3)


##############################################################################
##############################################################################
##############################################################################



# ddx3(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def ddx3(Nx,Ny,Nz,degree,var3D):
    "Calling the ddx3() function"
    
    import numpy as np
    from scipy.fftpack import diff
    #from Analysis import fourierderivative


    dfdx3 = np.empty((Nx,Ny,Nz),'d',order='F')  #This variable has the corresponding values of the derivative.
    signal1D = np.empty(Nx,'d',order = 'F')

    for k in range(0,Nz):
        for j in range(0,Ny):
            signal1D = var3D[:,j,k]
            dfdx3[:,j,k] = diff(signal1D,degree)
    
    return(dfdx3)


##############################################################################
##############################################################################
##############################################################################


# Fddx_1D(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def Fddx_1D(Nx,var1D,order,a,b,n):
    "Calling the Fddx_1D() function which computes the spectral derivative in the x-direction"
    
    import numpy as np
    #from Analysis import fourierderivative


    #FOURIERDERIVATIVE Fourier derivative
    #dfdx3 = Fddx3(var3D,order,a,b,n) approximates the derivative to a
    #discrete function var3D over the domain (a,b) with number of points n 
    #`order' indicates the Order of the derivative.  var3D is a matrix that must be
    #uniformly sampled, periodic, and contain an even number of samples in the 
    #direction we intend to compute the derivative.
    #For best results, f should be periodic such that f(x + a) = f(x + b).
    #As an example,
    #
    #   x = linspace(0,pi);
    #   f = exp(cos(x).*sin(2*x));
    #   dfdx = fourierderivative(f,0,pi);
    #
    #   Results for nonperiodic f are dubious.

    k = 2.0*np.pi/(b-a)*np.concatenate((np.arange(0,n/2),np.arange(-n/2,0)),0)
    tmp = np.fft.fft(var1D,n,0)
    tmp[int(n/2)] = 0.0
    dfdx = np.real(np.fft.ifft( ((1j*k)**order)*tmp,n,0))
    
    return(dfdx)



##############################################################################
##############################################################################
##############################################################################

# Fddx2D(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def Fddx2D(Nx,Ny,var2D,order,a,b,n):
    "Calling the Fddx2D() function which computes the spectral derivative in the x-direction"
    
    import numpy as np
    #from Analysis import fourierderivative


    #FOURIERDERIVATIVE Fourier derivative
    #dfdx3 = Fddx3(var3D,order,a,b,n) approximates the derivative to a
    #discrete function var3D over the domain (a,b) with number of points n 
    #`order' indicates the Order of the derivative.  var3D is a matrix that must be
    #uniformly sampled, periodic, and contain an even number of samples in the 
    #direction we intend to compute the derivative.
    #For best results, f should be periodic such that f(x + a) = f(x + b).
    #As an example,
    #
    #   x = linspace(0,pi);
    #   f = exp(cos(x).*sin(2*x));
    #   dfdx = fourierderivative(f,0,pi);
    #
    #   Results for nonperiodic f are dubious.

    dfdx = np.zeros((Nx,Ny),dtype='float')
    kx = np.zeros((Nx,Ny),dtype='float',order='F')

    for j in range(0,Ny):
        kx[:,j] = 2*np.pi/(b-a)*np.concatenate((np.arange(0,n/2),np.arange(-n/2,0)),0)
        

    tmp = np.fft.fft2(var2D[:,:],(Nx,Ny),axes=(0,1))
    tmp[int(n/2),:] = 0.0
    dfdx[:,:] = np.real(np.fft.ifft2( ((1j*kx)**order)*tmp,(Nx,Ny),axes=(0,1)))
    
    return(dfdx)


##############################################################################
##############################################################################
##############################################################################

# Fddy2D(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def Fddy2D(Nx,Ny,var2D,order,a,b,n):
    "Calling the Fddy2D() function which computes the spectral derivative in the y-direction"
    
    import numpy as np
    #from Analysis import fourierderivative


    #FOURIERDERIVATIVE Fourier derivative
    #dfdy3 = Fddy3(var3D,order,a,b,n) approximates the derivative to a
    #discrete function var3D over the domain (a,b) with number of points n 
    #`order' indicates the Order of the derivative.  var3D is a matrix that must be
    #uniformly sampled, periodic, and contain an even number of samples in the 
    #direction we intend to compute the derivative.
    #For best results, f should be periodic such that f(x + a) = f(x + b).
    #As an example,
    #
    #   y = linspace(0,pi);
    #   f = exp(cos(x).*sin(2*x));
    #   dfdy = fourierderivative(f,0,pi);
    #
    #   Results for nonperiodic f are dubious.

    dfdy = np.zeros((Nx,Ny),dtype='float')
    ky = np.zeros((Nx,Ny),dtype='float',order='F')

    for j in range(0,Nx):
        ky[:,j] = 2*np.pi/(b-a)*np.concatenate((np.arange(0,n/2),np.arange(-n/2,0)),0)
        

    tmp = np.fft.fft2(var2D[:,:],(Nx,Ny),axes=(0,1))
    tmp[int(n/2),:] = 0.0
    dfdy[:,:] = np.real(np.fft.ifft2( ((1j*ky)**order)*tmp,(Nx,Ny),axes=(0,1)))
    
    return(dfdy)


##############################################################################
##############################################################################
##############################################################################


# Fddx(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def Fddx(Nx,Ny,Nz,var3D,order,a,b,n):
    "Calling the Fddx() function which computes the spectral derivative in the x-direction"
    
    import numpy as np
    #from Analysis import fourierderivative


    #FOURIERDERIVATIVE Fourier derivative
    #dfdx3 = Fddx3(var3D,order,a,b,n) approximates the derivative to a
    #discrete function var3D over the domain (a,b) with number of points n 
    #`order' indicates the Order of the derivative.  var3D is a matrix that must be
    #uniformly sampled, periodic, and contain an even number of samples in the 
    #direction we intend to compute the derivative.
    #For best results, f should be periodic such that f(x + a) = f(x + b).
    #As an example,
    #
    #   x = linspace(0,pi);
    #   f = exp(cos(x).*sin(2*x));
    #   dfdx = fourierderivative(f,0,pi);
    #
    #   Results for nonperiodic f are dubious.

    dfdx = np.zeros((Nx,Ny,Nz),dtype='float',order='F')
    kx = np.zeros((Nx,Ny),dtype='float',order='F')

    for j in range(0,Ny):
        kx[:,j] = 2*np.pi/(b-a)*np.concatenate((np.arange(0,n/2),np.arange(-n/2,0)),0)
        
    for k in range(0,Nz):
        tmp = np.fft.fft2(var3D[:,:,k],(Nx,Ny),axes=(0,1))
        tmp[int(n/2),:] = 0.0
        dfdx[:,:,k] = np.real(np.fft.ifft2( ((1j*kx)**order)*tmp,(Nx,Ny),axes=(0,1)))
    
    return(dfdx)


##############################################################################
##############################################################################
##############################################################################

# Fddy(): function that begins the process to compute the horizontal-y derivative
##############################################################################

def Fddy(Nx,Ny,Nz,var3D,order,a,b,n):
    "Calling the Fddy() function which computes the spectral derivative in the x-direction"
    
    import numpy as np
    #from Analysis import fourierderivative


    #FOURIERDERIVATIVE Fourier derivative
    #dfdx3 = Fddx3(var3D,order,a,b,n) approximates the derivative to a
    #discrete function var3D over the domain (a,b) with number of points n 
    #`order' indicates the Order of the derivative.  var3D is a matrix that must be
    #uniformly sampled, periodic, and contain an even number of samples in the 
    #direction we intend to compute the derivative.
    #For best results, f should be periodic such that f(x + a) = f(x + b).
    #As an example,
    #
    #   x = linspace(0,pi);
    #   f = exp(cos(x).*sin(2*x));
    #   dfdx = fourierderivative(f,0,pi);
    #
    #   Results for nonperiodic f are dubious.

    dfdy = np.zeros((Nx,Ny,Nz),dtype='float',order='F')
    ky = np.zeros((Nx,Ny),dtype='float',order='F')

    for i in range(0,Nx):
        ky[i,:] = 2*np.pi/(b-a)*np.concatenate((np.arange(0,n/2),np.arange(-n/2,0)),0)

    
    for k in range(0,Nz):
        tmp = np.fft.fft2(var3D[:,:,k],(Nx,Ny),axes=(0,1))
        tmp[:,int(n/2)] = 0.0
        dfdy[:,:,k] = np.real(np.fft.ifft2( ((1j*ky)**order)*tmp,(Nx,Ny),axes=(0,1)))
    
    return(dfdy)


##############################################################################
##############################################################################
##############################################################################



# ddy3(): function that begins the process to compute the horizontal-x derivative
##############################################################################

def ddy3(Nx,Ny,Nz,degree,var3D):
    "Calling the ddx3() function"
    
    import numpy as np
    from scipy.fftpack import diff
    #from Analysis import fourierderivative


    dfdy3 = np.empty((Nx,Ny,Nz),'d',order='F')  #This variable has the corresponding values of the derivative.
    signal1D = np.empty(Nx,'d',order = 'F')

    for k in range(0,Nz):
        for i in range(0,Nx):
            signal1D = var3D[i,:,k]
            dfdy3[i,:,k] = diff(signal1D,degree)
    
    return(dfdy3)


##############################################################################
##############################################################################
##############################################################################

# dealias1_1D(): function that ffts the signal, and padds it with zeros and brings it back to the real space longer
##############################################################################

def dealias1_1D(N,var1D):
    "Calling the dealias1_1D() function"
    
    import numpy as np
    #from Analysis import fourierderivative

    
    Fv1 = np.fft.fft(var1D,N,norm='ortho')
    #Fv1 = np.fft.fft(var1D,N)
    #freq = np.fft.fftfreq(N)

    K = 3/2*N
    tmp_vpad1 = np.zeros(int(K),dtype='complex',order='C')
    vpad1 = np.zeros(int(K),dtype='double',order='C')

    indvpad = np.concatenate((np.arange(int(0),int(N/2)),np.arange(int(K-N/2),int(K))),axis=0)
    #indvpad_x = [1:Nx/2, Kx-Nx/2+1:Kx];
    #indvpad_y = [1:Ny/2, Ky-Ny/2+1:Ky];

    for i in range(0,N):
        tmp_vpad1[indvpad[i]] = Fv1[i]

   
    vpad1 = np.sqrt(3/2)*np.real(np.fft.ifft(tmp_vpad1,int(3*N/2),norm='ortho'))
    #vpad1 = (1/np.sqrt(3/2))*np.real(np.fft.ifft(tmp_vpad1,int(3*N/2)))
    #vpad1 = np.real(np.fft.ifft(tmp_vpad1,int(3*N/2)))
   
    
    return(vpad1)


##############################################################################
##############################################################################
##############################################################################

# dealias2_1D(): function that ffts the padded signal, removes the zeros and brings it back to the real space
##############################################################################

def dealias2_1D(N,var1D_padded):
    "Calling the dealias2_1D() function"
    
    import numpy as np
    #from Analysis import fourierderivative


    K = 3/2*N
    
    tmp_var1D = np.zeros(N,dtype='complex',order='C')
    var1D = np.zeros(N,dtype='double',order='C')

    indvpad = np.concatenate((np.arange(int(0),int(N/2)),np.arange(int(K-N/2),int(K))),axis=0)   
    #indvpad_x = [1:Nx/2, Kx-Nx/2+1:Kx];
    #indvpad_y = [1:Ny/2, Ky-Ny/2+1:Ky];
    
    Fv2 = np.fft.fft(var1D_padded,int(3/2*N),norm='ortho')
    #Fv2 = np.fft.fft(var1D_padded,int(3/2*N))
    #Fv2 = np.sqrt(3/2)*Fv2

    for i in range(0,N):
        tmp_var1D[i] = Fv2[indvpad[i]]
    
    var1D = (1/np.sqrt(3/2))*np.real(np.fft.ifft(tmp_var1D,N,norm='ortho'))
    #var1D = np.sqrt(3/2)*np.real(np.fft.ifft(tmp_var1D,N))
    #var1D = np.real(np.fft.ifft(tmp_var1D,N))
   
    
    return(var1D)


##############################################################################
##############################################################################
##############################################################################


# dealias1(): function that ffts the signal, and padds it with zeros and brings it back to the real space longer
##############################################################################

def dealias1(Nx,Ny,var2D):
    "Calling the dealias1() function"
    
    import numpy as np
    #from Analysis import fourierderivative

    #Fv1 = np.fft.fft2(var2D,(Nx,Ny),axes=(0,1))
    Fv1 = np.fft.fft2(var2D,(Nx,Ny),axes=(0,1),norm='ortho')

    Kx = int(3/2*Nx)
    Ky = int(3/2*Ny)
    tmp_vpad1 = np.zeros((int(Kx),int(Ky)),dtype='complex',order='F')
    vpad1 = np.zeros((int(Kx),int(Ky)),dtype='double',order='F')

    indvpad_x = np.concatenate((np.arange(int(0),int(Nx/2)),np.arange(int(Kx-Nx/2),int(Kx))),axis=0)
    indvpad_y = np.concatenate((np.arange(int(0),int(Ny/2)),np.arange(int(Ky-Ny/2),int(Ky))),axis=0)    
    #indvpad_x = [1:Nx/2, Kx-Nx/2+1:Kx];
    #indvpad_y = [1:Ny/2, Ky-Ny/2+1:Ky];

    for i in range(0,Nx):
        for j in range(0,Ny):
            tmp_vpad1[indvpad_x[i],indvpad_y[j]] = Fv1[i,j]

    #vpad1 = np.real(np.fft.ifft2(tmp_vpad1,(Kx,Ky),axes=(0,1)))
    vpad1 = ((np.sqrt(3/2))**2)*np.real(np.fft.ifft2(tmp_vpad1,(Kx,Ky),axes=(0,1),norm='ortho'))
   
    
    return(vpad1)


##############################################################################
##############################################################################
##############################################################################



# dealias2(): function that ffts the padded signal, removes the zeros and brings it back to the real space
##############################################################################

def dealias2(Nx,Ny,var2D_padded):
    "Calling the dealias2() function"
    
    import numpy as np
    #from Analysis import fourierderivative


    Kx = int(3/2*Nx)
    Ky = int(3/2*Ny)
    
    tmp_var2D = np.zeros((Nx,Ny),dtype='complex',order='F')
    var2D = np.zeros((Nx,Ny),dtype='double',order='F')

    indvpad_x = np.concatenate((np.arange(int(0),int(Nx/2)),np.arange(int(Kx-Nx/2),int(Kx))),axis=0)
    indvpad_y = np.concatenate((np.arange(int(0),int(Ny/2)),np.arange(int(Ky-Ny/2),int(Ky))),axis=0)    
    #indvpad_x = [1:Nx/2, Kx-Nx/2+1:Kx];
    #indvpad_y = [1:Ny/2, Ky-Ny/2+1:Ky];
    
    #Fv2 = np.fft.fft2(var2D_padded,(Kx,Ky),axes=(0,1))
    Fv2 = np.fft.fft2(var2D_padded,(Kx,Ky),axes=(0,1),norm='ortho')
    
    #Fv2 = ((3/2)**2)*Fv2  #In this case it is squared because the transform is in two dimensions.

    for i in range(0,Nx):
        for j in range(0,Ny):
            tmp_var2D[i,j] = Fv2[indvpad_x[i],indvpad_y[j]]

    #var2D = np.real(np.fft.ifft2(tmp_var2D,(Nx,Ny),axes=(0,1)))
    var2D = ((1/np.sqrt(3/2))**2)*np.real(np.fft.ifft2(tmp_var2D,(Nx,Ny),axes=(0,1),norm='ortho'))
   
    
    return(var2D)


##############################################################################
##############################################################################
##############################################################################



