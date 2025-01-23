"""
MATH3036 CW1 FEMtools module

@author: Zichuan Yu
"""

import numpy as np



def AssembleFEMmatrix1(N):
    """
    Returns a numpy array of the 
    Petrov-Galerkin FEM matrix for the bilinear form
       b(u,v) := \int_0^1 u' v dx 
    using FEM trial space 
       Uh := continuous piecewise linears, and being = 0 for x=0, 
    with basis { phi_j } being the hat functions (node-wise), 
    and using the FEM test space
       Vh := discontinuous piecewise constants
    with basis { psi_i } being indicator function (element-wise),
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
        
    Returns
    -------
    A : numpy.ndarray, shape (N,N)
        Array containing the Petrov-Galerkin FEM matrix.
    """

    # Mesh
    h = 1/N;
    x  = h*np.arange(1,N+1,1)


    # Assemble matrix
    A = np.diag(np.ones(N))         \
      - np.diag(np.ones(N-1),-1)
        
    return A


def AssembleFEMvector1(N,f):
    """
    Returns a numpy array of the 
    FEM vector for the linear form
       l(v) := \int_0^1 f v dx 
    using FEM test space
       Vh := discontinuous piecewise constants
    with basis { psi_i } being indicator function (element-wise),
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
    f : function
        Input function.
        
    Returns
    -------
    b : numpy.ndarray, shape (N,)
        Array containing the FEM vector.
    """
    
    # Mesh mid points (center of each element)
    h = 1/N;
    xc = h*np.arange(N) + h/2

    # Assemble vector (using mid-point rule)
    b = h*f(xc)
    
    return b


def AssembleFEMmatrix2(N):
    # Assemble matrix (coincidentally(!) the same as AssembleFEMmatrix1)
    A = np.diag(np.ones(N)) \
        - np.diag(np.ones(N - 1), -1)

    return A


def AssembleFEMvector2(N,f):
    # Mesh mid points (center of each element)
    h = 1/N;
    xc_l = h*np.arange(0,N-1) + h/2   # All midpoints, except right-most
    xc   = h*np.arange(0,N)   + h/2   # All midpoints

    # Assemble vector (using mid-point rule)
    b = h*1/2*( f(xc) + np.r_[0,f(xc_l)] )
    
    return b


def AssembleFEMmatrix3(N):
    """
    Returns a numpy array of the
    Galerkin FEM matrix for the bilinear form
        b(u,v) := \int_0^1 u' v' dx
    using the FEM trial space Uh and the FEM test space Vh
    (in this case, Uh = Vh)
        Vh := continuous piecewise linears, equal to 0 for x=0 and x=1,
    with basis { phi_j } being hat functions (node-wise),
    for a uniform mesh with N equal-sized elements of width h=1/N and
    N+1 grid-nodes.

    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
    
    Returns
    -------
    A: numpy.ndarray, shape (N-1,N-1)
       Array containing the Galerkin FEM matrix.
    """

    A1 = 2 * np.diag(np.ones(N - 1)) \
         - np.diag(np.ones(N - 2), -1) \
         - np.diag(np.ones(N - 2), 1)

    # Assemble FEM matrix
    A = N*A1

    return A


def AssembleFEMvector3(N,f):
    """
    Returns a numpy array of the
    FEM vector for the linear form
        l(v) := \int_0^1 f v dx
    using the FEM test space
        Vh := continuous piecewise linears, equal to 0 for x=0 and x=1,
    with basis { phi_i } being hat functions (node-wise),
    for a uniform mesh with N equal-sized elements of width h=1/N and
    N+1 grid-nodes.

    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
    f : function
        Input function.
    
    Returns
    -------
    b: numpy.array, shape (N-1,)
       Array containing the FEM vector.
    """

    # Mesh
    h = 1/N
    x_i = np.arange(h, 1, h)

    x2 = np.arange(0, 1-h, h)
    x_im = 0.5*(x2+x_i)

    x3 = np.arange(2*h, 1+h, h)
    x_ip = 0.5*(x_i+x3)

    # Assemble FEM vector (using (composite) Simpson rule)
    b = (h/3)*(f(x_im)+f(x_i)+f(x_ip))

    return b


def AssembleFEMmatrix4(N,c):
    A1 = 2 * np.diag(np.ones(N - 1)) \
         - np.diag(np.ones(N - 2), -1) \
         - np.diag(np.ones(N - 2), 1)

    A2 = 4 * np.diag(np.ones(N - 1)) \
         + np.diag(np.ones(N - 2), -1) \
         + np.diag(np.ones(N - 2), 1)

    # Assemble FEM matrix
    A = N*A1 + N*(c/6)*A2

    return A


def Interpolate(N,f):
    # Mesh
    h = 1/N
    x_i = np.arange(h, 1, h)

    # Method of interpolation
    f_vec = f(x_i)

    return f_vec


def AssembleFEMmassMatrix3(N):
    h = 1/N
    M1 = 4 * np.diag(np.ones(N - 1)) \
         + np.diag(np.ones(N - 2), -1) \
         + np.diag(np.ones(N - 2), 1)

    # Assemble FEM mass matrix M
    M = (h/6)*M1

    return M


def HeatEqFEM(tau,Ntime,N,u0):    
    u_array = np.zeros((Ntime+1, N-1))

    # Matrix K
    K = AssembleFEMmatrix3(N)

    # Matrix A := M + tau*K
    A = AssembleFEMmassMatrix3(N) + tau*K

    # Obtain the initial value of the coefficients vector u^k using the method of interpolation
    u0_vec = Interpolate(N,u0)

    # Backward Eular discretization in time & Galerkin FEM in space
    for k in range(Ntime+1):
        u_array[k] = u0_vec
        b = AssembleFEMmassMatrix3(N)@u0_vec
        u0_vec = np.linalg.solve(A,b)

    return u_array


def Assemble2dFEMvector(N):
    h = 1/N

    # Assemble 2D FEM vector
    b = (h**2)*np.ones((N-1)**2)

    return b


def Assemble2dFEMmatrixN4():
    N = 4

    # Construct the diagonal-element vector with only 4s
    v1 = 4*np.ones((N-1)**2)

    # Construct the diagonal-element vector with -1s and 0s
    v2 = -np.ones((N-1)**2-1)
    i = 1
    while (N-1)*i <= ((N-1)**2-1):
        v2[(N-1)*i-1]=0
        i=i+1

    # Construct the diagonal-element vector with only -1s
    v3 = -np.ones((N-2)*(N-1))

    # Construct the corresponding diagonal (diagonal-like) matrices
    A1 = np.diag(v1)
    A2 = np.diag(v2,-1)
    A3 = np.diag(v2,1)
    A4 = np.diag(v3,N-1)
    A5 = np.diag(v3,1-N)

    # Assemble 2D FEM matrix when N=4
    A = A1 + A2 + A3 + A4 + A5

    return A
    

def Assemble2dFEMmatrix(N):
    """
    This matrix has three types of special diagonals which are different from zero, hence, the method below is to
    firstly construct these five diagonal-element vectors respectively, then form the corresponding
    diagonal (diagonal-like) matrices, and finally add them together to form the assemble 2D FEM matrix.
    """

    # Construct the diagonal-element vector with only 4s
    v1 = 4*np.ones((N-1)**2)

    # Construct the diagonal-element vector with -1s and 0s
    v2 = -np.ones((N-1)**2-1)
    i = 1
    while (N-1)*i <= ((N-1)**2-1):
        v2[(N-1)*i-1]=0
        i=i+1

    # Construct the diagonal-element vector with only -1s
    v3 = -np.ones((N-2)*(N-1))

    # Construct the corresponding diagonal (diagonal-like) matrices
    A1 = np.diag(v1)
    A2 = np.diag(v2,-1)
    A3 = np.diag(v2,1)
    A4 = np.diag(v3,N-1)
    A5 = np.diag(v3,1-N)

    # Assemble 2D FEM matrix
    A = A1 + A2 + A3 + A4 + A5

    return A
