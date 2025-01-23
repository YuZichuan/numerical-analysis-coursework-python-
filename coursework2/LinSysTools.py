"""
MATH3036 CW2 module

@author: Zichuan Yu
"""

import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore',invalid='ignore')

def GramSchmidt(A):

    # Initialize
    Q = np.zeros(np.shape(A))

    # Number of columns
    n = np.shape(A)[1]

    # Gram-Schmidt loop
    for j in np.arange(n):
        v = A[:, j]
        for i in np.arange(j):
            r_ij = Q[:, i].T @ A[:, j]
            v = v - r_ij * Q[:, i]
        r_jj = np.linalg.norm(v)
        Q[:, j] = v / r_jj

    # Return Q
    return Q


def GramSchmidtQR(A):
    
    # Initialize
    Q = np.zeros(np.shape(A))
    
    # Number of columns
    n = np.shape(A)[1]
    
    # Initialize
    R = np.zeros([n,n])

    for j in np.arange(n):
        v = A[:, j]
        for i in np.arange(j):
            r_ij = Q[:, i].T @ A[:, j]
            v = v - r_ij * Q[:, i]
            R[i][j]=r_ij

        r_jj = np.linalg.norm(v)
        R[j][j]=r_jj
        Q[:, j] = v / r_jj



    
    # Return Q,R
    return Q,R



def ModGramSchmidtQR(A):
    
    # Initialize
    Q = np.zeros(np.shape(A))
    V = np.copy(A)
    
    # Number of columns
    n = np.shape(A)[1]
    
    # Initialize
    R = np.zeros([n,n])

    for i in np.arange(n):
        v_i = V[:,i]
        r_ii = np.linalg.norm(v_i)
        R[i][i] = r_ii
        q_i = v_i / r_ii
        Q[:,i] = q_i

        for j in np.arange(i+1,n):

            r_ij = q_i.T @ V[:,j]
            R[i][j] = r_ij
            V[:,j] = V[:,j] - r_ij*q_i

    # Return Q,R
    return Q,R


def CGmethod(A,b,x0,kmax):

    # Initialize
    x_array = np.zeros([np.shape(x0)[0], kmax + 1])

    # Store initial approximation
    x = x0
    x_array[:, [0]] = x

    # Initial r and p
    r_old = b - A @ x0
    p = r_old

    # CG loop
    for i in np.arange(kmax):
        # Step length
        a = (r_old.T @ r_old) / (p.T @ A @ p)

        # Update approximation
        x = x + a * p

        # Update residual and search direction
        r = r_old - a * A @ p
        b = r.T @ r / (r_old.T @ r_old)
        p = r + b * p

        # Update r_old
        r_old = r

        # Store
        x_array[:, [i + 1]] = x

    # Return
    return x_array




def SDmethod(A,b,x0,kmax):
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])

    for i in np.arange(kmax+1):
        x_array[:, [i]] = x0
        r = b - A @ x0
        a = (r.T @ r) / (r.T @ A @ r)
        x0 = x0 + a*r

    # Return
    return x_array


def PlotTwoMethods(A,b,x0,kmax,x_exact):
    
    k_range = np.arange(kmax+1);
    e1 = np.zeros([kmax+1,1])
    e2 = np.zeros([kmax+1,1])

    sd = SDmethod(A,b,x0,kmax)
    cg = CGmethod(A,b,x0,kmax)


    for i in np.arange(kmax+1):
        e1[i] = np.linalg.norm(x_exact-sd[:,[i]])
        e2[i] = np.linalg.norm(x_exact-cg[:,[i]])

    # Preparing figure
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("||x-x_k||")
    ax.set_title("Convergence behaviour")
    ax.grid(True)
    
    # Plot
    ax.plot(k_range , e1, "o", label="SD method",linestyle="--")
    ax.plot(k_range , e2, "d", label="CG method",linestyle="-.")
    
    # Add legend
    ax.legend();    
    
    return fig, ax




def PCGmethod(A,P,b,x0,kmax):

    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    z_array = np.zeros([np.shape(x0)[0],kmax+1])
    r_array = np.zeros([np.shape(x0)[0],kmax+1])
    
    r = b - A @ x0
    z = P @ r
    p = z

    for i in np.arange(kmax+1):
        x_array[:, [i]] = x0
        r_array[:, [i]] = r
        z_array[:, [i]] = z

        a = (r.T @ z) / (p.T @ A @ p)
        b0 = 1 / (r.T @ z)
        x0 = x0 + a*p
        r = r - a*A @ p
        z = P @ r
        b1 = b0*(r.T @ z)
        p = z + b1*p

    # Return
    return x_array, r_array, z_array


def PlotThreeMethods(A,P,b,x0,kmax,x_exact):
    
    k_range = np.arange(kmax+1);
    e1 = np.zeros([kmax+1,1])
    e2 = np.zeros([kmax+1,1])
    e3 = np.zeros([kmax+1,1])

    sd = SDmethod(A,b,x0,kmax)
    cg = CGmethod(A,b,x0,kmax)
    pcg = PCGmethod(A,P,b,x0,kmax)[0]

    for i in np.arange(kmax+1):
        e1[i] = np.linalg.norm(x_exact-sd[:,[i]])
        e2[i] = np.linalg.norm(x_exact-cg[:,[i]])
        e3[i] = np.linalg.norm(x_exact-pcg[:,[i]])

    # Preparing figure
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("||x-x_k||")
    ax.set_title("Convergence behaviour")
    ax.grid(True)
    
    # Plot
    ax.plot(k_range , e1, "o", label="SD method",linestyle="--")
    ax.plot(k_range , e2, "d", label="CG method",linestyle="-.")
    ax.plot(k_range , e3, "s", label="PCG method",linestyle="--")
    
    # Add legend
    ax.legend();    
    
    return fig, ax



def PowerMethod(A,v0,kmax):
    
    # Initialize
    v_array = np.zeros([np.shape(v0)[0],kmax])
    eigval_array = np.zeros(kmax)
    
    # Initial eigenvector 
    v = v0
    
    for k in np.arange(kmax):
        
        # Apply A
        w = A@v
        
        # Normalize
        v = w / np.linalg.norm(w,2)
        
        # Rayleigh quotient
        eigval = v.T @ A @ v
        
        # Store
        v_array[:,[k]] = v
        eigval_array[k] = eigval
        
    return eigval_array, v_array



def InverseIteration(A,v0,mu,kmax):
    
    # Initialize
    v_array = np.zeros([np.shape(v0)[0],kmax])
    eigval_array = np.zeros(kmax)

    for k in np.arange(kmax):
        w = np.linalg.solve((A-mu*np.eye(np.shape(v0)[0])),v0)
        v0 = w / np.linalg.norm(w,2)
        eigval = v0.T @ A @ v0

        v_array[:, [k]] = v0
        eigval_array[k] = eigval

    return eigval_array, v_array





def RayleighQuotientIteration(A,v0,kmax):
    
    # Initialize
    v_array = np.zeros([np.shape(v0)[0],kmax])
    eigval_array = np.zeros(kmax)
    eigval0 = v0.T @ A @ v0

    for k in np.arange(kmax):
        w = np.linalg.solve((A - eigval0*np.eye(np.shape(v0)[0])),v0)
        v0 = w / np.linalg.norm(w,2)
        eigval0 = v0.T @ A @ v0

        v_array[:, [k]] = v0
        eigval_array[k] = eigval0

    return eigval_array, v_array
