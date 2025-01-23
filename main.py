"""
MATH3036 Coursework 1 main script

@author: Zichuan Yu

version: 2 Feb 2023, 9 Feb 2023
"""



#%% Question 1 (Compute FEM approximation)

import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Set mesh parameter (N+1 is number of nodes)
N = 4;

# Set problem data
def f(x):
    return np.pi * np.cos(np.pi*x)

# Assemble system
A = FEM.AssembleFEMmatrix1(N)
b = FEM.AssembleFEMvector1(N,f)

print("A =",A)
print("b =",b)

# Solve system
u = np.linalg.solve(A,b)
print("u =",u)


#%% Question 1 (Plot FEM approximation and exact solution)

# Add boundary condition u(0) =  0
u_all = np.r_[0,u]
h = 1/N;
x_all = h*np.arange(0,N+1,1)

# Plot FEM approximation  
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(x_all,u_all,'bo-')

# Exact solutuion
def u_ex(x):
    return np.sin(np.pi*x)

# Plot exact solution
N_p = 100
x_p = 1/N_p * np.arange(0,N_p+1,1)
u_ex_p = u_ex(x_p)
ax.plot(x_p,u_ex_p,'r-')



#%% Question 2 (Compute FEM approximation)

import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Set mesh parameter (N+1 is number of nodes)
N = 4;

# Set problem data
def f(x):
    return np.pi * np.cos(np.pi*x)

# Assemble system
A = FEM.AssembleFEMmatrix2(N)
b = FEM.AssembleFEMvector2(N,f)

print("A =",A)
print("b =",b)

# Solve system
u = np.linalg.solve(A,b)
print("u =",u)


#%% Question 2 (Plot FEM approximation and exact solution)

# u is piecewise constant (per element)
u_elem = u
h = 1/N;
x = h*np.arange(0,N+1)  # all nodes
x_l = h*np.arange(0,N-1,1)  # left nodes for each element
x_r = h*np.arange(1,N  ,1)  # right nodes for each element


# Plot FEM approximation  
fig, ax = plt.subplots()  # Create a figure containing a single axes.
for i in np.arange(N):
    ax.plot([x[i],x[i+1]],[u_elem[i],u_elem[i]],'bo-')

# Exact solutuion
def u_ex(x):
    return np.sin(np.pi*x)

# Plot exact solution
N_p = 100
x_p = 1/N_p * np.arange(0,N_p+1,1)
u_ex_p = u_ex(x_p)
ax.plot(x_p,u_ex_p,'r-')




#%% Question 3, (Compute FEM approximation)

import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Set mesh parameter (N+1 is number of nodes)
N = 4;

# Set problem data
def f(x):
    return np.pi**2 * np.sin(np.pi*x)

# Assemble system
A = FEM.AssembleFEMmatrix3(N)
b = FEM.AssembleFEMvector3(N,f)

print("A =",A)
print("b =",b)

# Solve system
u = np.linalg.solve(A,b)
print("u =",u)


#%% Question 3 (Plot FEM approximation and exact solution)

# Add boundary condition u(0) =  0, u(1) = 0
u_all = np.r_[0,u,0]
h = 1/N;
x_all = h*np.arange(0,N+1,1)

# Plot FEM approximation  
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(x_all,u_all,'bo-')

# Exact solutuion
def u_ex(x):
    return np.sin(np.pi*x)

# Plot exact solution
N_p = 100
x_p = 1/N_p * np.arange(0,N_p+1,1)
u_ex_p = u_ex(x_p)
ax.plot(x_p,u_ex_p,'r-')



#%% Question 4, (Compute FEM matrix)

import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Set mesh parameter (N+1 is number of nodes)
N = 4;

# Set coefficient
c = 2;

# Assemble system
A = FEM.AssembleFEMmatrix4(N,c)
print("A =",A)



#%% Question 5

import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Set mesh parameter 
N = 8
tau = 0.02
Ntime = 16

# Set initial condition function
def u0(x):
    return np.sin(np.pi*x) + np.sin(2*np.pi*x)


# Interpolation of u0
u0_vec = FEM.Interpolate(N,u0)
print("u0_vec =",u0_vec)


# Mass matrix 
M = FEM.AssembleFEMmassMatrix3(N)
print("M =",M)


# Solve heat equation
u_array = FEM.HeatEqFEM(tau,Ntime,N,u0)
print("u_array =",u_array)


# Plot approximations
fig, ax = plt.subplots()  # Create a figure containing a single axes.
h = 1/N
x_all = h*np.arange(0,N+1,1)
for k in np.arange(Ntime+1):
    uk = u_array[k,:]
    uk_all = np.r_[0,uk,0]
    ax.plot(x_all,uk_all,'o-')

plt.show()


#%% Question 6  (Compute the FEM system for N = 4)


import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM

# Assemble A when N=4
A = FEM.Assemble2dFEMmatrixN4()
print("A = ",A)

# Assemble b when N=4
N = 4
b = FEM.Assemble2dFEMvector(N)
print("b =",b)

# Solve system
u = np.linalg.solve(A,b)
print("u =",u)


#%% Question 7 (Compute the FEM system for arbitrary N)


import numpy as np
import matplotlib.pyplot as plt
import FEMtools as FEM


N = 8
A = FEM.Assemble2dFEMmatrix(N)
print("A = ",A)

b = FEM.Assemble2dFEMvector(N)
print("b =",b)

# Solve system
u = np.linalg.solve(A,b)
print("u =",u)

#%% Question 7 continued (Plot the approximation using a scatter plot)

# Create all interior nodes
h = 1/N
x = h*np.linspace(1, N-1, N-1)
y = h*np.linspace(1, N-1, N-1)
xx, yy = np.meshgrid(x, y)

# Reshape the vector u into an array of size (N-1)x(N-1))
zz = np.reshape(u,(N-1,N-1))

# Plot the FEM approximation u
fig = plt.figure()  
ax = fig.add_subplot(projection='3d')
ax.scatter(xx, yy, zz,c=zz,s=40)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([min([0,min(u)]),max([0.075,max(u)])])

plt.show()


