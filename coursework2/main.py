"""
MATH3036 Coursework 2 main script

@author: Zichuan Yu

version: 8 Mar 2023
"""



#%% Question 1 

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1, 4, 0],[np.sqrt(2), np.sqrt(2), 0],[0,np.sqrt(2),1]])
print("A =\n",A)

Q = LST.GramSchmidt(A)
print("Q =\n",Q)



#%% Question 2

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[2, -1],[-1, 2]])
b = np.array([[0],[3]])
x0 = np.array([[0],[0]])
kmax = 2

x_array = LST.CGmethod(A,b,x0,kmax)
print("x_array =\n",x_array)




#%% Question 3 (a) Square system

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1, 4, 0],[np.sqrt(2), np.sqrt(2), 0],[0,np.sqrt(2),1]])
b = np.array([[np.sqrt(3)],[np.sqrt(6)],[2]])
print("A =\n",A)
print("b =\n",b)

Q,R = LST.GramSchmidtQR(A)
print("Q =\n",Q)
print("R =\n",R)

x = spla.solve_triangular(R,Q.T@b)
print("x =\n",x)


#%% Question 3 (b) Overdetermined system


import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1., 0],[1, 1],[1, 2]])
b = np.array([[2.],[18],[40]])
print("A =\n",A)
print("b =\n",b)

Q,R = LST.GramSchmidtQR(A)
print("Q =\n",Q)
print("R =\n",R)

x = spla.solve_triangular(R,Q.T@b)
print("x =\n",x)



#%% Question 4 (a) Square system

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1, 4, 0],[np.sqrt(2), np.sqrt(2), 0],[0,np.sqrt(2),1]])
b = np.array([[np.sqrt(3)],[np.sqrt(6)],[2]])
print("A =\n",A)
print("b =\n",b)

Q,R = LST.ModGramSchmidtQR(A)
print("Q =\n",Q)
print("R =\n",R)

x = spla.solve_triangular(R,Q.T@b)
print("x =\n",x)


#%% Question 4 (b) Overdetermined system


import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1., 0],[1, 1],[1, 2]])
b = np.array([[2.],[18],[40]])
print("A =\n",A)
print("b =\n",b)

Q,R = LST.ModGramSchmidtQR(A)
print("Q =\n",Q)
print("R =\n",R)

x = spla.solve_triangular(R,Q.T@b)
print("x =\n",x)


#%% Question 4 (c) Checking orthogonality of vectors q_i in Q

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

# Set size of system
N = 32

# Set Random Generator Seed
prng = np.random.RandomState(1234567890)

# Create random system with widely varying singular values
U,X = np.linalg.qr(prng.rand(N,N))
V,X = np.linalg.qr(prng.rand(N,N))
S = np.diag(2.**(-1*np.arange(N)))
A = U@S@V

# Construct QR factorizations
Q1,R1 = LST.GramSchmidtQR(A)
Q2,R2 = LST.ModGramSchmidtQR(A)

# Check orthogonality of q_i and q_{i+1}
print("Classical Gram-Schmidt Q check:\n",np.diag(Q1.T@Q1,-1))
print("\nModified Gram-Schmidt Q check:\n",np.diag(Q2.T@Q2,-1))


#%% Question 5  2x2 system

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[2, -1],[-1, 2]])
b = np.array([[0],[3]])
x0 = np.array([[0],[0]])
kmax = 6

x_array = LST.SDmethod(A,b,x0,kmax)
print("x_array =\n",x_array)


#%% Question 6  nxn system


import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

n = 100
d = 0.5 + np.sqrt(1+np.arange(n))
A = np.diag(d) \
    + np.diag(np.ones(n-1), 1) \
    + np.diag(np.ones(n-1),-1) \
    + np.diag(np.ones(round(n/2)-1), round(n/2)+1) \
    + np.diag(np.ones(round(n/2)-1),-round(n/2)-1)
b = np.ones([n,1])
x0 = np.zeros([n,1])
kmax = 25
x_exact = np.linalg.solve(A,b)

LST.PlotTwoMethods(A,b,x0,kmax,x_exact)
plt.show()

#%% Question 7  2x2 system

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[2, -1],[-1, 2]])
P = np.array([[1/2, 0 ],[0, 1/2]])
b = np.array([[0],[3]])
x0 = np.array([[0],[0]])
kmax = 2

x_array,r_array,z_array = LST.PCGmethod(A,P,b,x0,kmax)
print("x_array =\n",x_array)
print("r_array =\n",r_array)
print("z_array =\n",z_array)

#%% Question 8  nxn system


import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

n = 100
d = 0.5 + np.sqrt(1+np.arange(n))
A = np.diag(d) \
    + np.diag(np.ones(n-1), 1) \
    + np.diag(np.ones(n-1),-1) \
    + np.diag(np.ones(round(n/2)-1), round(n/2)+1) \
    + np.diag(np.ones(round(n/2)-1),-round(n/2)-1)
P = np.diag(1/d)
b = np.ones([n,1])
x0 = np.zeros([n,1])
kmax = 25
x_exact = np.linalg.solve(A,b)

LST.PlotThreeMethods(A,P,b,x0,kmax,x_exact)
plt.show()



#%% Question 9 


import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1, 3],[2, 2]])
v0 = np.array([[1],[0]])
kmax = 6
eigval_array, v_array = LST.PowerMethod(A,v0,kmax)
print("\nPower Method")
print("eigval_array =\n",eigval_array)
print("v_array =\n",v_array)

mu = -2
eigval_array, v_array = LST.InverseIteration(A,v0,mu,kmax)
print("\nInverse Iteration")
print("eigval_array =\n",eigval_array)
print("v_array =\n",v_array)

#%% Question 10 


import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

eigval_array, v_array = LST.RayleighQuotientIteration(A,v0,kmax)
print("\nRayleigh Quotient Iteration")
print("eigval_array =\n",eigval_array)
print("v_array =\n",v_array)
