import numpy as np
import matplotlib.pyplot as plt

# We use np.linalg.norm(x) to Norm stuff

# Define Projection Operator

def Proj(v,e):
    return np.dot(e,v)/np.dot(e,e)*e

def GramSchmidt(a): # a is the input matrix (consisting of COLUMN VECTORS!!!)
    dim = np.shape(a)[0] # Dimension of Vector
    shape = np.shape(a)
    u = np.ndarray((shape)) # Vectors of orthogonal base
    e = np.ndarray((shape)) # Unit Vectors for our calculation
    
    for i in range(dim):
        u[i] = a[i]
        e[i] = u[i]/np.linalg.norm(u[i])
        print("e_", i, " = ", e[i])
        for j in range(i-1):
            u[i] -= Proj(a[i],e[j])
        print("u_", i, " = ", u[i])

    return u

a = np.ndarray((3,3)) # Contains the COLUMN Vectors of an Orthogonal Matrix (T)
a[0] = 1/3*np.array([2,1,2])
a[1] = 1/3*np.array([-2,2,1])
a[2] = 1/3*np.array([1,2,-2])

u = GramSchmidt(a)
print(u)
