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
    R = np.ndarray((shape)) # Triagonal Matrix R
    
    # Calculate Orthogonal basis
    for i in range(dim):
        u[i] = a[i]
        e[i] = u[i]/np.linalg.norm(u[i])
        print("e_"+ str(i), " = ", e[i])
        for j in range(i-1):
            u[i] -= Proj(a[i],e[j])
        print("u_"+str(i), " = ", u[i])

    # Generate R
    for i in range(np.shape(u)[0]):
        for j in range(i,np.shape(u)[0]):
            R[i][j] = np.dot(e[i],a[j])

    return u, R

a = np.ndarray((3,3)) # Contains the COLUMN Vectors of an Orthogonal Matrix (T)
a[0] = 1/3*np.array([2,1,2])
a[1] = 1/3*np.array([-2,2,1])
a[2] = 1/3*np.array([1,2,-2])

u, R = GramSchmidt(a)

print("u =\n", u)
print("R =\n", np.round(R,2))
