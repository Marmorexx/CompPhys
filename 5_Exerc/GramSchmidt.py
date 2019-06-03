import numpy as np
import matplotlib.pyplot as plt

# We use np.linalg.norm(x) to Norm stuff

# Define Projection Operator

def Proj(v,e):
    return np.dot(e,v)/np.dot(e,e)*e

def GramSchmidt(a): # a is the input matrix (consisting of COLUMN VECTORS!!!)
    dim = np.shape(a)[0] # Dimension of Vector
    shape = np.shape(a)
    u = np.zeros((shape)) # Vectors of orthogonal base
    e = np.zeros((shape)) # Unit Vectors for our calculation
    R = np.zeros((shape)) # Triagonal Matrix R

    # Calculate Orthogonal basis
    for i in range(dim):
        u[i] = a[i]
        e[i] = u[i]/np.linalg.norm(u[i])
        print("e_"+ str(i), " = ", e[i])
        for j in range(i):
            u[i] -= Proj(a[i],e[j])
        print("u_"+str(i), " = ", u[i])

    # Generate R
    for i in range(np.shape(u)[0]):
        for j in range(i,np.shape(u)[0]):
            R[i][j] = np.dot(e[i],a[j])

    return e, R

########################## Enter Matrix here ###################################

a = np.ndarray((3,3)) # Contains the COLUMN Vectors of an Orthogonal Matrix (T)
a[0] = np.array([2,1,2])
a[1] = np.array([-2,2,1])
a[2] = np.array([1,2,-2])

########################## Enter Matrix here ###################################

e, R = GramSchmidt(a)

print("a =\n", a)
print("e =\n", e)
print("R =\n", np.round(R,2))

test = np.dot(e,e.T)

print("test =\n", test)
