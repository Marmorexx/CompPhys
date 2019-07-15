import numpy as np
import matplotlib.pyplot as plt

# Let s_a be a 2-Dimensional Matrix with n^2 entries

# Brings our matrix s_a into a one-dimensional form
def State(s_a):
    n = len(s_a[0])
    state = np.zeros(n**2,dtype=int)
    for i in range(n):
        for j in range(n):
            state[j+n*i] = s_a[i][j]
    return state

# Use the 1-Dim state to calculate the Energy
def Energy(B,J,state):
    return -B*np.sum(state)-J*sumindex(state)

# Sum over all 
def sumindex(state):
    n2= len(state)
    n = int(np.sqrt(len(state)))
    Sum = 0
    for i in range(n2):
        # Special case for the corners of the state matrix
        if i == 0: # Upper left corner
            Sum += state[i]*state[i+1]+state[i]*state[n]
        elif i == n-1: # Upper right corner
            Sum += state[i]*state[i-1]+state[i]*state[2*n-1]
        elif i == n2-n: # Lower left corner
            Sum += state[i]*state[i+1]+state[i]*state[n2-2*n]
        elif i == n2-1: # Lower right corner
            Sum += state[i]*state[i-1]+state[i]*state[n2-n-1]

        # Special Cases for the sides of the state matrix
        elif i < n: # Top row
            Sum += state[i]*(state[i-1]+state[i+1]+state[i+n])
        elif i >= n2-n: # Bottom Row
            Sum += state[i]*(state[i-1]+state[i+1]+state[i-n])
        elif i%n == 0: # Left column
            Sum += state[i]*(state[i+1]+state[i-n]+state[i+n])
        elif i%n == n-1: # Right column
            Sum += state[i]*(state[i-1]+state[i-n]+state[i+n])

        # All the other cases, where the atom is surrounded on all sides
        else:
            Sum+= state[i]*(state[i-1]+state[i+1]+state[i-n]+state[i+n])
    return Sum

# Calculate total magnetic moment
def M(state):
    return np.sum(state)

# Calculate total magnetic moment per atom
def m(state):
    n2 = len(state)
    return M(state)/n2

###############################################################################
### Create a test matrix s_a
s_a = np.array((
    [1,-1,1],
    [1,1,1],
    [1,1,1]))
B = 2
J = 1
###############################################################################
S = State(s_a)

print("Input State:")
print(s_a)
print("\n1-Dim form of state:")
print(S)
print("\nSummation <ab>:")
print(sumindex(S))
print("\nEnergy of state:")
print(Energy(B,J,S))
print("\nTotal magnetic moment:")
print(M(S))
print("\nTotal magnetic moment per atom:")
print(m(S))

