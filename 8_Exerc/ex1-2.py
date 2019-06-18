import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
# Variables:
b = 8/3
sig = 10

#Determine Polynomial coefficients
def polyarray(b,sig,r):
    p3 = 1
    p2 = 1 + b + sig
    p1 = b*(sig + r)
    p0 = 2*sig*b*(r - 1)
    return np.array([p3,p2,p1,p0])

for r in [1.3456,1.5,24,28]:
    poly = polyarray(b,sig,r)
    roots = np.roots(poly)
    
    print('\nSolutions for r = '+str(r)+'\n')
    print('x1 = '+str(np.round(roots[0],2)))
    print('x2 = '+str(np.round(roots[1],2)))
    print('x3 = '+str(np.round(roots[2],2)))

    plt.scatter(roots.real, roots.imag, label='r = {}'.format(r))
plt.legend()
plt.show()
    
    
