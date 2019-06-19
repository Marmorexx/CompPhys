import numpy as np
import matplotlib.pyplot as plt

# Variables:
b = 8/3
sig = 10
plt.style.use('bmh')

#Determine Polynomial coefficients
def polyarray(b,sig,r):
    p3 = 1
    p2 = 1 + b + sig
    p1 = b*(sig + r)
    p0 = 2*sig*b*(r - 1)
    return np.array([p3,p2,p1,p0])

plt.axvspan(-15,0, alpha=0.2, color='green', label='stable')
plt.axvspan(0,5, alpha=0.2, color='red',label='unstable')
for r in [1.3456,1.5,20,24.74,28]:
    poly = polyarray(b,sig,r)
    roots = np.roots(poly)
    
    print('\nSolutions for r = '+str(r)+'\n')
    print('x1 = '+str(np.round(roots[0],2)))
    print('x2 = '+str(np.round(roots[1],2)))
    print('x3 = '+str(np.round(roots[2],2)))

    plt.scatter(roots.real, roots.imag, label='r = {}'.format(r))
plt.xlim(-15,2)
plt.legend()
plt.show()
    
    
