import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def f(y0,x0): # y0 array that consists of [x,y,z]
    deriv = np.array([
    - sig*(y0[0] - y0[1]),
    r*y0[0] - y0[1] - y0[0]*y0[2],
    y0[0]*y0[1] - b*y0[2]])

    return deriv

def rk4_step(y0, x0, f, h, f_args = {}):
    k1 = h * f(y0, x0, **f_args)
    k2 = h * f(y0 + k1/2., x0 + h/2., **f_args)
    k3 = h * f(y0 + k2/2., x0 + h/2., **f_args)
    k4 = h * f(y0 + k3, x0 + h, **f_args)
    
    xp1 = x0 + h
    yp1 = y0 + 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)
    
    return(yp1,xp1)

def rk4(y0, x0, f, h, n, f_args = {}):
    yn = np.zeros((n+1, y0.shape[0]))
    xn = np.zeros(n+1)
    yn[0,:] = y0
    xn[0] = x0

    for n in np.arange(1,n+1,1):
        yn[n,:], xn[n] = rk4_step(y0 = yn[n-1,:], x0 = xn[n-1], f = f, h = h, f_args = f_args)

    return (yn,xn)

##### Starting Values #####
n = int(1.5e4)   # Steps
h = 0.01    # Stepsize
sig = 10.0
b = 8/3
x0 = 0
eps = 1e-3
###########################

# Solve Curve
fig = plt.figure(figsize=(8, 10))   # Define a figure
index = 0                           # and an index for arranging the plots

for r in [0.5, 1.15, 1.3456, 24.0, 28.0]: # Calculate for all r
    for pm in [1, -1]: # Calculate for C+ and C-
        index += 1

        # Define Starting points
        a0 = pm*(b*(r-1))**(0.5)
        if r < 1:
            y0 = np.array([pm*eps,pm*eps,pm*eps])
        else:
            y0 = np.array([a0,a0,(r-1)])
            y0 += eps
            print(y0)

        yn, xn = rk4(y0, x0, f, h, n)

        ax = fig.add_subplot(5,2,index, projection='3d')
        # Condition for Plots (TODO clean up later)
        if pm == 1:
            if r > 1:
                ax.plot([i[0] for i in yn], [i[1] for i in yn], [i[2] for i in yn],
                        linewidth=0.5,
                        label='r = {}, using '.format(r)+r'$c_{+}$')
                ax.scatter(y0[0], y0[1], y0[2], c='red')
                ax.scatter(y0[0]-eps, y0[1]-eps, y0[2]-eps, c='black')
                plt.legend()
            else:
                ax.plot([i[0] for i in yn], [i[1] for i in yn], [i[2] for i in yn],
                        linewidth=0.5,
                        label='r = {}, using '.format(r)+r'$+\varepsilon$')
                ax.scatter(y0[0], y0[1], y0[2], c='red')
                ax.scatter(y0[0]-eps, y0[1]-eps, y0[2]-eps, c='black')
                plt.legend()
        else:
            if r > 1:
                ax.plot([i[0] for i in yn], [i[1] for i in yn], [i[2] for i in yn],
                        linewidth=0.5,
                        label='r = {}, using '.format(r)+r'$c_{-}$')
                ax.scatter(y0[0], y0[1], y0[2], c='red')
                ax.scatter(y0[0]-eps, y0[1]-eps, y0[2]-eps, c='black')
                plt.legend()
            else:
                ax.plot([i[0] for i in yn], [i[1] for i in yn], [i[2] for i in yn],
                        linewidth=0.5,
                        label='r = {}, using '.format(r)+r'$-\varepsilon$')
                ax.scatter(y0[0], y0[1], y0[2], c='red')
                ax.scatter(y0[0]+eps, y0[1]+eps, y0[2]+eps, c='black')
                plt.legend()
        
        # This fixes the fontsize and makes the plots look less cluttered
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(5)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(5)
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(5)

plt.tight_layout()
plt.savefig('plot.pdf')
