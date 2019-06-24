import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def checkz(array):
    initial = array[0][2]
    up = True
    maxima = ([])
    for n in range(array.shape[0]-1):
        if up == True:
            if array[n][2] > array[n+1][2]:
                up = False
                maxima.append(array[n][2])
        if up == False:
            if array[n][2] < array[n+1][2]:
                up = True
    return maxima

def linear(x,m,b):
    return m*x+b

print('Initiating ...')
##### Starting Values #####
n = 15000   # Steps
h = 0.01    # Stepsize
r = 27
sig = 10.0
b = 8/3
x0 = 0
eps = 1e-3
###########################

# Solve Curve
plt.xkcd(2,2,3)
fig = plt.figure(figsize=(8, 4))    # Define a figure
index = 0                           # and an index for arranging the plots

for pm in [1, -1]: # Calculate for C+ and C-
    index += 1

    # Define Starting points
    a0 = pm*(b*(r-1))**(0.5)
    y0 = np.array([a0,a0,(r-1)])
    y0 += eps

    # Calculate Trajectories and Maxima of z
    print('Drawing Trajectory...')
    yn, xn = rk4(y0, x0, f, h, n)
    print('Searching for maximum z...')
    maxima = checkz(yn)
    print('Found {} Maxima'.format(len(maxima)))

    ax = fig.add_subplot(1,2,index)
    if pm == 1:
        print(r'Creating plot for $C_+$...')
        ax.plot(maxima[:-1], maxima[1:],
                s=0.5,
                label='r = {}, using '.format(r)+r'$c_{+}$')
        popt, pcov = curve_fit(linear, maxima[:-190], maxima[1:-189])
        xaxis = np.linspace(26,40,100)
        ax.plot(xaxis, linear(xaxis, popt[0], popt[1]),
                linewidth=0.9,
                color='red',
                label='m = {}'.format(popt[0]))
        plt.legend()
    else:
        print(r'Creating plot for $C_-$...')
        ax.plot(maxima[:-1], maxima[1:],
                s=0.5,
                label='r = {}, using '.format(r)+r'$c_{-}$')
        popt, pcov = curve_fit(linear, maxima[:-190], maxima[1:-189])
        xaxis = np.linspace(26,40,100)


        ax.plot(xaxis, linear(xaxis, popt[0], popt[1]),
                linewidth=0.9,
                color='red',
                label='m = {}'.format(popt[0]))
        plt.legend()
plt.savefig('plot.pdf')
plt.show()
