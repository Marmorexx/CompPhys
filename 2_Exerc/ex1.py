import numpy as np
import matplotlib.pyplot as plt

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
        
    return(yn, xn)

def exponential_decay(x,t,r=1):
    return -r*x

def analytical_solution(t,r=1):
    return np.exp(-r*t)

# set starting values
x0 = 0
y0 = np.array([1]) #has to be array
h = 0.1
n = 100

# Plot exponential decay
plt.figure()
y_vector, x_vector = rk4(y0, x0, exponential_decay, h, n)
plt.plot(x_vector, y_vector, label='Numerical Solution')
plt.plot(x_vector, analytical_solution(x_vector), label='Analytical solution')
plt.xlabel('Time')
plt.legend()
plt.savefig('1_h_eq_0p1.pdf')
plt.show

# Be advised that the integration can take a while for large values of n (e.g >=10^5).
