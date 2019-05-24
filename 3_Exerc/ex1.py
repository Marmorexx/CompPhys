# In collaboration with Group of Jan Hubrich
import numpy as np
import matplotlib.pyplot as plt

def numerov(x0,y_n,h,n,epsilon):
    x_n = np.zeros(n)
    x_n[0] = x0
    x_n[1] = x_n[0] + h
    output = np.zeros(n)
    output[0] = y_n[0] # = y_(n-1)
    output[1] = y_n[1] # = y_n
    
    for i in range(2, n):
        x_n[i] = x_n[i-1] + h

        y_n[2] = 2 * (1 - 5/12 * h**2 * k(x_n[i-1],epsilon)) * y_n[1] 
        y_n[2] -= (1 + 1/12 * h**2 * k(x_n[i-2],epsilon)) * y_n[0]
        y_n[2] /= (1 + 1/12 * h**2 * k(x_n[i],epsilon))

        output[i] = y_n[2]
        y_n[0] = y_n[1]
        y_n[1] = y_n[2]

    return (x_n,output)

def k(x,epsilon):
    return 2*epsilon-x**2

def H(x,n):
    if (n==0): return 1
    elif (n==1): return 2*x
    else: return 2*x*H(x,n-1)-2*(n-1)*H(x,n-2)

def factorial(n):
    if (n==0): return 1
    else: return n*factorial(n-1)

def analytical(x,n):
    return H(x,n)/(2**n*factorial(n)*np.sqrt(np.pi))**0.5*np.exp(-x**2/2)
stepsize = 0.001
steps = int(5/stepsize)
x0 = 0
a = 0.6

print("integrating from 0 to " + str(steps*stepsize) + " in steps of " + str(stepsize))

for n in range(5):
    epsilon = n+1/2

    xaxis = np.linspace(0,5,1000)

    if (n%2 == 0):
        y_init=np.ndarray((3))
        #y_init[0] = analytical(0, n) #use anal. initial
        if (n%4 == 0): y_init[0] = a
        else: y_init[0] = -a
        y_init[1] = y_init[0] - stepsize**2 * k(x0, epsilon) * y_init[0]/2
        x_vector, y_vector = numerov(x0,y_init,stepsize,steps,epsilon)

        plt.plot(xaxis, analytical(xaxis,n), label="n = "+str(n))
        plt.plot(x_vector,y_vector,label="Numerov n ="+str(n))
        print(analytical(stepsize,n))
    else:
        y_init=np.ndarray((3))
        y_init[0] = analytical(0,n)
        y_init[1] = analytical(stepsize,n) #Use anal. initial
        #if ((n-1)%4 == 0): y_init[1] = 0.001 
        #else: y_init[1] = -0.001
        x_vector, y_vector = numerov(x0,y_init,stepsize,steps,epsilon)

        plt.plot(xaxis, analytical(xaxis,n), label="n = "+str(n))
        plt.plot(x_vector,y_vector,label="Numerov n ="+str(n))
        print(analytical(stepsize,n))

plt.xlabel('x')
plt.ylabel('E')
plt.legend()
plt.show()
