import numpy as np
import matplotlib.pyplot as plt

# Global Variables
xaxis = np.zeros(10000)
yaxis = np.zeros(10000)
for epsilon in range(10000):

    def numerov(x0,y_n,h,n):
        x_n = np.zeros(n)
        x_n[0] = x0
        x_n[1] = x_n[0] + h
        output = np.zeros(n)
        output[0] = y_n[0] # = y_(n-1)
        output[1] = y_n[1] # = y_n
        
        for i in range(2, n):
            x_n[i] = x_n[i-1] + h

            y_n[2] = 2 * (1 - 5/12 * h**2 * k(x_n[i-1])) * y_n[1] 
            y_n[2] -= (1 + 1/12 * h**2 * k(x_n[i-2])) * y_n[0]
            y_n[2] /= (1 + 1/12 * h**2 * k(x_n[i]))

            output[i] = y_n[2]
            y_n[0] = y_n[1]
            y_n[1] = y_n[2]

        return (x_n,output)

    def k(x):
        return 2*epsilon-x**2

    #def initiate(x0,y0,h
    #def schroedinger(psi, E, V, 

    n = 100
    h = 0.1
    a = 0.5
    x0 = -5

    y = np.ndarray((3))
    y[0] = 0
    y[1] = a

    print("integrating from 0 to " + str(n*h) + " in steps of " + str(h))

    x, lel = numerov(x0,y,h,n)
    
    if (np.max(np.abs(lel)) <= 1e2):

        xaxis[epsilon] = epsilon
        yaxis[epsilon] = np.max(np.abs(lel))
    #plt.plot(x,lel)
plt.plot(xaxis, yaxis)
plt.yscale('log')
plt.xlabel(r"$\varepsilon$")
plt.ylabel('Max Value in output Array')
plt.savefig('epsilon.pdf')
plt.show()
