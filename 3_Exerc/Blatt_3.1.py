import numpy as np
import matplotlib.pyplot as plt

#global vars
e = 100

def numerov(n, x_0, y_n, h, k):
    output = np.zeros(n)
    x_n = np.zeros(n)
    output[0] = y_n[0]  #y_n[0] entspricht y_n-1
    output[1] = y_n[1]  #y_n[1] entspricht y_n
    x_n[0] = x_0
    x_n[1] = x_0 + h

    for i in range(2, n):
        x_n[i] = x_n[i-1] + h

        y_n[2] = 2 * (1 - 5/12 * h**2 * k(x_n[i-1])) * y_n[1] 
        y_n[2] -= (1 + 1/12 * h**2 * k(x_n[i-2])) * y_n[0]
        y_n[2] /= (1 + 1/12 * h**2 * k(x_n[i]))

        print (y)

        output[i] = y_n[2]
        y_n[0] = y_n[1]
        y_n[1] = y_n[2]
    return x_n, output

def k_x(x):
    return 2*e - x**2

#integration variables
n_int = 1000
h_int = 0.01
print ("Integrating from 0 to " + str(n_int*h_int) + " in steps of " + str(h_int) +"!")

#start values
a = 1

y = np.ndarray((3))
y[0] = 0
y[1] = a

x, lel = numerov(n_int, 0, y, h_int, k_x)

plt.plot(x, lel)
plt.show()
