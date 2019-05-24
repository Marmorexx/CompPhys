import numpy as np
import matplotlib.pyplot as plt
import sys

a = int(sys.argv[1])

n1 = int(sys.argv[2])

n2 = int(sys.argv[3])

start = np.log((a+1)/a)

#Rekursion - geht bei hohen n (>21) kaputt
def yn(a_var, n_min, n_max):
    if n_max == 0:
        return start
    if n_max == n_min:
        return start
    else:
        return 1/n_max - a_var*yn(a_var,n_min,n_max-1)

def zero(x):
    return x*0

y = ([])
for j in range(n1,n2+1):
    y.append(yn(a,n1,j))

xaxis = np.linspace(n1,n2,(n2-n1+1))
plt.plot(xaxis, y)
plt.plot(xaxis, zero(xaxis), linestyle='--', linewidth=1)
plt.xlabel('n_max')
plt.ylabel('Integral')
plt.savefig('iteration.pdf')
