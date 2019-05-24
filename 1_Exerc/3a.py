import numpy as np
import matplotlib.pyplot as plt

# Plot of integrand
def integrand(x_var,n_var,a_var):
    return x_var**n_var/(x_var+a_var)

xaxis = np.linspace(0,1,100)

n_a = np.array([1,5,10,20,30,50])

for i in range(0,len(n_a)-1):
    plt.plot(xaxis, integrand(xaxis,n_a[i],a))
plt.savefig('integrand.pdf')
