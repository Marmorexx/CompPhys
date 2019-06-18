import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
def polynomial(lam, b, sig, r):
    return lam**3 + (1 + b + sig)*lam**2 + b*(sig + r)*lam + 2*sig*b*(r - 1)

xaxis = np.linspace(-12.5,2.5,1000)
sig = 10
b = 8/3
for r in [0.5, 1.15, 1.3456, 1.5]:
    plt.plot(xaxis,polynomial(xaxis,b,sig,r), label='r = '+str(r))

plt.legend()
plt.show()
