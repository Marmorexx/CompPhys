from rk4 import rk4
import numpy as np
import matplotlib.pyplot as plt

#Vars
G = 1
M = 1
h = 1
n = 10

#Init
y0 = np.array([-0.97000436, 0.24308753])
v0 = np.array([-0.46620368, -0.43236573])
y1 = np.array([0.97000436, -0.24308753])
v1 = np.array([-0.46620368, -0.43236573])
y2 = np.array([0., 0.])
v2 = np.array([0.93240737, 0.86473146])

liste = (y0, v0, y1, v1, y2, v2)

yall = np.ndarray(shape=(6,2))
for i in range(yall.shape[0]):
    yall[i] = liste[i]

x0 = np.array([0])

dist = lambda x1, x2:  x2 - x1

def f(y, x):
    out = np.zeros(y.shape)
    for idx in range(y.shape[0]):
        if (idx % 2 == 0):
            for i in range(y.shape[0]/2):
                if (idx != i*2):
                    out[idx+1] = out[idx+1] + G * M * dist(y[idx], y[i*2]) / np.linalg.norm(dist(y[idx], y[i*2]))**3
        if (idx % 2 == 1):
            out[idx-1] = out[idx]

    return out 

y, x = rk4(y0, x0, f, h, n)


plt.figure(figsize=(16, 4))

plt.subplot(131)
plt.title("Numerical and Analytical Solution")
plt.plot(x, y)

plt.savefig("fig2.pdf")

