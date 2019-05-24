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

#a)######################################################################################################################

#function for the 3 masses with place and velocity each in 2d -> y.shape = 12dim
def fa(y, x, m):
    out = np.zeros(y.shape)
    for idx in np.arange(0,len(y),4):
            # 0 ist x_0
            # 1 ist y_0
            # 2 ist vx_0
            # 3 ist vy_0
            # dr/dt = v
            out[idx] = y[idx + 2]  # "f(x) = vx"
            out[idx + 1] = y[idx + 3]  # "f(y) = vy"
            # loop to sum over all forces
            for jdx in np.arange(0, len(y), 4):
                if (idx != jdx):
                    # dv/dt = a
                    out[idx + 2] = out[idx + 2] + G * m[(int)(jdx / 4)] * (y[jdx] - y[idx]) / ((y[jdx] - y[idx]) ** 2 + (
                    y[jdx + 1] - y[idx + 1]) ** 2) ** 1.5
                    out[idx + 3] = out[idx + 3] + G * m[(int)(jdx / 4)] * (y[jdx + 1] - y[idx + 1]) / ((y[jdx] - y[
                        idx]) ** 2 + (y[jdx + 1] - y[idx + 1]) ** 2) ** 1.5

    return out

def setup(stepsize, iterations, init_pos, masses):
    y, x = init_pos
    y, x = rk4(y, x, fa, stepsize, iterations, {"m" : masses})
    
    out_x0 = np.zeros(x.shape)
    out_y0 = np.zeros(x.shape)
    out_x1 = np.zeros(x.shape)
    out_y1 = np.zeros(x.shape)
    out_x2 = np.zeros(x.shape)
    out_y2 = np.zeros(x.shape)

    for idx, i in enumerate(y):
        out_x0[idx] = i[0]
        out_y0[idx] = i[1]
        out_x1[idx] = i[4]
        out_y1[idx] = i[5]
        out_x2[idx] = i[8]
        out_y2[idx] = i[9]

    return ((out_x0, out_y0), (out_x1, out_y1), (out_x2, out_y2))

#Vars
G = 1
M = 1
x0 = np.array([0]) #time begins at t = 0

#init
h = 0.01
n = 1400

y0 = np.array([-0.97000436, 0.24308753])
v0 = np.array([-0.46620368, -0.43236573])
y1 = np.array([0.97000436, -0.24308753])
v1 = np.array([-0.46620368, -0.43236573])
y2 = np.array([0., 0.])
v2 = np.array([0.93240737, 0.86473146])


mass = np.array([M, M, M])

liste = (y0, v0, y1, v1, y2, v2)

yall = np.zeros(12)

for idx, i in enumerate(liste):
    for kdx, k in enumerate(i):
        yall[idx*len(i) + kdx] = k

input_vec = (yall, x0)

#endInit

a, b, c = setup(h, n, input_vec, mass)
d, e, f = setup(h*0.1, n, input_vec, mass)

#x (here the time) is not needed for the orbits only for the length of the array

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Overlayed")
plt.plot(a[0], a[1])
plt.plot(b[0], b[1])
plt.plot(c[0], c[1])
plt.subplot(122)
plt.plot(d[0], d[1])
plt.plot(e[0], e[1])
plt.plot(f[0], f[1])

plt.savefig("fig2a.pdf")

#b)######################################################################################################################

#Vars
h = 0.001
n = 2200

#Init
y0 = np.array([3., -1.])
v0 = np.array([-0., 0.])
y1 = np.array([-1., 2.])
v1 = np.array([0., 0.])
y2 = np.array([-1., -1.])
v2 = np.array([0., 0.])

mass = np.array([3*M, 4*M, 5*M])

liste = (y0, v0, y1, v1, y2, v2)

for idx, i in enumerate(liste):
    for kdx, k in enumerate(i):
        yall[idx*len(i) + kdx] = k

input_vec = (yall, x0)

#endInit

a, b, c = setup(h, n, input_vec, mass)

plt.figure(figsize=(8, 8))
plt.subplot(111)
plt.title("Overlayed")
plt.plot(a[0], a[1])
plt.plot(b[0], b[1])
plt.plot(c[0], c[1])

plt.savefig("fig2b.pdf")

#c)######################################################################################################################

#Vars
h = 0.001
n = 2850

#Init
y0 = np.array([3., -1.])
v0 = np.array([-0.8, 0.])
y1 = np.array([-1., 2.])
v1 = np.array([0., 0.])
y2 = np.array([-1., -1.])
v2 = np.array([0., 0.])

mass = np.array([3*M, 4*M, 5*M])

liste = (y0, v0, y1, v1, y2, v2)

for idx, i in enumerate(liste):
    for kdx, k in enumerate(i):
        yall[idx*len(i) + kdx] = k

input_vec = (yall, x0)

#endInit

a, b, c = setup(h, n, input_vec, mass)
d, e, f = setup(h*10, n, input_vec, mass)

plt.figure(figsize=(8, 8))
plt.subplot(111)
plt.title("Overlayed")
plt.plot(a[0], a[1])
plt.plot(b[0], b[1])
plt.plot(c[0], c[1])

plt.savefig("fig2c.pdf")
