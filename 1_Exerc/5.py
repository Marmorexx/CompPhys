import matplotlib.pyplot as plt
import numpy as np
import math as m

# Set Constants
G = 1
Ma = 1
Mb = 1

# Set up starting conditions

xa = 0
ya = 0
za = 0
vec_a0 = np.array([xa, ya, za])

vxa = 0.5
vya = -0.5
vza = 0
va0 = np.sqrt(vxa**2+vya**2+vza**2)

xb = 1
yb = 0
zb = 0
vec_b0 = np.array([xb,yb,zb])

vxb = -0.5
vyb = 0.5
vzb = 0
vb0 = np.sqrt(vxb**2+vyb**2+vzb**2)

t = 0
dt = 10e-2

# Iniciate Vectors
xalist = []
yalist = []

xblist = []
yblist = []

dE = [] # for difference of energy per step calculation

while t < 10:
    # Compute Force, aka F = GMaMb/|r|**3*vecr
    rx = xb-xa
    ry = yb-ya
    rz = zb-za

    absr3 = ( rx**2+ry**2+rz**2 )**1.5

    fx = -G*Ma*Mb/absr3*rx
    fy = -G*Ma*Mb/absr3*ry
    fz = -G*Ma*Mb/absr3*rz

    # Update the quantities of Mb

    vxb += fx*dt/Mb
    vyb += fy*dt/Mb
    vzb += fz*dt/Mb

    xb += vxb*dt
    yb += vyb*dt
    zb += vzb*dt

    # Update the quantities of Ma

    vxa += -fx*dt/Ma
    vya += -fy*dt/Ma
    vza += -fz*dt/Ma

    xa += vxa*dt
    ya += vya*dt
    za += vza*dt

    t += dt

    # create norm function
    norm = lambda vec: sum([i**2 for i in vec])**.5

    # calculate Runge-Lenz
    mu = 1 / (1 / Ma + 1/Mb)
    M = Ma
    r = np.array([xa, ya, za])
    v = np.array([vxa, vya, vza])
    p = M*v
    L = np.cross(r,p)
    j = L / mu
    phi = np.arctan(r[1] / r[0])
    e = np.cross(v,j) / (G*M) - r / norm(r)

    # calculate energies at 0th step and n-th step
    vec_a = np.array([xa,ya,za])
    vec_b = np.array([xb,yb,zb])
    va = np.array([vxa,vya,vza])
    vb = np.array([vxb,vyb,vzb])

    E_0 = 2*M/2*va0**2-2*G*M/norm(vec_a0-vec_b0)
    E_n = M/2*(norm(va)**2+norm(vb)**2)-2*G*M/norm(vec_a-vec_b)

    # Difference in Energies

    dE.append(E_n)

    # Save Information in lists

    xalist.append(xa)
    yalist.append(ya)
    xblist.append(xb)
    yblist.append(yb)

print(norm(e))


plt.plot(np.arange(0,t,dt),dE)
plt.axis("equal")
plt.xlabel('$t$')
plt.ylabel('$\Delta E$')
plt.savefig('1e.pdf')
plt.show()
