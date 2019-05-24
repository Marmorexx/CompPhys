import numpy as np
import matplotlib.pyplot as plt

# {{{ Runge-Kutta

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
    mindist12 = 5 # initial distance
    mindist23 = 3 # initial distance
    mindist31 = 4 # initial distance
    mint12 = 0
    mint23 = 0
    mint31 = 0
    
    for n in np.arange(1,n+1,1):
        yn[n,:], xn[n] = rk4_step(y0 = yn[n-1,:], x0 = xn[n-1], f = f, h = h, f_args = f_args)
        # Calculate distances of masses:
        r1 = np.array([yn[n,0],yn[n,1]])
        r2 = np.array([yn[n,4],yn[n,5]])
        r3 = np.array([yn[n,8],yn[n,9]])
        
        dist12 = dist(r1,r2)
        dist23 = dist(r2,r3)
        dist31 = dist(r3,r1)

        # Yes, this can be done with np.min, though we realized too late that we
        # were going to save all of those distances in arrays anyways.
        if (dist12 <= mindist12): # if minimum distance smaller than before, save it
            mindist12 = dist12
            mint12 = n 
        if (dist23 <= mindist23):
            mindist23 = dist23
            mint23 = n 
        if (dist31 <= mindist31):
            mindist31 = dist31
            mint31 = n 

    print('Minimum distances for h = ', round(h,10), 'and n =', n)

    print('Minimum Distance of Masses 1 and 2: ',
            round(mindist12,2), 'at t =', mint12)
    print('Minimum Distance of Masses 2 and 3: ',
            round(mindist23,2), 'at t =', mint23)
    print('Minimum Distance of Masses 3 and 1: ',
            round(mindist31,2), 'at t =', mint31, '\n')

    return(yn, xn)

# {{{ Extra function that we only want to use for saving the arrays to save computation time
def rk4_withplots(y0, x0, f, h, n, f_args = {}): #same function, saves additional arrays
    yn = np.zeros((n+1, y0.shape[0]))
    xn = np.zeros(n+1)
    yn[0,:] = y0
    xn[0] = x0
    mindist12 = 5 # initial distance
    mindist23 = 3 # initial distance
    mindist31 = 4 # initial distance
    mint12 = 0
    mint23 = 0
    mint31 = 0
    distances12 = np.zeros(n+1) # we store this for later
    distances23 = np.zeros(n+1)
    distances31 = np.zeros(n+1)
    timeaxis = np.zeros(n+1)
    E = np.zeros(n+1)
    
    for n in np.arange(1,n+1,1):
        yn[n,:], xn[n] = rk4_step(y0 = yn[n-1,:], x0 = xn[n-1], f = f, h = h, f_args = f_args)

        # Calculate distances of masses:
        r1 = np.array([yn[n,0],yn[n,1]])
        r2 = np.array([yn[n,4],yn[n,5]])
        r3 = np.array([yn[n,8],yn[n,9]])
        
        dist12 = dist(r1,r2)
        dist23 = dist(r2,r3)
        dist31 = dist(r3,r1)

        distances12[n] = dist12
        distances23[n] = dist23
        distances31[n] = dist31
        timeaxis[n] = n

        T1 = T((yn[n,2]**2+yn[n,3]**2)**0.5,m1)
        T2 = T((yn[n,6]**2+yn[n,7]**2)**0.5,m2)
        T3 = T((yn[n,10]**2+yn[n,11]**2)**0.5,m3)
        V12 = V(dist12,m1+m2)
        V23 = V(dist23,m2+m3)
        V31 = V(dist31,m3+m1)
        E[n] = T1+T2+T3+V12+V23+V31

        if (dist12 <= mindist12): # if minimum distance smaller than before, save
            mindist12 = dist12
            mint12 = n 
        if (dist23 <= mindist23):
            mindist23 = dist23
            mint23 = n 
        if (dist31 <= mindist31):
            mindist31 = dist31
            mint31 = n 

    print('Minimum distances for h = ', round(h,10), 'and n =', n)

    print('Minimum Distance of Masses 1 and 2: ',
            round(mindist12,2), 'at t =', mint12)
    print('Minimum Distance of Masses 2 and 3: ',
            round(mindist23,2), 'at t =', mint23)
    print('Minimum Distance of Masses 3 and 1: ',
            round(mindist31,2), 'at t =', mint31, '\n')

    return(yn, xn, distances12, distances23, distances31, timeaxis, E)
# }}}

# }}}

# {{{ define functions

def function(y0,x0,m): # This calculates ALL derivatives
    out = np.zeros(y0.shape) # y.shape = (12,)
    for index in np.arange(0,len(y0),4): # iterate over the 3 Masses
            out[index] = y0[index+2] # f(x) = vx
            out[index+1] = y0[index+3] # f(y) = vy
            for indexx in np.arange(0,len(y0),4):
                if (index != indexx): # sum over non-equal indizes
                    out[index+2] += G*m[(int)(indexx/4)]*(y0[indexx]-y0[index]
                    )/((y0[indexx]-y0[index])**2+(y0[indexx+1]-y0[index+1]
                    )**2)**1.5 # f(vx) = vx+G*m*(r/|r|*3)

                    out[index+3] += G*m[(int)(indexx/4)]*(y0[indexx+1]-y0[index+1]
                    )/((y0[indexx]-y0[index])**2+(y0[indexx+1]-y0[index+1]
                    )**2)**1.5 # f(vy) = vy+G*m*(r/|r|*3)
    return out

def final(h,n,init_vec,m):
    y0,x0 = init_vec
    y0,x0 = rk4(y0,x0,function,h,n,{"m" : m})

    out_x0 = np.zeros(x0.shape)
    out_y0 = np.zeros(x0.shape)
    out_x1 = np.zeros(x0.shape)
    out_y1 = np.zeros(x0.shape)
    out_x2 = np.zeros(x0.shape)
    out_y2 = np.zeros(x0.shape)

    for index, i in enumerate(y0):
        out_x0[index] = i[0]
        out_y0[index] = i[1]
        out_x1[index] = i[4]
        out_y1[index] = i[5]
        out_x2[index] = i[8]
        out_y2[index] = i[9]

    return ((out_x0, out_y0), (out_x1, out_y1), (out_x2, out_y2))

def final_withplots(h,n,init_vec,m): # Takes into account extra output for 2b
    y0,x0 = init_vec
    y0,x0, dist12, dist23, dist31, taxis, E = rk4_withplots(y0,x0,function,h,n,{"m" : m})

    out_x0 = np.zeros(x0.shape)
    out_y0 = np.zeros(x0.shape)
    out_x1 = np.zeros(x0.shape)
    out_y1 = np.zeros(x0.shape)
    out_x2 = np.zeros(x0.shape)
    out_y2 = np.zeros(x0.shape)

    for index, i in enumerate(y0):
        out_x0[index] = i[0]
        out_y0[index] = i[1]
        out_x1[index] = i[4]
        out_y1[index] = i[5]
        out_x2[index] = i[8]
        out_y2[index] = i[9]

    return ((out_x0, out_y0), (out_x1, out_y1), (out_x2, out_y2), dist12, dist23, dist31, taxis, E)

# For b calculation
def dist(r1,r2):
   return ((r2[0]-r1[0])**2+(r2[1]-r1[1])**2)**0.5

def T(v,m):
    return 1/2*m*v**2

def V(r,m):
    return -G*m/r

# }}}

# {{{ a) set starting values and plot
G = 1
m1 = 1
m2 = 1
m3 = 1
masses = np.array([m1,m2,m3])
h = 1e-1 # Stepsize
n = 200 # Number of Steps
x0 =np.array([0]) # is time, begins at 0

y0 = np.array([-0.97000436, 0.24308753])
v0 = np.array([-0.46620368, -0.43236573])
y1 = np.array([0.97000436, -0.24308753])
v1 = np.array([-0.46620368, -0.43236573])
y2 = np.array([0., 0.])
v2 = np.array([0.93240737, 0.86473146])

values = (y0, v0, y1, v1, y2, v2) #this is still a multidimensional array
y_ges = np.zeros(12) #want to write all values in a 1-Dimensional array

# This fills in y_ges in a dynamic way
for index, i in enumerate(values):
    for kindex, k in enumerate(i):
        y_ges[index*len(i) + kindex] = k

input_vector = (y_ges, x0)

# {{{ Calculte and Plot for two different Stepsizes h

a,b,c = final(h,n,input_vector,masses)
d,e,f = final(h*0.1,n,input_vector,masses)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title("Shape")
plt.plot(a[0],a[1])
plt.plot(b[0],b[1])
plt.plot(c[0],c[1])
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(122)
plt.title('Trajectories')
plt.plot(d[0],d[1], label='m1')
plt.plot(e[0],e[1], label='m2')
plt.plot(f[0],f[1], label='m3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('fig2a.pdf')

# }}}

# }}}

# {{{ b) set starting values and plot

m1 = 3
m2 = 4
m3 = 5
masses = np.array([m1,m2,m3])
h = 1e-3 # Stepsize
n = 5000 # Number of Steps
x0 =np.array([0]) # is time, begins at 0

y0 = np.array([3,-1])
v0 = np.array([0,0])
y1 = np.array([-1,2])
v1 = np.array([0,0])
y2 = np.array([-1,-1])
v2 = np.array([0,0])

values = (y0, v0, y1, v1, y2, v2) #this is still a multidimensional array
y_ges = np.zeros(12) #want to write all values in a 1-Dimensional array

# This fills in y_ges in a dynamic way
for index, i in enumerate(values):
    for kindex, k in enumerate(i):
        y_ges[index*len(i) + kindex] = k

input_vector = (y_ges, x0)

# {{{ Calculte and Plot trajectories for two different stepsizes h

a,b,c = final(h,n,input_vector,masses)
d,e,f = final(h,n,input_vector,masses)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title("h = 0.1, n = 50")
plt.plot(a[0],a[1], label='m1')
plt.plot(b[0],b[1], label='m2')
plt.plot(c[0],c[1], label='m3')
plt.plot(y0[0],y0[1],'*',color='black', label='x0')
plt.plot(y1[0],y1[1],'*',color='black')
plt.plot(y2[0],y2[1],'*',color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.subplot(122)
plt.title('h = 0.01, n = 500')
plt.plot(d[0],d[1], label='m1')
plt.plot(e[0],e[1], label='m2')
plt.plot(f[0],f[1], label='m3')
plt.plot(y0[0],y0[1],'*',color='black', label='x0')
plt.plot(y1[0],y1[1],'*',color='black')
plt.plot(y2[0],y2[1],'*',color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('fig2b.pdf')

# }}}

# {{{ Calculate Distances and Energies for one stepsize

a,b,c,dist12,dist23,dist31,taxis, E = final_withplots(h,n,input_vector,masses)

plt.figure(figsize=(15,4))
plt.subplot(131)
plt.title("Trajectories")
plt.plot(a[0],a[1], label='m1')
plt.plot(b[0],b[1], label='m2')
plt.plot(c[0],c[1], label='m3')
plt.plot(y0[0],y0[1],'*',color='black', label='x0')
plt.plot(y1[0],y1[1],'*',color='black')
plt.plot(y2[0],y2[1],'*',color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.subplot(132)
plt.title('Distances')
plt.yscale('log')
plt.plot(taxis[1:], dist12[1:], label='$1 \leftrightarrow 2$')
plt.plot(taxis[1:], dist23[1:], label='$2 \leftrightarrow 3$')
plt.plot(taxis[1:], dist31[1:], label='$3 \leftrightarrow 1$')
plt.xlabel('t')
plt.ylabel('Distance')
plt.legend()
plt.subplot(133)
plt.title('Energy Error')
plt.plot(taxis[1:], E[1:])
plt.xlabel('t')
plt.ylabel('Energy Error')
plt.yscale('log')
plt.savefig('test.pdf')
plt.show()


# }}}

# }}}
