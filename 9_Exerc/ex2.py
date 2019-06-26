import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

def init(initVal):
    global rand
    rand = initVal

def generate_random(a,m,c):
    global rand
    rand = (a*rand+c)%m
    return rand

def rejectionmethod(a,m,c,A,initVal,steps):
    # initialize with any starting value
    init(initVal)

    # Create Random number distribution
    print("Creating Distribution...")
    r_i = ([])
    s_i = ([])
    b = 1/A
    for i in range(steps):
        tmpr = generate_random(a,m,c)/(m-1) # r_i
        tmps = generate_random(a,m,c)/(m-1) # s_i
        print(tmpr,tmps)
        x = A * tmpr    # x_i = a*r_i
        f_x = b * x     # f(x_i) = b*x_i
        if (tmps < f_x): 
            r_i.append(x)
            s_i.append(tmps)
            print("value {} accepted".format(round(tmps,2)))
        else: 
            print("value {} rejected".format(round(tmps,2)))
    return r_i, s_i

#####################################################
# Syntax: rejectionmethod(a,m,c,A,initVal,steps)    #
#####################################################

A = .5
r1, s1 = rejectionmethod(1060,60751,12835,A,.5,1000)
r2, s2 = rejectionmethod(1060,96911,12835,A,.5,100000)
x = np.linspace(0,A,100)

plt.figure(figsize=(8,4))
plt.subplot(221)
plt.title('n = 1000')
plt.scatter(r1,s1,s=5)
plt.subplot(223)
plt.hist(r1, density=True)      #Doesnt Norm to 1... Scaling is wrong
plt.plot(x, 2*x)
plt.subplot(222)
plt.title('n = 100000')
plt.scatter(r2,s2,s=1)
plt.subplot(224)
plt.hist(r2,50,density=True)    #Doesnt Norm to 1.. Scaling is wrong
plt.plot(x, 1/A*x)
plt.show()
