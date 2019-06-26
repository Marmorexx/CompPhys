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

def rejectionmethod(a,m,c,initVal,steps):
    # initialize with any starting value
    init(initVal)

    # Create Random number distribution
    print("Creating Distribution...")
    r_i = ([])
    s_i = ([])
    rej = 0
    acc = 0
    for i in range(steps):
        tmpr = generate_random(a,m,c)/(m-1) # r_i
        tmps = generate_random(a,m,c)/(m-1) # s_i
        f_x = np.sqrt(1-tmpr**2)     # f(x_i) = b*x_i
        if (tmps < f_x): 
            r_i.append(tmpr)
            s_i.append(tmps)
            #print("value {} accepted".format(round(tmps,2)))
            acc+=1
        else: 
            #print("value {} rejected".format(round(tmps,2)))
            rej+=1
    return r_i, s_i, acc, rej

steps1, steps2 = int(1e3), int(1e6)
r1, s1, acc1, rej1 = rejectionmethod(1060, 96911, 12835, 1, steps1)
r2, s2, acc2, rej2 = rejectionmethod(1060, 96911, 12835, 1, steps2)

pi1 = r'$\pi \approx$ '+str(acc1/(acc1+rej1)*4)
pi2 = r'$\pi \approx$ '+str(acc2/(acc2+rej2)*4)

plt.figure(figsize=(11,5))
plt.tight_layout()

plt.subplot(121)
plt.title('n = {}'.format(steps1))
plt.scatter(r1,s1,s=5, label=pi1)
plt.legend()

plt.subplot(122)
plt.title('n = {}'.format(steps2))
plt.scatter(r2[::100],s2[::100],s=1, label=pi2)
plt.legend()

plt.show()
