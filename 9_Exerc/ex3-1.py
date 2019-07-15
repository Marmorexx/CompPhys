import numpy as np
import matplotlib.pyplot as plt
import progressbar

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
    # Create a progress bar
    bar = progressbar.ProgressBar(maxval=steps, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()])
    bar.start()
    for i in range(steps):
        bar.update(i+1)
        tmpr = generate_random(a,m,c)/(m-1) # r_i
        tmps = generate_random(a,m,c)/(m-1) # s_i
        f_x = np.sqrt(1-tmpr**2)     # f(x_i) = b*x_i
        if (tmps < f_x): 
            r_i.append(tmpr)
            s_i.append(tmps)
            acc+=1
    bar.finish()
    return r_i, s_i, acc

###################### Input Parameters #######################
a = 1060
m = 96911
c = 12835
initVal = 1
steps1, steps2 = int(1e3), int(1e8) # Iterations
###############################################################

r1, s1, acc1 = rejectionmethod(a, m, c, initVal, steps1)
r2, s2, acc2 = rejectionmethod(a, m, c, initVal, steps2)

pi1 = r'$\pi \approx$ '+str(acc1/steps1*4)
pi2 = r'$\pi \approx$ '+str(acc2/steps2*4)

plt.figure(figsize=(11,5))
plt.tight_layout()

plt.subplot(121)
plt.title('n = {}'.format(steps1))
plt.scatter(r1,s1,s=5, label=pi1)
plt.legend()

plt.subplot(122)
plt.title('n = {}'.format(steps2))
plt.scatter(r2[::1000],s2[::1000],s=1, label=pi2) # just plot every 100th point
plt.legend()                                    # to avoid big filesizes

plt.show()
