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

def rejectionmethod(a,m,c,initVal,steps,function):
    # initialize with any starting value
    init(initVal)

    # Create Random number distribution
    print("Creating Distribution...")
    r_i = ([])
    s_i = ([])
    accepted = 0
    # Create a progress bar
    bar = progressbar.ProgressBar(maxval=steps, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()])
    bar.start()
    for i in range(steps):
        bar.update(i+1)
        tmpr = generate_random(a,m,c)/(m-1) # r_i
        tmps = generate_random(a,m,c)/(m-1) # s_i
        if (tmps < function(tmpr)): 
            r_i.append(tmpr)
            s_i.append(tmps)
            accepted+=1
    bar.finish()
    return r_i, s_i, accepted

# Add functions to integrate

def square(x):
    return x**2

def cubic(x):
    return x**3

def sine(x):
    return np.sin(x)

def exponential(x):
    return np.exp(x)

###################### Input Parameters #######################
a = 1060
m = 96911
c = 12835
initVal = .5
steps = int(1e6)
###############################################################

# Start Program

r1, s1, acc1 = rejectionmethod(a,m,c,initVal,steps, square)
r2, s2, acc2 = rejectionmethod(a,m,c,initVal,steps, cubic)
r3, s3, acc3 = rejectionmethod(a,m,c,initVal,steps, sine)
r4, s4, acc4 = rejectionmethod(a,m,c,initVal,steps, exponential)

# Plot Data
print('Plotting...')

plt.figure(figsize=(16,16))
plt.subplot(221)
plt.scatter(r1[::100],s1[::100],s=5,
        label=r'$\int f(x) = $'+str(round(acc1/steps,2)))
plt.legend()
plt.title(r'$\int \ x^2 \ dx$')
plt.subplot(222)
plt.scatter(r2[::100],s2[::100],s=5,
        label=r'$\int f(x) = $'+str(round(acc2/steps,2)))
plt.legend()
plt.title(r'$\int \ x^3 \ dx$')
plt.subplot(223)
plt.scatter(r3[::100],s3[::100],s=5,
        label=r'$\int f(x) = $'+str(round(acc3/steps,2)))
plt.legend()
plt.title(r'$\int \ sin(x) \ dx$')
plt.subplot(224)
plt.scatter(r4[::100],s4[::100],s=5,
        label=r'$\int f(x) = $'+str(round(acc4/steps,2)))
plt.legend()
plt.title(r'$\int \ e^x \ dx$')
plt.show()
