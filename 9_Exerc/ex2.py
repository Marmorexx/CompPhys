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

def rejectionmethod(a,m,c,A,initVal,steps):
    # initialize with any starting value
    init(initVal)

    # Create Random number distribution
    print("Creating Distribution...")
    r_i = ([])
    s_i = ([])
    # Create a progress bar
    bar = progressbar.ProgressBar(maxval=steps, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()])
    bar.start()
    for i in range(steps):
        bar.update(i+1)
        tmpr = generate_random(a,m,c)/(m-1) # r_i
        tmps = generate_random(a,m,c)/(m-1) # s_i
        x = A * tmpr    # x_i = a*r_i
        f_x = x/A     # f(x_i) = b*x_i
        if (tmps < f_x): 
            r_i.append(x)
            s_i.append(tmps)
    bar.finish()
    return r_i, s_i

###################### Input Parameters #######################
a = 1060
m = 96911
c = 12835
initVal = .5
steps1, steps2 = int(1e3), int(1e5) # Iterations
A = .5
###############################################################

# Start Program
r1, s1 = rejectionmethod(a,m,c,A,initVal,steps1)
r2, s2 = rejectionmethod(a,m,c,A,initVal,steps2)
x = np.linspace(0,A,100)

# Plot Data
print('Plotting data...')
plt.figure(figsize=(8,4))
plt.subplot(221)
plt.title('n = {}'.format(steps1))
plt.scatter(r1,s1,s=5)
plt.subplot(223)
plt.hist(r1, normed=1)      #TODO Doesnt Norm to 1... Scaling is wrong
plt.plot(x, 2*x/A**2)
plt.subplot(222)
plt.title('n = {}'.format(steps2))
plt.scatter(r2,s2,s=1)
plt.subplot(224)
plt.hist(r2,50,normed=1)    #TODO Doesnt Norm to 1.. Scaling is wrong
plt.plot(x, 2*x/A**2)
plt.show()
