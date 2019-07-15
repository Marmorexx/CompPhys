import numpy as np
import matplotlib.pyplot as plt
import progressbar
import scipy.integrate as integrate

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

    print('Computing Integral of '+function.__name__+' with '+\
str(steps)+' iterations (out of '+str(int(10**N_max))+')...')
    
    # Generate random numbers
    r = np.ndarray((steps))
    for i in range(steps):
        r[i] = generate_random(a,m,c)/(m-1) # r_i

    integral = 1/steps*np.sum(function(r))
    return r, integral

### Add functions to integrate

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
initVal = .5 # this is our x_0
x_min = 0
x_max = 2
N_min = 2 # Minimum order of Magnitude
N_max = 6 # Maximum order of Magnitude
density = 2 # Relative plot density (2 = 10^2 calculations)
###############################################################

#### Start Program

Range = np.logspace(N_min,N_max,int(10**(density)), dtype=int)
acc1 = np.ndarray((len(Range)))
acc2 = np.ndarray((len(Range)))
acc3 = np.ndarray((len(Range)))
acc4 = np.ndarray((len(Range)))

# Create a progress bar
for N in enumerate(Range):
    r1, integral1 = rejectionmethod(a,m,c,initVal,int(N[1]), square)
    acc1[N[0]] = abs(integral1-integrate.quad(square,0,1)[0])
for N in enumerate(Range):
    r2, integral2 = rejectionmethod(a,m,c,initVal,N[1], cubic)
    acc2[N[0]] = abs(integral2-integrate.quad(cubic,0,1)[0])
for N in enumerate(Range):
    r3, integral3 = rejectionmethod(a,m,c,initVal,N[1], sine)
    acc3[N[0]] = abs(integral3-integrate.quad(sine,0,1)[0])
for N in enumerate(Range):
    r4, integral4 = rejectionmethod(a,m,c,initVal,N[1], exponential)
    acc4[N[0]] = abs(integral4-integrate.quad(exponential,0,1)[0])

print('Plotting...')
plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.35)
thiccness = 1

plt.subplot(221)
plt.plot(Range,acc1, linewidth=thiccness)
plt.xlabel('# of Iterations')
plt.ylabel('Error between MC and Analytical')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\int_0^1 \ x^2 \ dx$',pad=20)

plt.subplot(222)
plt.plot(Range,acc2, linewidth=thiccness)
plt.xlabel('# of Iterations')
plt.ylabel('Error between MC and Analytical')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\int_0^1 \ x^3 \ dx$',pad=20)

plt.subplot(223)
plt.plot(Range,acc3, linewidth=thiccness)
plt.xlabel('# of Iterations')
plt.ylabel('Error between MC and Analytical')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\int_0^1 \ \sin (x) \ dx$',pad=20)

plt.subplot(224)
plt.plot(Range,acc4, linewidth=thiccness)
plt.xlabel('# of Iterations')
plt.ylabel('Error between MC and Analytical')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$\int_0^1 \ e^x \ dx$',pad=20)

plt.show()
