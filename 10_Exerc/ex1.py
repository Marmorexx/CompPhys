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

    # For integrals larger than 1, create more than 1 distribution
    needed_Iter = int(function(1)-1e-3)+1   # substract a small value, so
    integral_steps = ([])                   # that functions like 'x' finish
    r_i = ([])                              # after just 1 iteration
    s_i = ([])

    if needed_Iter > 1:
        print('Integral larger than 1... '\
            +str(needed_Iter)+' iterations needed in total')

    # Start integration procedure
    for c in range(needed_Iter):

        print('Creating Distribution '+str(c+1)+' for f(x) = '+function.__name__)
        accepted = 0

        # Create a progress bar
        bar = progressbar.ProgressBar(maxval=steps, \
                widgets=[progressbar.Bar('=', '[', ']'), ' ',
                progressbar.Percentage()])
        bar.start()
        
        # Generate random numbers
        for i in range(steps):
            bar.update(i+1)
            tmpr = generate_random(a,m,c)/(m-1) # r_i
            tmps = generate_random(a,m,c)/(m-1)+c# s_i
            if (tmps < function(tmpr)): 
                r_i.append(tmpr)
                s_i.append(tmps)
                accepted+=1
        bar.finish()
        
        # Append the calculated Integral
        print('Integral = ', accepted/steps)
        integral_steps.append(accepted/steps)

    # Calculate the sum of all Integrals
    integral = sum(integral_steps)
    return r_i, s_i, integral

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
a = 2**16+3 #1060
m = 2**31 #96911
c = 12399 #12835
initVal = .5
steps = int(1e6)
###############################################################

#### Start Program

r1, s1, integral1 = rejectionmethod(a,m,c,initVal,steps, square)
r2, s2, integral2 = rejectionmethod(a,m,c,initVal,steps, cubic)
r3, s3, integral3 = rejectionmethod(a,m,c,initVal,steps, sine)
r4, s4, integral4 = rejectionmethod(a,m,c,initVal,steps, exponential)

# {{{ Plot Data

print('Plotting...')

xaxis = np.linspace(0,1,100)

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.35)
plt.subplot(221)
plt.scatter(r1[::100],s1[::100],s=5,
        label=r'$\int_0^1\ f(x) = $'+str(round(integral1,5)))
plt.plot(xaxis, square(xaxis), c='r', alpha=0.6, label=r'$f(x)$')
plt.legend()
plt.title(r'$f(x) = \int \ x^2 \ dx$', pad=20)
plt.subplot(222)
plt.scatter(r2[::100],s2[::100],s=5,
        label=r'$\int_0^1\ f(x) = $'+str(round(integral2,5)))
plt.plot(xaxis, cubic(xaxis), c='r', alpha=0.6, label=r'$f(x)$')
plt.legend()
plt.title(r'$f(x) = \int \ x^3 \ dx$', pad=20)
plt.subplot(223)
plt.scatter(r3[::100],s3[::100],s=5,
        label=r'$\int_0^1\ f(x) = $'+str(round(integral3,5)))
plt.plot(xaxis, sine(xaxis), c='r', alpha=0.6, label=r'$f(x)$')
plt.legend()
plt.title(r'$f(x) = \int \ \sin(x) \ dx$', pad=20)
plt.subplot(224)
plt.scatter(r4[::100],s4[::100],s=3,
        label=r'$\int_0^1\ f(x) = $'+str(round(integral4,5)))
plt.plot(xaxis, exponential(xaxis), c='r', alpha=0.6, label=r'$f(x)$')
plt.legend()
plt.title(r'$f(x) = \int \ e^x \ dx$', pad=20)
plt.show()

# }}}
