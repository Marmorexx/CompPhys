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
    pi_estimations = np.ndarray((len(steps)))
    for setsize in enumerate(steps):
        # Create a progress bar for cool progress visualization
        bar = progressbar.ProgressBar(maxval=setsize[1], \
                widgets=[progressbar.Bar('=', '[', ']'), ' ',
                progressbar.Percentage()])
        bar.start()
        acc = 0
        print(r'Approximating $\pi$ with ', setsize[1], ' random numbers...')
        for i in range(setsize[1]):
            bar.update(i+1)
            tmpr = generate_random(a,m,c)/(m-1) # r_i
            tmps = generate_random(a,m,c)/(m-1) # s_i
            f_x = np.sqrt(1-tmpr**2)     # f(x_i) = b*x_i
            if (tmps < f_x): 
                acc+=1
        pi_estimations[setsize[0]] = (acc/setsize[1]*4)
        bar.finish()
        print("pi = {}".format(acc/setsize[1]*4))
    return pi_estimations

####################### Choose plot parameters ##########################

oom = 5         # orders of magnitude to be calculated                  
density = 100     # Relative Data density (integer > 1)                   
#NOTE Don't choose o.o.m larger than 6 to avoid long computation times    
#########################################################################

# Making it user friendly
if oom <= 2:
    print("Error: Please choose a higher order of magnitude ( > 2 )")
if (oom >= 7) or (1/10**(6-oom)*density > 10):
    if oom >= 7:
        print("Warning: Choosing more than 6 orders of Magnitude might result\
 in long computation times.\nContinue anyways? [y/n]")
    else:
        print("You chose a high density, which might result in long computation\
 times.\nContinue anyways? [y/n]")
    con = False
    while con == False:
        answer = str(input())
        if answer == 'y':
            con = True
            print("Calculating...")
        elif answer == 'n':
            print('Exiting...')
            exit()
        else: print("please only press y or n, followed by the return-key") 

# Create an x-axis
n = np.logspace(2,oom,density*(oom-1), dtype=int)

# Create the array containing our pi estimates
pi_array_1 = rejectionmethod(1060, 96911, 12835, 1, n)

# Plot Accuracy as function of Setsize
plt.title('Accuracy of $\pi$ for different setsizes')
plt.xlabel('Setsize')
plt.ylabel('Accuracy')
plt.plot(n, 1-abs(pi_array_1-np.pi),label='generator')
plt.yscale('log')
plt.xscale('log')

plt.show()
