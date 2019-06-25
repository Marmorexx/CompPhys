import numpy as np
import matplotlib.mlab as mlab
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('_classic_test')

def init(initVal):
    global rand
    rand = initVal

def generate_random(a,m,c):
    global rand
    rand = (a*rand+c)%m
    return rand

def createarrays(a,m,c,initVal,steps):
    init(initVal)

    #First, we create the normal Random number array
    print("Random Number sequence: ")
    randomarray = np.zeros(steps, dtype=int)
    for i in range(steps):
        randomarray[i] = generate_random(a,m,c)

    #Then the normalized one
    print("\nNormalizing...")
    nrandomarray = np.zeros(steps)
    for i in range(steps):
        nrandomarray[i] = randomarray[i]/((m-1)/6)

    return randomarray, nrandomarray


###### Initial Values 1: #######
# a = 106                      # 
# m = 6075                     # 
# c = 1283                     # 
# x0 = 1                       # 
# steps = 10                   # 
# createarrays(a,m,c,x0,steps) # 
################################

r1, rnorm1 = createarrays(1060,60751,12835,2,int(1e5))

# Add up entries of random numbers to packets of 10 each
print("r1 has {} entries".format(len(rnorm1)))
n = int(len(rnorm1)/10)
dices = ([])                        # -Create a list where we store the infor-
for i in range(n):                  # mation of our dice rolls
    dsum = 10                       # -So we actually hit entries between 10 and
    for j in range(10):             # 60, and not 0 and 50
        dsum += int(rnorm1[10*i+j])
    dices.append(dsum)

# Fit a gaussian
(mu, sigma) = norm.fit(dices)

# Draw Dice throws in a plane
xaxis = np.linspace(0,len(dices)-1,len(dices))
plt.subplot(211)
plt.title('Sum of dice rolls for each experiment')
plt.scatter(xaxis,dices,s=1)
plt.ylim(10,60)
plt.xlim(0,len(dices))

for i in range(10,61): # Read out histogram entries
    print(dices.count(i), " occurences of the number {}".format(i))
print("Mean value: {}".format(np.mean(dices)))

# Draw Histogram and gaussian
plt.subplot(212)
nhist, bins, patches = plt.hist(dices, bins=range(60),normed=1, alpha=0.75)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r', linewidth=2)
plt.xlim(10,61)
plt.show()
