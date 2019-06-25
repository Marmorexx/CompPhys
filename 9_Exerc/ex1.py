import numpy as np
import sys
import matplotlib.pyplot as plt

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

    print('\n'.join(map(str, randomarray)))

    #Then the normalized one
    print("\nNormalizing...")
    nrandomarray = np.zeros(steps)
    for i in range(steps):
        nrandomarray[i] = randomarray[i]/(m-1)

    print('\n'.join(map(str, nrandomarray)))

    return randomarray, nrandomarray


###### Initial Values 1: #######
# a = 106                      # 
# m = 6075                     # 
# c = 1283                     # 
# x0 = 1                       # 
# steps = 10                   # 
# createarrays(a,m,c,x0,steps) # 
################################

r1, rnorm1 = createarrays(1060,60751,12835,1,100000)
r2, rnorm2 = createarrays(1060,60751,12835,2,100000) #just change initial value
print("Mean Value of sequence 1: mu = {}".format(np.round(np.mean(rnorm1),3)))
print("Mean Value of sequence 2: mu = {}".format(np.round(np.mean(rnorm2),3)))

#Drawing the random Number plane
plt.figure()
plt.scatter(rnorm1, rnorm2, s=5)
plt.savefig('F_Random_number_plane.pdf')

#Drawing the deterministic character
plt.figure()
plt.scatter(rnorm1[:-1], rnorm1[1:], s=10)
plt.savefig('F_Deterministic_character.pdf')
