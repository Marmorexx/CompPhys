import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

#function to build integral of
def f(x):
	return np.exp(-x**2)

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
	r_i = np.array([])
	for i in range(steps):
		tmpr = generate_random(a,m,c)/(m-1) # random array
		x_i = xa + (xb-xa)*tmpr
		r_i = np.append(x_i,r_i)
		
	return r_i

###################### Input Parameters #######################
a = 2e16+3
m = 2e31
c = 0
initVal = 1
steps1 = int(1e4) 	#Iterations
xa, xb = -5,5		#Integrationboundaries
###############################################################
r1 = rejectionmethod(a, m, c, initVal, steps1)

I = np.sum(f(r1))/len(r1)*(xb-xa)

plt.figure(figsize=(11,5))
plt.tight_layout()

plt.subplot(111)
plt.title('n = {}'.format(len(r1)))
plt.scatter(r1,f(r1),label=I)
plt.legend()
plt.show()