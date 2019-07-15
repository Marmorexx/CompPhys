import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

#function to build integral of
def f(x):
	return np.exp(-x**2)
	
#weightfunction for x_i, should be near f(x)
def g(x):
	return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

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
	s_i = np.array([])
	acc = 0
	for i in range(steps):
		if acc < 10000:
			tmpr = generate_random(a,m,c)/(m-1) # random array
			tmps = generate_random(a,m,c)/(m-1) # second random array, to distribute
											# x_i in an additional dimension 
			x_i = xa + (xb-xa)*tmpr			#Adapt random array to bounds of integration
			y_i = ya + (yb-ya)*tmps			#Adapt 2. random array, has to go higher than function
		
			if (y_i < g(x_i)):	# function how x_i should be distributet
				r_i = np.append(x_i,r_i)
				s_i = np.append(y_i,s_i)
				acc+=1					# number of x_i (=N)

	return r_i,s_i,acc

###################### Input Parameters #######################
a = 2e16+3
m = 2e31
c = 0
initVal = 1
steps1 = int(2e7) # Iterations

xa, xb = -5,5		#Integrationboundaries
ya, yb = 0, 10
###############################################################

r1,s1,N = rejectionmethod(a, m, c, initVal, steps1)

I = np.sum(f(r1)/g(r1))/N

plt.figure(figsize=(11,5))
plt.tight_layout()

plt.subplot(111)
plt.title('n = {}'.format(N))
plt.scatter(r1,s1,s=1, label='Integral ='+str(I))
plt.plot(r1,f(r1),linewidth=0, color='red',marker='x')
plt.legend()
plt.show()