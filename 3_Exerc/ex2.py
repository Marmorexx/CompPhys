# In collaboration with Group of Jan Hubrich
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
def numerov(it_count, it_step, x_0, y_01, func, epsilon):
    x_n = np.zeros(it_count)
    y_n = np.zeros(it_count)
    # init x_n
    for i in range(it_count):
        x_n[i] = x_0 + i * it_step
    # first 2 elements of y_n
    y_n[0] = y_01[0]
    y_n[1] = y_01[1]
    # loop for the iterations
    for i in range(2, it_count):
        y_n[i] = 2 * (1- 5/12 * it_step**2 * func(x_n[i-1],epsilon)) * y_n[i-1]
        y_n[i] -= (1 + 1/12 * it_step**2 * func(x_n[i-2],epsilon)) * y_n[i-2]
        y_n[i] /= (1 + 1/12 * it_step**2 * func(x_n[i],epsilon))
    return x_n, y_n

def k(x,epsilon):
    return epsilon-x

# Set Starting values
stepsize = 0.001
steps = int(10/stepsize)
x0 = 0
a = 0.001
y_init=np.zeros(3)
y_init[0] = 0
y_init[1] = a

print("integrating from 0 to " + str(steps*stepsize) + " in steps of " + str(stepsize))

testx = int(steps-1) # This is where we test our function (choose high x)

# Choose the Interval, in which you assume epsilon, and the amount of attempts
lambda_min = 2.27
lambda_max = 2.36
attempts = 40

# Check for the sign at lambda_min
x_vector, y_vector = numerov(steps,stepsize,x0,y_init,k,lambda_min)
sign_0 = int(np.sign(y_vector[testx]))

#start the Bisection Procedure
for n in range(attempts):
    epsilon = (lambda_min+lambda_max)/2 # set epsilon to lambda_half

    x_vector, y_vector = numerov(steps,stepsize,x0,y_init,k,epsilon)

    sign = int(np.sign(y_vector[testx])) # check for the sign of our divergence

    print('Testing...')
    print('epsilon = ', epsilon)
    print('lambda_min = ', lambda_min)
    print('lambda_max = ', lambda_max)
    print('y = ',y_vector[testx]) 
    
    # choose, in which way to correct the interval
    if sign == sign_0: 
        print('changing lambda_min')
        lambda_min = epsilon
    if sign != sign_0:
        print('changing lambda_max')
        lambda_max = epsilon
    
print(r"Approximation for $\varepsilon$: ", epsilon)

plt.plot(x_vector,y_vector,label=r"$\varepsilon$ = "+str(epsilon))
plt.xlabel('x')
plt.ylabel('E')
plt.yscale('symlog')
plt.legend()
plt.show()
