import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

# Our cubic function
def cubic(x,c):
    return -2/15*x**3+x**2-(2/15+1/c)*x+1

# Count roots
def countzero(axis, arr, c):
    counter = 0
    sign = np.sign(arr[0])
    for i in range(len(arr)):
        if(np.sign(arr[i]) != sign):
            if (sign == 1): #Check if stable
                print('point at x = ', round(axis[i],2), 'for c = ', c, ' is stable')
            else:
                print('point at x = ', round(axis[i],2), 'for c = ', c, ' is unstable')
            sign = sign*(-1)
            counter += 1
    return counter


    
# Define x axis and a range of Ac that will be tested
xaxis = np.linspace(0,7,1000)
crange = np.round(np.linspace(0.01,1,100),5)

# Store z here for further analysis
c1 = ([])
c3 = ([])

for c in crange:
    ns = countzero(xaxis, cubic(xaxis,c), c)
    if ns == 3:
        plt.subplot(211)
        plt.plot(xaxis,cubic(xaxis, c),color='green',linewidth=1)
        c3.append(c)
        plt.title('Three Stationary Points')
    if ns == 1:
        plt.subplot(212)
        plt.plot(xaxis,cubic(xaxis, c),color='black',linewidth=1)
        plt.title('One Stationary Point')
        c1.append(c)

print('For ', min(c1), ' <= z < ', min(c3),
        ' and ', max(c3), ' < z <= ' , max(c1), ' there is one stationary point')
print('For ', min(c3), ' <= z <= ', max(c3), ' there are three stationary points')

plt.subplot(211)
plt.ylim(-4,4)
plt.xlim(0,7)
plt.subplot(212)
plt.ylim(-4,4)
plt.xlim(0,7)
plt.show()
