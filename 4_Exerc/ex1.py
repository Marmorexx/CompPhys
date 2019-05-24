import numpy as np
import matplotlib.pyplot as plt

# globals

eps=1e-6

# funcs

def swap(vec1, vec2):
    tmp = np.copy(vec2)
    vec2 = np.copy(vec1)
    vec1 = np.copy(tmp)
    return vec1, vec2

def create_mat(N, a=0., b=1., c=0.):
    mat = np.zeros((N, N))
    for idx in range(N):
        mat[idx][idx] = b
        if (1 <= idx):
            mat[idx][idx-1] = a
            mat[idx-1][idx] = c
    return mat

def create_vec(N, a=0.):
    vec = np.zeros(N)
    for idx in range(N):
        vec[idx] = a
    return vec

def GJ(mat_i, vec_i, pr=False): # (2.1)
    #mat_i: input matrix in the shape (N, N)
    #vec_i: input vector in the shape (N, )
    #pr   : input boolean for printing the array of the computational steps
    #vec  : output vector in the shape (N, )


    N = mat_i.shape[0]
    # init N x (N+1) array
    mat = np.zeros((mat_i.shape[0], mat_i.shape[1] + 1)) 
    for i in range(N):
        for j in range(N):
            mat[i][j] = mat_i[i][j]
        mat[i][N] = vec_i[i]

    if (pr): 
        print("1st")
        print(np.round(mat,2))
        print(" ")
    
    # start elementary matrix computations
    for j in range(N):
        max_val = 0
        max_idx = j
        for idx, i in enumerate(mat):
            if idx >= j:
                if (i[j] > max_val):
                    max_val = i[j]
                    max_idx = idx
        # swap row max_idx to the j-th row
        if (max_idx != j):
            mat[j], mat[max_idx] = swap(mat[j], mat[idx]) # perform on mat

            if(pr):
                print("Swap Rows")
                print(np.round(mat,2))
                print(" ")

        # subtract top row from others (
        for kdx in range(N):
            if (kdx != j):
                fac = np.copy(mat[kdx][j])  #Need to create a copy so fac doesnt
                for l in range(N+1):        #get changed
                    mat[kdx][l] = np.copy(mat[kdx][l] - fac * mat[j][l]/mat[j][j])
                    
        if (pr):
            print("Steps")
            print(np.round(mat,2))
            print(" ")

    # norm to 1
    for j in range(N):
        mat[j] = mat[j] / mat[j][j]
       
    if (pr):
        print("Normalize")
        print(np.round(mat,2))
        #print(" ")

    # put last column in vector -> Solution
    vec = np.zeros(vec_i.shape)
    for i in range(N):
        vec[i] = mat[i][N]

    return vec

################################################################################
#                           INPUT OF MATRIX AND VECTOR                         #
################################################################################

# Solve any System of Linear equations (thats solvable):
# A = np.array([[-1, 1, 1], [1, -3, -2], [5, 1, 4]])
# b = np.array([0, 5, 3])

# Exercise 2.4
A = create_mat(10, -1, 2, -1)
b = create_vec(10, 0.1)

################################################################################
#                               END OF INPUT SECTION                           #
################################################################################

# Choose if you want to have each of the steps printed or not (Console input)
errormsg = True
while errormsg is True:
    print("Print each step? [y/n]")
    printinput = input()
    if printinput == "y":
        x = GJ(A, b, True) # True shows the steps of Computation in the matrix
        errormsg = False
    elif printinput == "n":
        x = GJ(A, b)
        errormsg = False
    else:
        print("Error, please only type 'y' or 'n' \n")

# prints the calculations of the system (2.3)
def printt():
    print("Ax =\n", np.round(A.dot(x),2),"\n")
    print("A =\n", np.round(A,2), "\n")
    print("x =\n", np.round(x,2), "\n")
    print("b =\n", np.round(b,2))

printt()

# check solution (2.2)
flag = 0
for k in range(b.shape[0]):
    if ((A.dot(x))[k] != b[k]):
        flag = 1

# output solution with respect to the check above and calculate deviation (2.5)
if (flag == 0):
    print("\nThe answer is:\n", x)
else:
    print("\nSomething went wrong!\n")
    print("The answer may be:\n")
    print(x, "\n")
    print("A rounding error may have occured!\n")
    print("b differs from Ax by:\n",A.dot(x)[:] - b[:])
