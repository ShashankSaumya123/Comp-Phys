#importing libraries
import math
import copy
import numpy as np

#Function to read the file and getting numbers in a Matrix
def file_opener(file_name):
    Matrix = []
    with open(file_name) as file:
        M_string = file.readlines()
    for line in M_string:
        Matrix.append(list(map(lambda i: float(i), line.split(" "))))
    return Matrix


#Partial Pivoting function (finding the maximum value number to put in the row to get more computational stability)
def partialPivot(A,B,sizeA,sizeB):
    for i in range(sizeA):
        if(A[i][i]==0):
            i_max = A[i][i]
            counter = i
            for j in range(i+1,sizeA):
                if(A[j][i]>i_max):
                    i_max = A[j][i]
                    counter = j
            swapRow(A,B,i,j,sizeA,sizeB)



#Row swapping Function
def swapRow(mat1, mat2, i, j, sizeA, sizeB):
    for k in range(sizeA):
        temp1 = mat1[i][k]
        mat1[i][k] = mat1[j][k]
        mat1[j][k] = temp1
    
    for l in range(sizeB):
        temp2 = mat2[i][l]
        mat2[i][l] = mat2[j][l]
        mat2[j][l] = temp2


#Gauss-Jordan executing function
def gaussJordan(A, B):
    sizeA = len(A[0])
    sizeB = len(B[0])
    for k in range(sizeA):
        #Dividing by pivot
        pivot = A[k][k]
        for j in range(k,sizeA):
            A[k][j] /= pivot
        
        for l in range(sizeB):
            B[k][l] /= pivot
        
        #Making the rest of the terms in kth column zero
        for i in range(sizeA):
            if i==k or A[i][k] == 0:
                continue
            factor = A[i][k]
            for j in range(k,sizeA):
                A[i][j] -= (factor*A[k][j])
                
            for l in range(sizeB):
                B[i][l] -= (factor * B[k][l])
    

#Matrix Multiplication function
def matMultiply(M, N):
    size = len(M)
    M_cross_N = [[], [], []]
    for i in range(size):
        for j in range(size):
         M_cross_N[i].append(sum(map(lambda k: M[i][k] * N[k][j], range(size))))
    return M_cross_N


#Function to create Identity Matrix
def identity(n):
    I = [[0 for y in range(n)] for x in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I













#Function for LU decomposition
#Both L and U are directly kept together in A
#Note:- Diagonal elements of L should be 1 but are not stored in A
def luDecompose(A):
    """
    B[i][j] = ith row and jth column
    """
    n = len(A[0])
    for i in range(n):
        # Upper Triangle Matrix (i is row index, j is column index, k is summation index)
        for j in range(i,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(i==k):
                    sum += A[k][j]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[i][k]*A[k][j]
        
            A[i][j] = A[i][j] - sum
        
        # Lower Triangle Matrix (j is row index, i is column index, k is summation index)
        for j in range(i+1,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(j==k):
                    sum += A[k][i]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[j][k]*A[k][i]
            A[j][i] = (A[j][i] - sum)/A[i][i]


#Cholesky Decomposition
def choleskyDecompose(A,B = None):
    """
    B[i][j] = ith row and jth column
    """
    m = len(A[0])
    
    L = np.zeros((m,m))
    #Create L row by row
    for i in range(m):
        for j in range(i+1):
            sum = 0
            for k in range(j):
                sum += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = np.sqrt(A[i][i] - sum)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum))
    
    #Forward Backward substitution for Cholesky
    if(B!=None):
        n = len(B[0])
        #Forward Substitution
        for j in range(n):
            for i in range(m):
                sum = 0
                for k in range(i):
                    sum += L[i][k]*B[k][j]
                B[i][j] = (B[i][j] -  sum)/L[i][i]

        #Backward Substitution
        for j in range(n):
            for i in range(m-1,-1,-1):
                sum = 0
                for k in range(i+1,m):
                    sum += L[k][i]*B[k][j]
                B[i][j] = (B[i][j] - sum)/L[i][i]
            
        return L,B
    else: return L

#Function for Forward-Backward Substitution
#Works in complimentary to LU Decomposition function
def forwardBackwardSubstitution(A,B):
    m = len(A)
    n = len(B[0])
    #Forward Substitution
    for j in range(n):
        for i in range(m):
            sum = 0
            for k in range(i):
                sum += A[i][k]*B[k][j]
            B[i][j] = B[i][j] -  sum

    #Backward Substitution
    for j in range(n):
        for i in range(m-1,-1,-1):
            sum = 0
            for k in range(i+1,m):
                sum += A[i][k]*B[k][j]
            B[i][j] = (B[i][j] - sum)/A[i][i]


#Jacobi Method to find inverse
def JacobiInv(A,b,x=None, tol = 10**(-6), residue_list = False):
    """Solves the equation Ax=b via the Jacobi iterative method.
       A is the matrix
       b is the solution
       Note:- Also changes the original matrices. Store beforehand if needed.
    """

    # Flattening the solution array
    b = np.array(b)
    b = np.ndarray.flatten(b)

    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(A))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diag(A)
    LU = np.array(A) - np.diagflat(D)
    
    xn = (b - np.dot(LU,x)) / D

    # Iterate till you reach the tolerance
    errl = []
    err = np.linalg.norm(xn-x)
    errl.append(err)
    
    while err>tol:
        # Store previous x
        x = xn
        xn = (b - np.dot(LU,x)) / D
        err = np.linalg.norm(xn-x)
        errl.append(err)

    if(residue_list == True): return errl, xn
    return xn

# Gauss-Seidel function for solving x
def GaussSeidel(A, b, x=None, tol = 1e-6, residue_list = False):
    n = len(A)
    if x is None: x = np.zeros(n)
    err = np.inf
    errl = []

    while err>tol:
        sum = 0
        for i in range(n):
            # temp variable d to store b[j]
            d = b[i]
            
            # to calculate x
            for j in range(n):
                if(i != j):
                    d-=A[i][j] * x[j]
            # Storing previous and updating the value of our solution
            temp = x[i]
            x[i] = d / A[i][i]
            sum += (x[i]-temp)**2
        
        #Error update
        err = sum
        errl.append(err)
    if(residue_list == True): return errl, x
    return x

# Conjugate Gradient function to find solution x
def ConjGrad(A,b,x = None, tol = 1e-4, max_iter = 100, residue_list = False):
    n = len(A)
    if x is None: x = np.ones(n)
    r = b - np.dot(A,x)
    d = copy.deepcopy(r)
    rl = []
    count = 0
    while (np.dot(r,r)>tol and count<max_iter):

        rn = np.dot(r,r)
        rl.append(rn)       # Appending the residue value

        a = (rn)/(np.dot(d,np.dot(A,d)))
        x += a*d
        r -= a*np.dot(A,d)

        b = np.dot(r,r)/rn
        d = r + b*d
        count += 1
    if(residue_list == True): return rl, x
    return x


# Function to find the inverse of  matrix using solver algos
def Inverse(matrix, xsolvername = "JacobiInv", tols = 1e-4, residue_lists = False):
    # Choosing the solver function
    if(xsolvername == "JacobiInv"): xsolver = JacobiInv
    if(xsolvername == "Gauss-Seidel"): xsolver = GaussSeidel
    if(xsolvername == "ConjugateGrad"): xsolver = ConjGrad

    I = np.identity(len(matrix))
    Inv = np.zeros((len(matrix),len(matrix)))
    if(residue_lists == False):
        for i in range(len(matrix)):
            Inv[:,i] = xsolver(matrix, I[i],tol = tols, residue_list = residue_lists)
        return Inv 
    else:
        resl = []
        for i in range(len(matrix)):
            res, inv = xsolver(matrix, I[i], tol = tols, residue_list = residue_lists)
            Inv[:,i] = inv
            if(i == 0): resl = res
        return resl ,Inv


# Eigenvalues Calculators

# Function which calculates the largest eigenvalue using power method
def PowerMethodCalc(A, x, tol = 1e-4):
    oldEVal = 0 # Dummy initial instance
    eVal = 2

    while abs(oldEVal-eVal)>tol:
        x = np.dot(A,x)
        eVal = max(abs(x))
        x = x/eVal

        oldEVal=eVal

    return eVal,x

# Wrapper function which allows us to get multiple eigenvalues
def EigPowerMethod(A, x=None, n=1, tol = 1e-4):
    if x is None: x = np.ones(len(A))
    eig = []
    Vl = []
    E,V = PowerMethodCalc(A,x,tol)
    eig.append(E)
    Vl.append(V)
    if n>1:
        iter = n-1
        while iter > 0:
            V = V/np.linalg.norm(V)
            V = np.array([V])
            A = A - E*np.outer(V,V)
            E,V = PowerMethodCalc(A,x,tol)
            eig.append(E)
            Vl.append(V)
            iter -= 1
    return eig,Vl

#Jacobi Method for eigenvalues (Given's Rotation)
def JacobiEig(A):
    n = len(A)
    # Find maximum off diagonal value in upper triangle
    def maxfind(A):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if (abs(A[i][j]) >= Amax):
                    Amax = abs(A[i][j])
                    k = i
                    l = j
        return Amax,k,l

    def GivensRotate(A, tol = 1e-4, max_iter = 5000):
        max = 4
        iter = 1
        while (abs(max) >= tol and iter < max_iter):
            max,i,j = maxfind(A)
            if A[i][i] - A[j][j] == 0:
                theta = math.pi / 4
            else:
                theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2

            Q = np.eye(n)
            Q[i][i] = Q[j][j] = math.cos(theta)
            Q[i][j] = -1*math.sin(theta)
            Q[j][i] = math.sin(theta) 
            AQ = matMultiply(A,Q)

            # Q inv = Q transpose
            Q = np.array(Q)
            QT = Q.T.tolist()

            A = matMultiply(QT,AQ)
            iter += 1
        return A
    sol = GivensRotate(A)
    return np.diagonal(sol)

# Fitting

# Polynomial Chi Square fitting
def PolynomialChiSqFit(x, y, u = None, n=4, solver = GaussSeidel):
    """
    x & y are dataset
    u: Uncertainty, 1 if not given
    n: number of parameters. (order+1). 2 for linear
    """

    if u == None: u = np.ones(len(x))

    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sum =0 
            for k in range(len(x)):
                sum  += x[k]**(i+j)/u[k]**2
            A[i][j] = sum
    for i in range(n):
        sum = 0
        for j in range(len(x)):
            sum += x[j]**(i)*y[j]/u[j]**2
        B[i] = sum
    C = solver(A,B)
    fit = []
    for i in range(len(x)):
        fit.append(C[0]+C[1]*x[i]+C[2]*(x[i]**2)+C[3]*(x[i]**3)) # Edit according the order of polynomial needed
    
    cond = np.linalg.cond(A)
    Cov = Inverse(A)
    #print(Cov)

    """
    #chi_2 = sum(  ( (Obs-Expect)^2 )/Expect   )

    chi2 = 0
    for i in range(len(x)):
        chi2 += ((y[i]-fit[i])**2)/fit[i]
    
    dof = len(x)-len(C)
    """

    return C,Cov,cond

# Chebyschev polynomial Chi Square fitting
def ChebyPolynomialChiSqFit(x, y, u = None, n=4, solver = GaussSeidel):
    """
    x & y are dataset
    u: Uncertainty, 1 if not given
    n: number of parameters. (order+1). 2 for linear
    """

    def cheby(i,X):
        if(i == 0): return 1
        if(i == 1): return X
        if(i == 2): return 0.5*((3*X*X)-1)
        if(i == 3): return 0.5*((5*X*X*X)-(3*X))
        if(i == 4): return (1/8)*((35*X*X*X*X)-(30*X*X)+3)
        if(i == 5): return (1/8)*((63*X*X*X*X*X)-(70*X*X*X)+(15*X))
        if(i == 6): return (1/16)*((231*X*X*X*X*X*X)-(315*X*X*X*X)+(105*X*X)-5)
    

    if u == None: u = np.ones(len(x))

    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sum =0 
            for k in range(len(x)):
                sum  += (cheby(i,x[k])*cheby(j,x[k]))/u[k]**2
            A[i][j] = sum
    for i in range(n):
        sum = 0
        for j in range(len(x)):
            sum += cheby(i,x[j])*y[j]/u[j]**2
        B[i] = sum
    C = solver(A,B)
    # fit = []
    # for i in range(len(x)):
    #     fit.append(C[0]+C[1]*x[i]+C[2]*(x[i]**2)+C[3]*(x[i]**3)) # Edit according the order of polynomial needed
    
    cond = np.linalg.cond(A)
    Cov = Inverse(A)
    #print(Cov)

    return C,Cov,cond



# Function to create jacknife resampling
# Outputs a matrix with each row as a subsample
def JackKnife(Pop):
    n = len(Pop)
    # Resampling
    sub = []
    for i in range(n):
        s = copy.deepcopy(Pop)
        del s[i]
        sub.append(s)
    
    # Mean of subsamples
    m = []
    for i in range(n):
        m.append(sum(sub[i])/(n-1))
    mu = sum(m[i] for i in range(n))/n

    # Std Deviation
    std = sum((m[i] - mu)**2 for i in range(n))/n

    # Standard Error
    stderr = (n-1)*std
    return sub

#DFT
def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X

# Pseudo RNG (Produce numbers from 0 to 1)
def MLCG_RNG(m = 420, a = 11,seed = 69, n = 10):
    x = [seed]
    for i in range(n):
        xn = (a*x[-1])%m
        x.append(xn)
    x = [xs/m for xs in x]
    return x

def RandomWalks(N):
    """
    Random walker moves in 1D.
    """
    x = [0] #Array to keep track of pos (starts from centre)
    rando = MLCG_RNG(m = 1021,a = 65,seed = 69, n = N)  #Array of random numbers
    for i in range(N):
        if(rando[i]>=0.5): x.append(x[-1]-1) #Go left
        else: x.append(x[-1]+1)
    return x


#Function to create derivative functions
def derivative(f,h = 10**-6):
    def df_dx(x):
        return (f(x+h)-f(x-h))/(2*h)
    return df_dx


#Function to create polynomials
def polynomialCreater(l):
    n = len(l)-1
    def func(x):
        f = 0
        for i in range(n+1):
            f += l[i]*(x**(n-i))
        return f
    return func


#Function to bracket the root
def bracketChecker(f,a,b,shift = 0.2):
    i = 0
    while (i<20):
        if (f(a)*f(b) > 0):
            if (abs(f(a))<abs(f(b))):
                a -= shift
                i += 1
        else: return (a,b)
    print("The root is not Bracketed for these inputs. Stopping the function...")
    return (None,None)


#Function to solve by bisection method
def bisection(f,a,b,fileName,tol = 10**-6):
    #making sure a is left of b
    if(a>b):
        temp = b
        b = a
        a = temp
    
    #Bracketing the roots
    a,b = bracketChecker(f,a,b)
    if((a,b) == False): return None

    #Checking if either of a or b is a root
    if(f(a)*f(b) == 0):
        if(f(a)==0):
            return a
        else: return b

    else:
        i = 0
        c = 0
        while(abs(a-b) > tol and i<200):
            c_old = c
            c = (a+b)/2.0
            abserr = c - c_old
            if (abs(f(c)) < tol):
                return c
            if (f(a)*f(c)<0):
                b = c
            else: a = c
            
            #Appending the data to separate file
            with open(fileName, 'a') as dat_file:
                print(i,",",abserr, file=dat_file)
            i += 1
    return (a+b)/2.0


#Function to solve by False Position method
def falsePosition(f,a,b,fileName,tol = 10**-6):
    #making sure a is left of b
    if(a>b):
        temp = b
        b = a
        a = temp
    
    #Bracketing the roots
    a,b = bracketChecker(f,a,b)
    if((a,b) == False): return None

    #Checking if either of a or b is a root
    if(f(a)*f(b) == 0):
        if(f(a)==0):
            return a
        else: return b

    else:
        i = 0
        c = 0
        while(abs(a-b) > tol and i<200):
            c_old = c
            c = b - (((b-a)*f(b))/(f(b)-f(a)))
            abserr = c - c_old
            if (abs(f(c)) < tol):
                return c
            if (f(a)*f(c)<0):
                b = c
            else: a = c
            
            #Appending the data to separate file
            with open(fileName, 'a') as dat_file:
                print(i,",",abserr, file=dat_file)
            i += 1
    return (a+b)/2.0


#Solving by Newton-Rhapson Method
def newtonRhapson(f,a,fileName,tol = 10**-6):
    #Checking if a is already a root
    if(abs(f(a))<tol):
        return a

    i = 0
    b = 1   #just an initial value for b
    while(abs(b)>tol and i<200):
        df_dx = derivative(f)
        b = f(a)/df_dx(a)
        a -= b
        
        #Appending the data to separate file
        with open(fileName, 'a') as dat_file:
                print(i,",",b, file=dat_file)
        i += 1
    return a


#Function to solve the root of entire polynomial by Laguerre's method
def laguerreSolver(f,l,a,tol = 10**-6):
    sol = [] # The array to put roots into
    n = len(l) - 1 #order of polynomial
    
    # Parent loop
    while(n>0):
        #Creating derivatives of f
        df_dx = derivative(f)
        d2f_dx2 = derivative(df_dx)
        
        #Checking if provided a is the root itself
        if (f(a) == 0):
            sol.append(a)
            l = deflate(l,a,n)
            f = polynomialCreater(l)
            n -= 1


        else:
            # assigning random values to G,H and b for now.
            G = 0
            H = 0
            b = 1
            
            while(b>tol):
                G = df_dx(a)/f(a)
                H = (G*G) - (d2f_dx2(a)/f(a))
                if (G<0):
                    k = -1
                else:
                    k = 1
                b = n/(G + k*(math.sqrt((n-1)*((n*H)-(G*G)))))
                a -= b
            
            sol.append(a)
            l = deflate(l,a,n)
            f = polynomialCreater(l)
            n -= 1
    return sol


#The function that does synthetic division
def deflate(l,a,n):
    x = []
    x.append(l[0])
    for i in range(1,n):
        b = l[i] + a*x[-1]
        x.append(b)
    return x


# ASSIGNMENT 6 PART STARTS HERE
# Function that integrates by midpoint method
def midpointIntegral(f, lowerBound, upperBound, N):
    h = (upperBound - lowerBound)/N
    integral = 0
    for i in range(N):
        integral += f(lowerBound + (((2*i)+1)*(h/2)))
    integral *= h
    return integral


# Function that integrates by Trapezoidal method
def trapezoidalIntegral(f, lowerBound, upperBound, N):
    h = (upperBound - lowerBound)/N
    integral = f(lowerBound) + f(upperBound)
    for i in range(1,N):
        integral += 2*f(lowerBound + (i*h))
    integral *= h/2
    return integral


# Function that integrates by Simpsons method
def simpsonsIntegral(f, lowerBound, upperBound, N):
    h = (upperBound - lowerBound)/N
    integral = f(lowerBound) + f(upperBound)
    for i in range(1, N):
        
        if(i%2 != 0): weight = 4
        else: weight = 2
        
        integral += weight*f(lowerBound + (i*h))
    integral *= h/3
    return integral


from random import random       # The random number generator


# Function that integrates by Monte-Carlo method
def monteCarloIntegral(f, lowerBound, upperBound, N, fileName = None, rando = "rando", m = 420, a = 11, seed = 69):
    """
    f: function to be integrated
    lowerbound and upperbound: self explanatory
    N: number of points to take
    filename: If we wish to append the integral value to a separate file
    rando: decides which RNG to use
    m,a,seed: used for MLCG Pseudo RNG
    """
    
    integral = 0
    width = upperBound - lowerBound
    h = width/N
    rand = MLCG_RNG(m,a,seed,N)
    

    for i in range(N):
        if(rando == "MLCG"): 
            integral += f(lowerBound + (width*rand[i]))
        else:
            integral += f(lowerBound + (width*random()))
    integral *= h
    #Appending the data to separate file
    if(fileName != None):
        with open(fileName, 'a') as dat_file:
            print(N,",",integral, file=dat_file)
    return integral

def monteCarloIntegralImportanceSampling(f, p, y, lowerBound, upperBound, N, fileName = None, rando = "rando", m = 420, a = 11, seed = 69):
    """
    f: function to be integrated
    p: Importance sampling function, set it to 1 when not needed
    y: Function corresponding to importance sampling function, set to x when not needed
    lowerbound and upperbound: self explanatory
    N: number of points to take
    filename: If we wish to append the integral value to a separate file
    rando: decides which RNG to use
    m,a,seed: used for MLCG Pseudo RNG
    """
    
    integral = 0
    width = upperBound - lowerBound
    h = width/N
    rand = MLCG_RNG(m,a,seed,N)
    

    for i in range(N):
        if(rando == "MLCG"): 
            integral += (f(lowerBound + (width*y(rand[i]))))/(p(y(rand[i])))
            #integral += f(lowerBound + (width*rand[i]))
        else:
            x = random()
            integral += (f(lowerBound + (width*y(x))))/(p(y(x)))
            #integral += f(lowerBound + (width*random()))
    integral *= h
    #Appending the data to separate file
    if(fileName != None):
        with open(fileName, 'a') as dat_file:
            print(N,",",integral, file=dat_file)
    return integral

# ASSIGNMENT 7 PART STARTS HERE
# The forward Euler Function
def forwardEuler(dy_dx, xRange, iniCond = (0, 0), dx = 0.01):
    """
    parameters:
    - dy_dx: function
    - xRange: pair - (xMin, xMax) the range of over which the value of the function is required
    - iniCond = (0, 0): pair - values at some point
    - dx = 0.01: float - dx
    
    returns:
    - xl: list of integers of values for x
    - yl: list of integers of values for corresponding y
    """
    
    xl = [iniCond[0]]
    yl = [iniCond[1]]
    xMin, xMax = xRange

    # Towards +ve x direction
    x, y = iniCond
    while x < xMax:
        xPrev = x
        yPrev = y
        x = xPrev + dx
        y = yPrev + dx * dy_dx(xPrev, yPrev)
        xl.append(x)
        yl.append(y)

    # Towards -ve x direction
    x, y = iniCond
    while x > xMin:
        xPrev = x
        yPrev = y
        x = xPrev - dx
        y = yPrev - dx * dy_dx(xPrev, yPrev)
        xl.insert(0, x)
        yl.insert(0, y)

    # Removing x values which are not in xRange
    if xMin > iniCond[0]:
        for i in range(len(xl)):
            if xl[i] < xMin:
                continue
            else:
                xl = xl[i:]
                break
    if xMax < iniCond[0]:
        for i in range(len(xl)):
            if xl[-1] > xMax:
                continue
            else:
                xl = xl[:i]
                break

    return xl, yl


# Runge-Kutta Method
def RK4(dy_dx, xRange, iniCond = (0, 0), dx = 0.01):
    """
    parameters:
    - dy_dx: function
    - xRange: pair - (xMin, xMax) the range of over which the value of the function is required
    - iniCond = (0, 0): pair - values at some point
    - dx = 0.01: float - dx
    
    returns:
    - xl: list of integers of values for x
    - yl: list of integers of values for corresponding y
    """
    
    xl = [iniCond[0]]
    yl = [iniCond[1]]
    xMin, xMax = xRange
    
    # Towards +ve x direction
    while xl[-1] < xMax:
        x = xl[-1]
        y = yl[-1]
        k1 = dx * dy_dx(x, y)
        k2 = dx * dy_dx(x + (dx/2), y + (k1/2))
        k3 = dx * dy_dx(x + (dx/2), y + (k2/2))
        k4 = dx * dy_dx(x + dx, y + k3)
        xNew = x + dx
        yNew = y + (k1 + 2*k2 + 2*k3 + k4)/6
        xl.append(xNew)
        yl.append(yNew)

    # Towards -ve x direction
    while xl[0] > xMin:
        x = xl[0]
        y = yl[0]
        k1 = dx * dy_dx(x, y)
        k2 = dx * dy_dx(x - (dx/2), y - (k1/2))
        k3 = dx * dy_dx(x - (dx/2), y - (k2/2))
        k4 = dx * dy_dx(x - dx, y - k3)
        xNew = x - dx
        yNew = y - (k1 + 2*k2 + 2*k3 + k4)/6
        xl.insert(0, xNew)
        yl.insert(0, yNew)

    # Removing x values which are not in xRange
    if xMin > iniCond[0]:
        for i in range(len(xl)):
            if xl[i] < xMin:
                continue
            else:
                xl = xl[i:]
                break
    if xMax < iniCond[0]:
        for i in range(len(xl)):
            if xl[-1] > xMax:
                continue
            else:
                xl = xl[:i]
                break

    return xl, yl


def shootingMethod(dz_dx, root1, root2, odeSolver = "RK4", iterationLimit = 10, dx = 0.01):
    """
    parameters:
    - dz_dx: function, where z = dy/dx
    - root1: (x, y), one boundary
    - root2: (x, y), second boundary
    - odeSolver="rk4": other option is forwardEuler
    - iterationLimit = 10: integer
    - dx = 0.01: float
    
    returns:
    - xl: list of integers of values for x
    - yl: list of integers of values for corresponding y
    """
    
    if odeSolver == "RK4":
        odeSolve = RK4
    elif odeSolver == "forwardEuler":
        odeSolve = forwardEuler
    iterationCount = 0
    guess1 = [root1[0], 1]
    guess2 = [root1[0], -1]
    if root2[1] != 0:
        yl = [0]
    else:
        yl = [1]
    xl = [0]
    while(abs(yl[-1] - root2[1]) >= 10 ** -13 and iterationCount < iterationLimit):
        if iterationCount == 0:
            guess = guess1.copy()
        elif iterationCount == 1:
            guess1.append(yl[-1])
            guess = guess2.copy()
        else:
            if iterationCount == 2:
                guess2.append(yl[-1])
            else:
                guess1[2] = yl[-1]

            # Generating new guess
            guess = guess1[1] + (guess2[1] - guess1[1]) * (root2[1] - guess1[2])/(guess2[2] - guess1[2])
            guess1[1] = guess
            guess = guess1
            
        xl, zs = odeSolve(dz_dx, (guess[0] - dx, root2[0] + dx), (guess[0], guess[1]), dx = dx/2)

        xl = list(map(lambda x: round(x, 6), xl))

        def dy_dx(x, y):
            return zs[xl.index(round(x, 6))]

        xl, yl = odeSolve(dy_dx, (guess[0], root2[0]), root1, dx = dx)
        iterationCount += 1
    return xl, yl

def coupled_RK4(dz_dx, dy_dx, xRange, iniCond = (0, 0, 0), dx = 0.01):
    """
    parameters:
    - dy_dx: derivative function
    - dz_dx: derivative function
    - xRange: pair - (xMin, xMax) the range of over which the value of the function is required
    - iniCond1 = (0, 0, 0): pair - values (x,y,z)
    - dx = 0.01: float - dx
    
    returns:
    - xl: list of integers of values for x
    - yl: list of integers of values for corresponding y
    - yl: list of integers of values for corresponding z
    """
    
    xl = [iniCond[0]]
    yl = [iniCond[1]]
    zl = [iniCond[2]]
    xMin, xMax = xRange
    
    # Towards +ve x direction
    while xl[-1] < xMax:
        x = xl[-1]
        y = yl[-1]
        z = zl[-1]
        k1 = dx * dy_dx(x, y, z)
        l1 = dx * dz_dx(x, y, z)

        k2 = dx * dy_dx(x + (dx/2), y + (k1/2), z + (l1/2))
        l2 = dx * dz_dx(x + (dx/2), y + (k1/2), z + (l1/2))

        k3 = dx * dy_dx(x + (dx/2), y + (k2/2), z + (l2/2))
        l3 = dx * dz_dx(x + (dx/2), y + (k2/2), z + (l2/2))

        k4 = dx * dy_dx(x + dx, y + k3, z + l3)
        l4 = dx * dz_dx(x + dx, y + k3, z + l3)

        xNew = x + dx
        yNew = y + (k1 + 2*k2 + 2*k3 + k4)/6
        zNew = z + (l1 + 2*l2 + 2*l3 + l4)/6
        xl.append(xNew)
        yl.append(yNew)
        zl.append(zNew)

    # Towards -ve x direction
    while xl[0] > xMin:
        x = xl[0]
        y = yl[0]
        z = zl[0]

        k1 = dx * dy_dx(x, y, z)
        l1 = dx * dz_dx(x, y, z)
        
        k2 = dx * dy_dx(x - (dx/2), y - (k1/2), z - (l1/2))
        l2 = dx * dz_dx(x - (dx/2), y - (k1/2), z - (l1/2))

        k3 = dx * dy_dx(x - (dx/2), y - (k2/2), z - (l2/2))
        l3 = dx * dz_dx(x - (dx/2), y - (k2/2), z - (l2/2))

        k4 = dx * dy_dx(x - dx, y - k3, z - l3)
        l4 = dx * dz_dx(x - dx, y - k3, z - l3)

        xNew = x - dx
        yNew = y - (k1 + 2*k2 + 2*k3 + k4)/6
        yNew = z - (l1 + 2*l2 + 2*l3 + l4)/6

        xl.insert(0, xNew)
        yl.insert(0, yNew)
        zl.insert(0, zNew)

    # Removing x values which are not in xRange
    if xMin > iniCond[0]:
        for i in range(len(xl)):
            if xl[i] < xMin:
                continue
            else:
                xl = xl[i:]
                break
    if xMax < iniCond[0]:
        for i in range(len(xl)):
            if xl[-1] > xMax:
                continue
            else:
                xl = xl[:i]
                break

    return xl, yl, zl


#Partial Differential Equation Solvers
#Diffusion equations
#Explicit
def explicitPDEHeat(inispace,a,b,k,dt,dx,tsteps = 200):
    """
    d_t = k*d_xx

    a = left most boundary conditions
    b = right most boundary conditions

    inispace = initial condition

    dt = 1 timestep
    dx = 1 spacestep

    tsteps = no of timesteps
    """
    N = len(inispace)
    alpha = (k*dt)/(dx**2)

    #Loop for multiple time steps
    for t in range(1,tsteps+1):
        new = np.zeros(N)
        #Update the rest places
        for i in range(1,N-1):
            new[i] = inispace[i] + alpha*(inispace[i+1]+inispace[i-1]-2*inispace[i])

            #Update Boundaries
            new[0] = a(t*dt)
            new[-1] = b(t*dt)

        # Update previous column to new column
        inispace = copy.deepcopy(new)

    return new

#Implicit
def implicitPDEHeat(inispace,a,b,k,dt,dx,tsteps = 200):
    """
    d_t = k*d_xx

    a = left most boundary functions
    b = right most boundary functions

    inispace = initial condition

    dt = 1 timestep
    dx = 1 spacestep

    tsteps = no of timesteps
    """

    N = len(inispace)
    alpha = (k*dt)/(dx**2)

    #Create an array which only does not have boundary cases.
    ini = np.delete(inispace, [0,-1])
    new = np.zeros(N-2)

    #Creating matrix for multiplication
    A = np.zeros((N-2,N-2))
    for i in range(N-2):
        for j in range(N-2):
            if(i==j): A[i][j] = 1+(2*alpha)
            if(abs(i-j)==1): A[i][j] = -alpha
    
    A = Inverse(A)


    # Loop for timesteps
    for t in range(tsteps):

        #Update 2nd and 2nd-last point
        ini[0] = ini[0]+(alpha*a(t*dt))
        ini[-1] = ini[-1]+(alpha*b(t*dt))

        new = np.dot(A,ini)

        ini = copy.deepcopy(new)
    
    new = np.insert(new,0,a(tsteps*dt))
    new = np.append(new,b(tsteps*dt))

    return new


def Laplace2D(A, maxsteps, convergence):
    """
    Relaxes the matrix A until the sum of the absolute differences
    between the previous step and the next step (divided by the number of
    elements in A) is below convergence, or maxsteps is reached.

    Input:
     - A: matrix to relax
     - maxsteps, convergence: Convergence criterions

    Output:
     - A is relaxed when this method returns
    """

    iterations = 0
    diff = convergence +1

    Nx = A.shape[1]
    Ny = A.shape[0]
    
    while iterations < maxsteps and diff > convergence:
        #Loop over all *INNER* points and relax
        Atemp = A.copy()
        diff = 0.0
        
        for y in range(1,Ny-1):
            for x in range(1,Ny-1):
                A[y,x] = 0.25*(Atemp[y,x+1]+Atemp[y,x-1]+Atemp[y+1,x]+Atemp[y-1,x])
                diff  += math.fabs(A[y,x] - Atemp[y,x])

        diff /=(Nx*Ny)
        iterations += 1
        #print("Iteration #", iterations, ", diff =", diff)
    
    return A





#Gauss Quadrature
def modi_func_T(input_func, y, a, b):             ## func is multiplied by W, limit is changed (chebyshcev)
    y2 = ((a+b)/2) + ((b-a)*y/2)

    # For legendre gauss, remove the multiplied sqrt term
    return input_func(y2) * ((b-a)/2)

def chebyshev_T_sw(n):                          ## returns s(abscissa) and w(weights) lists of a func in a given limit and degree
    s, w = [], []
    for i in range(1,n+1):
        s.append(np.cos( (np.pi * (i-0.5)) / n))
        w.append(np.pi / n)
    return s, w

#print(chebyshev_T_sw(26))
#print(np.polynomial.chebyshev.chebgauss(26))   Numpy version of chebyschev-gauss
#polynomial.legendre.leggauss(26) Numpy version of legendre-gauss

# Weighted sum for integral
def gaussInte(func, l, u ,deg):
    s, w = chebyshev_T_sw(deg)
    sum = 0
    for i in range(len(s)):
        sum += w[i] * modi_func_T(func, s[i], l, u)

    return sum