#importing libraries
import math

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
def gaussJordan(A, B, sizeA, sizeB):
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
def matMultiply(M, N, size):
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
def luDecompose(A, n):
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


#Function for Forward-Backward Substitution
#Works in complimentary to LU Decomposition function
def forwardBackwardSubstitution(A,B,m,n):
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
def monteCarloIntegral(f, lowerBound, upperBound, N, fileName):
    integral = 0
    width = upperBound - lowerBound
    h = width/N
    for i in range(N):
        integral += f(lowerBound + (width*random()))
    integral *= h
    #Appending the data to separate file
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
    - dz_dx: function, where x = dy/dx
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
            
        xl, zs = odeSolve(dz_dx, (guess[0] - dx, root2[0] + dx), (guess[0], guess[1]), dx = dx/2,)
        xl = list(map(lambda x: round(x, 6), xl))

        def dy_dx(x, y):
            return zs[xl.index(round(x, 6))]

        xl, yl = odeSolve(dy_dx, (guess[0], root2[0]), root1, dx = dx)
        iterationCount += 1
    return xl, yl