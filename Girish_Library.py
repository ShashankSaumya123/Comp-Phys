#IMPORTS
import numpy as np
import math
from math import log,sqrt
import random
from matplotlib import pyplot as plt 

e = 2.718281828459045 

#############################################################################################################################################################
#############################################################################################################################################################

def read_matrix(str,col,row):
    A = [[0 for i in range(col)]for j in range(row)]       #MAtrix that stores the matrix for inverting
    file = open(str,"r+")     #Creating a file object with the matrix file
    for i in range(row):          #Creating the matrix to be inverted
        for j in range(col):
            A[i][j] = int(file.readline())
    return A

#############################################################################################################################################################
#############################################################################################################################################################

def usual_matrix(matrix):       #Function that Prints the matrix in usual form
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j],end = '   ')
        print("\n")

#############################################################################################################################################################
#############################################################################################################################################################

def multiply_matrix(A,B):       #Function that do the matrix multiplication
    if len(A[0]) == len(B):     #checks for valid matrices (A coloum = B row)
        cross_mat = [[0 for i in range(len(B[0]))]for j in range(len(A))]   #stores the cross multiplied matrix
        for i in range(len(A)):        #iterating along rows
            for j in range(len(B[0])):        #iterating along column
                for m in range(len(A)):        
                    cross_mat[i][j] = cross_mat[i][j] + A[i][m]*B[m][j]
        return cross_mat
    else:
        print("Please input two valid matrices")

#############################################################################################################################################################
#############################################################################################################################################################

def Augmented(A,B):     #Function that augment 2 matrices   
    aug_AB = [[0 for a in range(len(A)+len(B[0]))]for b in range(len(A[0]))]       #Stores the augmented matrix
    for i in range(len(A)):      #Creating an augmented matrix
        l = 0
        for j in range(len(A)+len(B[0])):         
            if j>=6:
                aug_AB[i][j] = B[i][l]
                l = l+1
            else:
                aug_AB[i][j] = A[i][j]
            
    return aug_AB

#############################################################################################################################################################
#############################################################################################################################################################

def Gauss_jorden(aug_mat):      #Function that do the gauss_jorden
    for i in range(len(aug_mat)):
        p = aug_mat[i][i]
        for j in range(len(aug_mat[0])):
            aug_mat[i][j] = aug_mat[i][j]/p
        for k in range(len(aug_mat)):
            if k == i or aug_mat[k][i] == 0:
                next
            else:
                factor = aug_mat[k][i]
                for j in range(len(aug_mat[0])):
                    aug_mat[k][j] = aug_mat[k][j] - factor*aug_mat[i][j]

    return aug_mat

#############################################################################################################################################################
#############################################################################################################################################################

def GaussSidel(A,b,tol):            #Function that does Gauss Sidel ALgorithm to solve Linear equation with dominant diagonal
    epsilon=tol    
    x = [1,0,0,0,0,0]
    count=0
    temp=1
    while temp>epsilon:
        temp = 0.0 
        temp1 = 0.0
        count = count + 1
        print(x)
        for i in range(len(x)):
            sums1 = sums2 = 0
            for j in range(1,i):
                sums1 = sums1 + A[i][j]*x[j]
            for k in range(i+1,len(x)):
                sums2 = sums2 + A[i][k]*x[k]
            temp1 = x[i]
            x[i]=(b[i]-sums1-sums2)/A[i][i]
            temp = temp + abs(temp1-x[i])
    return x

#############################################################################################################################################################
#############################################################################################################################################################

def conjGrad(A,x,b):        #Function that does conjugate gradiant algorithm to solve linear equation with Dominant diagonal
    tol = 0.00001
    n = len(b)
    r = b - np.dot(A,x)
    d = r.copy()
    i = 1
    while i<=n:
        u = np.dot(A,d)
        alpha = np.dot(d,r)/np.dot(d,u)
        x = x + alpha*d
        r = b - np.dot(A,x)
        if np.sqrt(np.dot(r,r)) < tol:
            break
        else:
            beta = -np.dot(r,d)/np.dot(d,u)
            d = r + beta*d
            i = i+1

    return x

#############################################################################################################################################################
#############################################################################################################################################################


def powerMethodHelper(A,x):         #Function that does power method to find eigen values
    oldEigenVal = 0
    tol = 0.0001
    while True:
        x = np.dot(A,x)             
        eigenVal = max(abs(x))
        x = x/eigenVal              
        
        if abs(oldEigenVal-eigenVal)>tol:
            oldEigenVal=eigenVal
        else:
            break
    return eigenVal,x

#############################################################################################################################################################
#############################################################################################################################################################

def powerMethod(A,x,n):         #FUnction that helps to find multiple eigen values where lambda 1>lambda2>lambda3..
    eigMat = []
    eigvmat = []
    E,V = powerMethodHelper(A,x)        
    eigMat.append(E)
    if n>1:
        count = n-1
        while count != 0:
            V = V/np.linalg.norm(V)
            V = np.array([V])
            A = A - (E*V*V.T)
            E,V = powerMethodHelper(A,x)
            eigMat.append(E)
            eigvmat.append(V)
            count = count - 1 
    return eigMat,eigvmat

#############################################################################################################################################################
#############################################################################################################################################################

def maxoff(A):              #Function that find largest off diagonal element in a symmetric metrix
    maxtemp = A[0][1]           #Let the first elem be the largest one
    k = 0
    l = 1
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > abs(maxtemp):         #COmpare every off diagonal element with the 'let' element
                maxtemp = A[i][j]
                k = i
                l = j
    return maxtemp, k, l

#############################################################################################################################################################
#############################################################################################################################################################

def jacobigg(A,b,tol):          #Function that do Jacobi method to find SOltion of linnear equation
    epsilon = tol
    x = [1,0,0,0,0,0]
    D = np.diag(A)                      
    LU = np.array(A) - np.diagflat(D)           #A = D + (L + U)
    xnew = (b - np.dot(LU,x))/D
    print(x)
    while np.linalg.norm(xnew - x)>epsilon: 
        x = xnew  
        xnew = (b - np.dot(LU,x))/D
        print(xnew)
    return xnew


#############################################################################################################################################################
#############################################################################################################################################################


def JacobiGivan(A):          #Function that do Jacobi method with Given's rotation
    epsilon = 0.0004
    max,i,j = maxoff(A)             #Max off diagonal element and its indices
    while abs(max) >= epsilon:              #Loop this off diagonal element tends to 0
        if A[i][i] - A[j][j] == 0:
            theta = math.pi / 4
        else:
            theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2

        Q = [[1,0,0],[0,1,0],[0,0,1]]           #Rotation matrix
        Q[i][i] = Q[j][j] = math.cos(theta)             #Qii = Qjj = cos(theta)
        Q[i][j] = -1*math.sin(theta)                    #Qij = -Qji = sing(theta)
        Q[j][i] = math.sin(theta)   
        AQ = multiply_matrix(A,Q) 
        Q = np.array(Q)
        QT = Q.T.tolist()
        A = multiply_matrix(QT,AQ)                  #New A for next iteration = Q_inverse * A * Q. Here Q_inverse = Q_transpose, cuz rotation matrix is orthogonal 
        max,i,j = maxoff(A)
    return A
    
#############################################################################################################################################################
#############################################################################################################################################################

def findMean(A):    #FUnction to find mean
    n = len(A)
    sum = 0
    mean = 0
    for i in range(n):
        sum = sum + A[i]
    return sum/n

#############################################################################################################################################################
#############################################################################################################################################################


def findVariance(A):        #Function to find variance
    n = len(A)
    mean = findMean(A)
    sum = 0
    for i in range(n):
        sum = sum + (A[i]-mean)**2
    return sum/n

#############################################################################################################################################################
#############################################################################################################################################################

def jackKnifeVarHelper(yi,yibar):               #FInd variance for Jackknife algorithm
    n = len(yi)
    sum = 0
    for i in range(n):
        sum = sum + (yi[i] - yibar)**2
    return ((n-1)/n)*sum

#############################################################################################################################################################
#############################################################################################################################################################

def jackKnife(A):       #Function that do Jackknife method of resampling
    n = len(A)
    yi = []    
    print(f"array = {A} , mean = {findMean(A)}")
    for i in range(n):
        B = A.copy()                    #Save a A copy to use in furthur iterations
        del(B[i])                       # N -> N-1  
        mean = findMean(B)              #Indidual resampled mean
        print(f"array = {B} , mean = {findMean(B)}")
        yi.append(mean)
    yibar = findMean(yi)
    print(yi)
    print(yibar)
    var = jackKnifeVarHelper(yi,yibar)
    print(var)

#############################################################################################################################################################
#############################################################################################################################################################

def bootstrap(A,b):     #Function that do Bootstrap algorithm to find mean and variance of a sample
    meanarr = []
    vararr = []
    n = len(A)
    for i in range(b):
        resample = random.choices(A,k=len(A))       #Resampling
        m = findMean(resample)                  #Mean of individual sample
        var = findVariance(resample)            #Var of indivudal sample
        meanarr.append(m)
        vararr.append(var)    
    print(findMean(meanarr))
    print(findMean(vararr))
    plt.hist(meanarr)

#############################################################################################################################################################
#############################################################################################################################################################

def chiLinear(x,y,sig):     #Function that do chi^2 linear fitting
    n = len(sig)
    s = sx = sy = sxx = sxy = wt = syy = 0
    for i in range(n):              #As given in notes
        wt = (1/sig[i]**2)
        s = s + wt
        sx = sx + x[i]*wt
        sy = sy + y[i]*wt
        sxx = sxx + (x[i]**2)*wt
        syy = syy + (y[i]**2)*wt
        sxy = sxy + x[i]*y[i]*wt

    delta = s*sxx - (sx)**2
    a = ((sxx*sy) - (sx*sxy))/delta         #Paramter a
    b = ((s*sxy) - (sx*sy))/delta           #Paramter
    siga = sxx/delta                        
    sigb = s/delta
    covab = -sx/delta
    r2 = sxy/(sxx*syy)
    dof = n - 2                             #Degree of Freedom = N - M where N is observation and M is no. of paramters
    return a,b,siga,sigb

#############################################################################################################################################################
#############################################################################################################################################################

def chi2(x,y,a,b,sig):                  #FUnction that finds chi2
    n = len(x)
    sum = 0
    for i in range(n):
        temp = (y[i] - a - b*x[i])/sig[i]
        sum = sum + (temp)**2

    return sum

#############################################################################################################################################################
#############################################################################################################################################################

def fwd_sbst(L,B):          #Function that do fwd substitution takes in input L and B
    y = [[0 for i in range(1)]for j in range(len(B))]   #U * x = y
    for i in range(len(L)):
        s = 0
        if i == 0:
            y[i][0] = B[i][0]
        else:
            s = sum(L[i][j] * y[j][0] for j in range(i))    #The sigma (sum) function
            y[i][0] = B[i][0] - s
    return y        #return the y

#############################################################################################################################################################
#############################################################################################################################################################

def bkwd_sbst(U,y):         #Function that do backward substitution
    x = [[0 for i in range(1)]for j in range(len(y))]
    for i in range(len(U)-1,-1,-1):
        if i == 3:
            x[i][0] = y[i][0]/U[i][i]
        else:
            s = 0
            s = sum(U[i][j]*x[j][0] for j in range(i+1,len(U)))
            x[i][0] = (y[i][0] - s)*(1/U[i][i])
    return x

#############################################################################################################################################################
#############################################################################################################################################################

def LUDecomposition(A):     #Function that do LUD takes in A and returns L and U
    L = [[0 for i in range(len(A[0]))]for j in range(len(A))]
    U = [[0 for i in range(len(A[0]))]for j in range(len(A))]
    for i in range(len(A)):
        for j in range(i,len(A)):       #Upper triangular matrix (j>i) gives U
            s = 0
            s = sum(L[i][k]*U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s
        
        for l in range(i,len(A)):       #Lower triangular matrix (j<i) gives L
            if l == i:
                L[i][i] = 1         #Diagonals are 1 in L
            else:
                s = 0
                s = sum(L[l][m]*U[m][i] for m in range(i))
                L[l][i] = (A[l][i] - s) * (1/U[i][i])
    return (L,U)

#############################################################################################################################################################
#############################################################################################################################################################

def fwd_bkwd_sbst(L,U,B):
    y = fwd_sbst(L,B)       #getting y from forward substitution
    x = bkwd_sbst(U,y)      #Getting x matrix or answer matrix from bakward Subst
    return x        
 
#############################################################################################################################################################
#############################################################################################################################################################
      
def get_determinant(L,U):     #Gets determinant
    sum_l = sum_u = 1
    for i in range(len(L)):     #Multiply the diagonals
        sum_l = sum_l * L[i][i]     #For L
        sum_u = sum_u * U[i][i]     #For U
    return sum_l*sum_u     #Determinant of A = Det(L) * Det(U)

#############################################################################################################################################################
#############################################################################################################################################################

def inv_mat(L,U):
    B = [[0 for i in range(len(L[0]))]for j in range(len(L))]       #stores the inverse
    I = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]       #the identity matrix
    b = [[0 for i in range(1)]for j in range(len(L))]      #That stores the column of B
    i = [[0 for i in range(1)]for j in range(len(L))]       #stores the column of I

    for h in range(len(B)):    
        for j in range(len(I)):
            i[j][0] = I[j][h]       #getting a column from I matrix
        b = fwd_bkwd_sbst(L,U,i)    #getting a inverse matrix column
        for k in range(len(B)):
            B[k][h] = round(b[k][0],5)      #storing the column in main inverse matrix
    return B        #returns the inverse matrix

#############################################################################################################################################################
#############################################################################################################################################################

def ChiPolyFit(x,y,sig,degree):     #FUnction that do polynomial model on Chi^2 method
    n = degree +1
    A = [[0 for i in range(n)]for j in range(n)]
    B = [[0] for i in range(n)]
    
    for i in range(n):
        for j in range(n):
            sum = 0
            for k in range(len(x)):
                sum = sum + (x[k]**(i+j))/(sig[k]**2)
            A[i][j] = sum
    
    for i in range(n):
        sum = 0
        for j in range(len(x)):
            sum = sum + ((x[j]**i)*y[j])/sig[j]**2
        B[i][0] = sum

    L,U = LUDecomposition(A)
    X = fwd_bkwd_sbst(L,U,B)
   

    return X
