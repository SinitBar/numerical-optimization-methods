import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    #PLACE YOUR CODE HERE
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    xmin = interval[0]
    L = np.abs(b - a)
    x1 = b - L / Phi # golden ratio has two points when left part is smaller then right
    x2 = a + L / Phi # and wneh it is bigger
    f_x1 = f(x1)
    f_x2 = f(x2)
    neval = 2

    while (np.abs(b - a) > tol):
        neval = neval + 1
        if (f_x1 > f_x2):
            a = x1
            xmin = x2
            x1 = x2 # order will be x2 x1 so change it back to x1 x2
            f_x1 = f_x2 # and remember found value
            L = np.abs(b - a)
            x2 = a + L / Phi
            f_x2 = f(x2)
        else:
            b = x2
            x2 = x1
            f_x2 = f_x1
            xmin = x1
            L = np.abs(b - a)
            x1 = b - L / Phi
            f_x1 = f(x1)
    
    if (xmin == x1):
        fmin = f_x1
    else:
        fmin = f_x2
    answer_ = [xmin, fmin, neval]
    return answer_


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def sdsearch(f, df, x0, tol):

# SDSEARCH searches for minimum using steepest descent method
# 	answer_ = sdsearch(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics
   
    #PLACE YOUR CODE HERE
    kmax = 1000
    k = 0
    deltaX = tol + 1
    x_k = x0
    coords = []
    neval = 0
    while(norm(deltaX) >= tol) and (k < kmax):
        k += 1
        df_x_k = df(x_k)
        f1dim = lambda a: f(x_k - a * df_x_k)
        foundArray = goldensectionsearch(f1dim, [0, 1], tol) # finds alpha from 0 to 1
        alpha = foundArray[0]
        deltaX = alpha * df_x_k
        x_k = x_k - deltaX
        #coords.append(x_k)
        neval += 1
    xmin = x_k
    fmin = f(x_k)
    answer_ = [xmin, fmin, neval, coords]
    return answer_


tol1 = 0.001
tol2 = 0.00001
print(sdsearch(fH, dfH, [1.3, 2], tol1))
print(sdsearch(fR, dfR, [1.0, -2], tol2)) 