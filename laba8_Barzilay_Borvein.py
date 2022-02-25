import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
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


def bbsearch(f, df, x0, tol):

# BBSEARCH searches for minimum using stabilized BB1 method
# 	answer_ = bbsearch(f, df, x0, tol)
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
    D = 0.1 # stabilizing parameter
    kmax = 1000
    k = 0
    x_k = x0
    g_k_prev = df(x0)
    g_k = g_k_prev
    f1dim = lambda a: f(x_k - a * g_k)
    foundArray = goldensectionsearch(f1dim, [0, 1], tol)
    alpha_k = foundArray[0]
    
    coords = []
    neval = 0
    deltaX = tol + 1
    
    while(norm(deltaX) >= tol) and (k < kmax):
        coords.append(x_k)
        k += 1
        neval += 1
        deltaX = -alpha_k * g_k
        x_k = x_k + deltaX
        g_k_prev = g_k
        g_k = df(x_k)
        deltaG = g_k - g_k_prev
        alpha_k = np.dot(deltaX.transpose(), deltaX) / np.dot(deltaX.transpose(), deltaG)
        alpha_stab = D / norm(g_k)
        alpha_k = min(alpha_k, alpha_stab)
    coords.append(x_k)
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

tol = 1e-9 # 10^(-9)
print(bbsearch(fR, dfR, np.array([[2.0], [-1.0]]), tol)) 
