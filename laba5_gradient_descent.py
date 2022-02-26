import numpy as np
import sys
from numpy.linalg import norm

#F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def f(X):
    x = X[0]
    y = X[1]
    #Версия питона в codeboard не поддерживает метод библиотеки numpy float_power 
    v = (x**2 + y - 11)**2 + (x + y** 2 - 7)**2
    return v
    
# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def df(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return v


#x0 = 1.3
#tol = 0.0000000001 # delta = 10^(-10) #change to 10^(-3)

def grsearch(x0, tol):

# GRSEARCH searches for minimum using gradient descent method
# 	answer_ = grsearch(x0,tol)
#   INPUT ARGUMENTS
#	x0 - starting point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of x values found during optimization 

    kmax = 1000
    k = 0
    alpha_k = 0.01
    x_k = x0
    deltaX = tol + 1
    neval = 0
    coords = []

    while((norm(deltaX) >= tol) and (k < kmax)):
        k += 1
        neval += 1
        g_k = - df(x_k)
        deltaX = alpha_k * g_k
        x_k = x_k + alpha_k * g_k
        coords.append(x_k)

    xmin = x_k
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_

#grsearch(x0, tol)
