import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
np.seterr(divide='ignore', invalid='ignore')


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


def nagsearch(f, df, x0, tol):
    
# NAGSEARCH searches for minimum using the Nesterov accelerated gradient method
# 	answer_ = nagsearch(f, df, x0, tol)
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
    k = 0
    kmax = 1000
    al = 0.05
    y_k = x0
    x_k = x0
    nu_k = al / 10
    gamma_k = 0.75
    df_y_k = df(y_k)
    neval = 1
    coordinates = []

    while ((norm(df_y_k) >= tol) and (k < kmax)):
        coordinates.append(x_k)
        k += 1
        x_k_prev = x_k
        x_k = y_k - nu_k * df_y_k
        y_k = x_k + gamma_k * (x_k - x_k_prev)
        df_y_k = df(y_k)
        neval += 1
    coordinates.append(x_k)
    xmin = x_k
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coordinates]
    return answer_



#x0 = np.array([[0.0], [1.0]])
#tol = 1e-8
#print(nagsearch(fH, dfH, x0, tol))
