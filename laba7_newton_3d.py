import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
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
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)

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
    v = (1 - x)**2 + 100*(y - x**2)**2
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
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v



def H(X, tol, df):
   #PLACE YOUR CODE HERE
   # df = [dx, dy] all this in some dot = f value
   deltaX = tol * 0.1
   # easy to drop next 4 strings but let them be to let code be more readable
   x0 = X[0]
   y0 = X[1]
   df_x = lambda dot: df(dot)[0]
   df_y = lambda dot: df(dot)[1]
   dxdx = (df_x([x0 + deltaX, y0]) - df_x([x0 - deltaX, y0])) / (2 * deltaX)
   dydy = (df_y([x0, y0 + deltaX]) - df_y([x0, y0 - deltaX])) / (2 * deltaX)
   dxdy = (df_x([x0, y0 + deltaX]) - df_x([x0, y0 - deltaX])) / (2 * deltaX)
   #ddf = [[dxdx, dxdy], [dxdy, dydy]] # Hessian matrix
   # to pass tests i did like this instead of 79 string:
   # Hessian matrix
   ddf = np.zeros((2, 2))
   ddf[0][0] = dxdx
   ddf[0][1] = dxdy
   ddf[1][0] = dxdy
   ddf[1][1] = dydy
   # if in 79 str add [] over every element, see error like on site
   return ddf


def nsearch(f, df, x0, tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(f, df, x0, tol)
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
        coords.append(x_k)
        H0 = H(x_k, tol, df)
        neval += 1
       # print(H0)
       # print(df(x_k))
        deltaX = np.linalg.lstsq(H0, df(x_k))[0]
        x_k = x_k - deltaX
    coords.append(x_k)
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval,  coords]
    return answer_


#tol = 0.001
#tol1 = 0.000000001
#print(nsearch(fH, dfH, [-2.0, -2.0], tol))
#print(nsearch(fR, dfR, [-1, -1], tol1)) 
