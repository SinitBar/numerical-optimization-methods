import numpy as np
import sys
from numpy.linalg import norm
#np.seterr(divide='ignore', invalid='ignore')


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


def zoom(phi, dphi, alo, ahi, c1, c2):
    j = 1
    jmax = 1000
    while j < jmax:
        a = cinterp(phi, dphi, alo, ahi)
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a
        else:
            if abs(dphi(a)) <= -c2 * dphi(0):
                return a  # a is found
            if dphi(a) * (ahi - alo) >= 0:
                ahi = alo
            alo = a
        j += 1
    return a


def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        a = a0
        return a
 
    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        a = a0
        return a
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)

    return a


def wolfesearch(f, df, x0, p0, amax, c1, c2):
    a = amax
    aprev = 0
    phi = lambda x: f(x0 + x * p0)
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))

    phi0 = phi(0)
    dphi0 = dphi(0)
    i = 1
    imax = 1000
    while i < imax:
        if (phi(a) > phi0 + c1 * a * phi0) or ((phi(a) >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2)
            return a

        if abs(dphi(a)) <= -c2 * dphi0:
            return a  # a is found already

        if dphi(a) >= 0:
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a

        a = cinterp(phi, dphi, a, amax)
        i += 1


    return a


def dfpsearch(f, df, x0, tol):
    # DFPSEARCH searches for minimum using DFP method
# 	answer_ = dfpsearch(f, df, x0, tol)
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
    x_k = x0
    neval = 1
    coordinates = []
    k = 0
    kmax = 1000
    amax = 3
    H_k = np.zeros((2, 2)) # inverse Hessian matrix approximation
    H_k[0][0] = 1
    H_k[1][1] = 1
    #H_k = np.array([[1, 0], [0, 1]]) # doesn't work like this wtf?!
    # interesting: when take H_k = 1 1, 1 1 like above, get all answers right 
    d_k = tol + 1 # quasi-newtonian vector
    c1 = tol
    c2 = 0.1
    df_x_k = df(x_k)

    while ((norm(d_k) >= tol) and (k < kmax)):
        coordinates.append(x_k)
        k += 1
        neval += 1
        coordinates.append(x_k)
        p_k = np.dot(-H_k, df_x_k) # search direction
        alpha_k = wolfesearch(f, df, x_k, p_k, amax, c1, c2)
        d_k = alpha_k * p_k
        x_k = x_k + d_k
        df_x_k_prev = df_x_k
        df_x_k = df(x_k)
        y_k = df_x_k - df_x_k_prev
        up = np.dot(H_k, y_k)
        up = np.dot(up, y_k.transpose())
        up = np.dot(up, H_k)
        down = np.dot(y_k.transpose(), H_k)
        down = np.dot(down, y_k)
        first = np.dot(d_k, d_k.transpose()) / np.dot(d_k.transpose(), y_k)
        second = up / down
        H_k = H_k + first - second
    coordinates.append(x_k)
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coordinates]
    return answer_

#tol = 1e-9
#print(dfpsearch(fH, dfH, np.array([[1.0], [0.0]]), tol)) # (3, 2) = 0
#print(dfpsearch(fR, dfR, np.array([[-3], [-3]]), tol)) # (1, 1) = 0
