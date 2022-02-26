import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv



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


def goldensectionsearch(f, interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    y1 = f(x1)
    y2 = f(x2)
    neval = 2
    xmin = x1
    fmin = y1

    # main loop
    while np.abs(L) > tol:
        if y1 > y2:
            a = x1
            xmin = x2
            fmin = y2
            x1 = x2
            y1 = y2
            L = b - a
            x2 = a + L / Phi
            y2 = f(x2)
            neval += 1
        else:
            b = x2
            xmin = x1
            fmin = y1
            x2 = x1
            y2 = y1
            L = b - a
            x1 = b - L / Phi
            y1 = f(x1)
            neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_

# linear search of minimizer in trusted region
def pparam(pU, pB, tau):
    if (tau <= 1):
        p = np.dot(tau, pU)
    else:
        p = pU + (tau - 1) * (pB - pU)
    return p

# mod is a model, Delta - current trusted radius delta_k
def doglegsearch(mod, g0, B0, Delta, tol):
    # dogleg local search
    xcv = np.dot(-g0.transpose(), g0) / np.dot(np.dot(g0.transpose(), B0), g0)
    pU = xcv *g0
    xcvb = inv(- B0)
    pB = np.dot(inv(- B0), g0)

    func = lambda x: mod(np.dot(x, pB))
    al = goldensectionsearch(func, [-Delta / norm(pB), Delta / norm(pB)], tol)[0]
    pB = al * pB
    func_pau = lambda x: mod(pparam(pU, pB, x))
    tau = goldensectionsearch(func_pau, [0, 2], tol)[0]
    pmin = pparam(pU, pB, tau)
    if norm(pmin) > Delta:
        pmin_dop = (Delta / norm(pmin))
        pmin = np.dot(pmin_dop, pmin)
    return pmin


def trustreg(f, df, x0, tol):
# TRUSTREG searches for minimum using trust region method
# 	answer_ = trustreg(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords, radii]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics
#   radii - array of trust regions radii
   
    #PLACE YOUR CODE HERE
    k = 0
    kmax = 1000
    neval = 0
    coordinates = []
    radii = []
    x_k = x0
    f_x_k = f(x_k)
    df_x_k = df(x_k)
    p_k = -df_x_k
    delta_max = 0.1
    delta_k = delta_max
    ro_k = 1
    B_k = np.zeros((2, 2))
    B_k[0][0] = 1
    B_k[1][1] = 1
    eta = 0.1
    d_k = p_k
    radii.append(delta_k)

    while ((norm(p_k) >= tol) and (k < kmax)):
        k += 1
        coordinates.append(x_k)
        mod = lambda p: f_x_k + np.dot(p.transpose(), df_x_k) + 0.5 * (np.dot(p.transpose(), np.dot(B_k, p)))
        p_k = doglegsearch(mod, df_x_k, B_k, delta_k, tol)
        f_x_k_prev = f_x_k
        f_x_k = f(x_k + p_k)
        ro_k = (f_x_k_prev - f_x_k) / (mod(np.zeros((2, 1))) - mod(p_k))
        if (np.abs(ro_k) > eta):
            x_k = x_k + p_k
            d_k = p_k
        else:
            break # because d_k = 0
        if (np.abs(ro_k) < 1 / 4):
            delta_k = delta_k / 4
        else: 
            if ((np.abs(ro_k) > 3 / 4) and (norm(p_k) == delta_k)):
                delta_k = min(2 * delta_k, delta_max)
        radii.append(delta_k)
        df_x_k_prev = df_x_k
        df_x_k = df(x_k)
        neval += 1
        y_k = df_x_k - df_x_k_prev
        first = (np.dot(y_k, y_k.transpose())) / (np.dot(y_k.transpose(), d_k))
        up = np.dot(B_k, np.dot(d_k, np.dot(d_k.transpose(), B_k)))
        down = np.dot(d_k.transpose(), np.dot(B_k, d_k))
        B_k = B_k + first - up / down

    coordinates.append(x_k)
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coordinates, radii]
    return answer_


#tol = 1e-3
#print(trustreg(fH, dfH, np.array([[2.0], [1.0]]), tol))
#print(trustreg(fR, dfR, np.array([[-2], [0]]), tol))
