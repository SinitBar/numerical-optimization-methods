import numpy as np

f = lambda x: 2 * x ** 2 - 9 * x - 31
tol = 0.0000000001 # delta = 10^(-10)
interval = [-2, 10]

def gsearch(interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    xmin = interval[0]
    coord = []
    L = np.abs(b - a)
    x1 = b - L / Phi # golden ratio has two points when left part is smaller then right
    x2 = a + L / Phi # and wneh it is bigger
    f_x1 = f(x1)
    f_x2 = f(x2)
    neval = 2

    while (np.abs(b - a) > tol):
        coord.append([x1, x2, a, b])
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
    answer_ = [xmin, fmin, neval, coord]
    return answer_

print(gsearch(interval, tol))