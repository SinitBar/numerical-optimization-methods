import numpy as np

f = lambda x: 2 * x ** 2 - 9 * x - 31
df = lambda x: 4 * x - 9
tol = 0.0000000001 # delta = 10^(-10)
a_start = -2
b_end = 10
interval = [a_start, b_end]

def bsearch(interval, tol):
    xmin = interval[0]
    a = interval[0]
    b = interval[1]
    neval = 0
    coords = [a, b]

    while ((np.abs(b - a) > tol) and (np.abs(df(a)) > tol)):
        xmin = (a + b) / 2
        coords.append(xmin)
        #neval = neval + 1 
        if (df(xmin) > 0):
            b = xmin
        else:
            a = xmin
    fmin = f(xmin)
    neval = neval + 1
    answer_ = [xmin, fmin, neval, coords]
    return answer_

#print(bsearch(interval, tol))
