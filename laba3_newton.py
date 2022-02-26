import numpy as np

#f = lambda x: x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20
#df = lambda x: 2 * x + 3 * np.sin(0.3 * np.pi * x) * np.pi
#ddf = lambda x: 2 + 0.9 * np.cos(0.3 * np.pi * x) * np.pi * np.pi
#tol = 0.0000000001 # delta = 10^(-10)
#interval = [-2, 7]
#x0 = 1.3

def nsearch(interval, tol, x0):
    neval = 0
    coords = []
    x_k = x0
    while((np.abs(df(x_k))) > tol):
        neval += 3
        coords.append(x_k)
        x_k = x_k - df(x_k) / ddf(x_k)
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

#print(nsearch(interval, tol, x0))
