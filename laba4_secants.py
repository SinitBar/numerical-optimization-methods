import numpy as np

#f = lambda x: x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20
#df = lambda x: 2 * x + 3 * np.sin(0.3 * np.pi * x) * np.pi
#tol = 0.0000000001 # delta = 10^(-10)
#interval = [-2, 5]

def ssearch(interval, tol):
    a = interval[0]
    b = interval[1]
    neval = 1
    coords = []
    x_k = b - df(b) * (b - a) / (df(b) - df(a))
    while(((np.abs(df(x_k))) > tol) and ((np.abs(b - a)) > tol)):
        df_b = df(b)
        x_k = b - df_b * (b - a) / (df_b - df(a))
        if (df(x_k) > 0):
            b = x_k
        else:
            a = x_k
        neval += 4
        coords.append([x_k, a, b])
    xmin = x_k
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

#print(ssearch(interval, tol))
