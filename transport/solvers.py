# -*- coding: utf-8 -*-


'''solvers for ordinary differential equations (ode).'''




import numpy as np
from builtins import range


def numerov(q, y0, y1, dx, full):
    '''integrate ode of type y''(x) + q(x)*y(x) = 0.
    
    Parameters
    ----------
    q : array-like
        function q(x) discretized at sampling points
        x_m = m * dx with m = 0 ... n-1.
    y0, y1 : scalar
        initial values given by y0 = y(0) and y1 = y(dx).
    dx : scalar
        step size to discretize functions.
    full : bool
        return full solution y_0, y_1, ..., y_(n-1) or 
        only the last two values y_(n-2), y_(n-1).
    '''
    
    # number of sampling points
    n = len(q)
    
    # convert array-like to array
    q = np.asarray(q)
    
    # coefficients appearing in Numerov iteration
    # a[i]*y[i] = b[i-1]*y[i-1] - a[i-2]*y[i-2]
    a = 12 + dx*dx * q
    b = 24 - 10*dx*dx * q
    
    
    if full:
        # calculate and return full solution.
        # extract more general type
        y = np.empty(n, dtype=type(y0+y1))
        y[0:2] = (y0, y1)
        
        # iterate Numerov algorithm
        for i in range(2, n):
            y[i] = (b[i-1]*y[i-1] - a[i-2]*y[i-2]) / a[i]
        
        return y
    
    else:
        # calculate and return last two values.
        # iterate Numerov algorithm
        for i in range(2, n):
            y0, y1 = y1, (b[i-1]*y1 - a[i-2]*y0) / a[i]
        
        return y0, y1
