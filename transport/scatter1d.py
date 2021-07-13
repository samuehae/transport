# -*- coding: utf-8 -*-


'''one-dimensional time-independent quantum scattering problem.

Schroedinger equation (dimensionless form)
------------------------------------------
y''(x) + (e - v(x)) * y(x) = 0

x : dimensionless coordinate measured in arbitrary length s
y(x) : dimensionless wave function measured in 1/sqrt(s)
e : dimensionless particle energy measured in hbar^2 / (2*m*s^2)
v(x) : dimensionless potential measured in hbar^2 / (2*m*s^2). 
    potential zero within lead regions (x < 0 and x > l) and
    possibly non-zero within scattering region (0 <= x <= l).


Free propagation ansatz
-----------------------
General ansatz (dimensionless wave vector k = sqrt(e))
y(x) = a*exp(ikx) + b*exp(-ikx) for x < 0
y(x) = c*exp(ikx) + d*exp(-ikx) for x > l


Right (x > l) incident particle
y(x) = b*exp(-ikx) for x < 0
y(x) = c*exp(ikx) + d*exp(-ikx) for x > l

reflection amplitude r = c/d
transmission amplitude t = b/d
'''




import numpy as np
from transport.solvers import numerov


def amplitudes(e, v, dx):
    '''returns reflection and transmission amplitudes r and t for 
    right incident particle (more details in module's docstring).
    
    Parameters
    ----------
    e : scalar
        dimensionless particle energy measured in hbar^2 / (2*m*s^2)
    v : array-like
        dimensionless potential within scattering region 
        measured in hbar^2 / (2*m*s^2)
    dx : scalar
        step size to discretize potential and wave function. 
        measured in arbitrary length s
    '''
    
    # number of sampling points
    n = len(v)
    
    # convert array-like to array
    v = np.asarray(v)
    
    # set up Schroedinger equation y''(x) + q(x)*y(x) = 0 with q(x) = e - v(x). 
    # additional sampling points in each lead region used to set up initial 
    # values and to match solution with free propagation ansatz
    q = np.concatenate(((e, e), e-v, (e, e)))
    
    # wave vector in lead regions
    k = np.sqrt(e)
    
    # initial values in left lead (x < 0)
    # set parameter b = exp(-ik dx)
    y0 = np.exp(1J*k*dx)    # y(x) at x = -2dx
    y1 = 1.0                # y(x) at x = -dx
    
    
    # calculate last two values of wave function in right lead region. 
    # used to match to free propagation ansatz
    y0, y1 = numerov(q, y0, y1, dx, full=False)
    
    
    # match numerical solution with free propagation ansatz
    det = np.exp(1J*k*dx) - np.exp(-1J*k*dx)
    
    d = (np.exp(1J*k*(n+1)*dx) * y0 - np.exp(1J*k*n*dx) * y1) / det
    c = (-np.exp(-1J*k*(n+1)*dx) * y0 + np.exp(-1J*k*n*dx) * y1) / det
    
    b = np.exp(-1J*k*dx)
    
    
    # return reflection and transmission amplitude
    return c/d, b/d



def wavefunction(e, v, dx):
    '''returns wave function within scattering region for 
    right incident particle (normalization d = 1).
    
    Parameters
    ----------
    e : scalar
        dimensionless particle energy measured in hbar^2 / (2*m*s^2)
    v : array-like
        dimensionless potential within scattering region 
        measured in hbar^2 / (2*m*s^2)
    dx : scalar
        step size to discretize potential and wave function. 
        measured in arbitrary length s
    '''
    
    
    # number of sampling points
    n = len(v)
    
    # convert array-like to array
    v = np.asarray(v)
    
    # wave vector in lead regions
    k = np.sqrt(e)
    
    # set up Schroedinger equation y''(x) + q(x)*y(x) = 0 with q(x) = e - v(x). 
    # additional sampling points in each lead region used to set up initial 
    # values and to match solution with free propagation ansatz
    q = np.concatenate(((e, e), e-v, (e, e)))
    
    # initial values in left lead (x < 0)
    # set parameter b = exp(-ik dx)
    y0 = np.exp(1J*k*dx)    # y(x) at x = -2dx
    y1 = 1.0                # y(x) at x = -dx
    
    
    # calculate full wave function in scattering region 
    # including two values on each side in the leads.
    y = numerov(q, y0, y1, dx, full=True)
    
    
    # extract last two values of wave function in right lead
    y0, y1 = y[-2], y[-1]
    
    
    # match numerical solution with free propagation ansatz
    # to normalize wave function (d = 1)
    det = np.exp(1J*k*dx) - np.exp(-1J*k*dx)
    
    d = (np.exp(1J*k*(n+1)*dx) * y0 - np.exp(1J*k*n*dx) * y1) / det
    
    # return wave function within scattering region
    # remove concatenated points and normalize
    return y[2:-2] / d
