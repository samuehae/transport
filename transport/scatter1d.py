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
dimensionless wave vector k = sqrt(e) measured in 1/s

Left (x < 0) incident particle
y(x) = a*exp(ikx) + b*exp(-ikx) for x < 0
y(x) = c*exp(ikx) for x > l

Right (x > l) incident particle
y(x) = a*exp(-ikx) + b*exp(ikx) for x > l
y(x) = c*exp(-ikx) for x < 0

For both incidence directions
reflection amplitude r = b/a
transmission amplitude t = c/a
'''




import numpy as np
from transport.solvers import numerov


def amplitudes(e, v, dx, left):
    '''returns reflection and transmission amplitudes r and t.
    
    Parameters
    ----------
    e : scalar or array-like
        dimensionless particle energy measured in hbar^2 / (2*m*s^2)
    v : array-like
        dimensionless potential within scattering region 
        measured in hbar^2 / (2*m*s^2)
    dx : scalar
        step size to discretize potential and wave function. 
        measured in arbitrary length s
    left : bool
        solves scattering problem for left (right) incident particle 
        if argument is true (false).
    '''
    
    # number of sampling points
    n = len(v)
    
    # additional sampling points in each lead region used to set up initial 
    # values and to match solution with free propagation ansatz
    v = np.concatenate(((0, 0), v, (0, 0)))
    
    # wave vector in lead regions
    # scalar or array-like as given by energy e
    k = np.sqrt(e)
    
    
    # set up Schroedinger equation y''(x) + q(x)*y(x) = 0 with q(x) = e - v(x). 
    # speeds up calculation if energy e is array-like, as numerov solves 
    # scattering problems at different energies in a vectorized way
    # meaning of dimensions: (position, energy)
    q = e - v[:, np.newaxis]
    
    
    if left:
        # case: particle is incident from left lead
        # propagate initial values backwards (from right to left lead)
        q = q[::-1]
    
    # else:
        # case: particle is incident from right lead
        # propagate initial values forwards (from left to right lead)
        # use vector q unmodified
    
    
    
    # initial values made independent of particle moving direction
    
    # case: particle incident from left lead
    # initial values in right lead with c = exp(-ik n*dx)
    
    # case: particle incident from right lead
    # initial values in left lead with c = exp(-ik dx)
    
    y0 = np.exp(1J*k*dx)
    y1 = 1.0
    
    
    # integrate Schroedinger equation forward (backward) in space. 
    # calculate last two values of wave function in right (left) lead
    y0, y1 = numerov(q, y0, y1, dx, full=False)
    
    
    # match numerical solution with free propagation ansatz
    det = np.exp(1J*k*dx) - np.exp(-1J*k*dx)
    
    if left:
        a = (np.exp(2J*k*dx) * y0 - np.exp(1J*k*dx) * y1) / det
        b = (-np.exp(-2J*k*dx) * y0 + np.exp(-1J*k*dx) * y1) / det
        c = np.exp(-1J*k*n*dx)
        
    else:
        a = (np.exp(1J*k*(n+1)*dx) * y0 - np.exp(1J*k*n*dx) * y1) / det
        b = (-np.exp(-1J*k*(n+1)*dx) * y0 + np.exp(-1J*k*n*dx) * y1) / det
        c = np.exp(-1J*k*dx)
    
    
    # return reflection and transmission amplitudes
    return b/a, c/a



def wavefunction(e, v, dx, left):
    '''returns wave function within scattering region (normalization a = 1).
    
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
    left : bool
        solves scattering problem for left (right) incident particle 
        if argument is true (false).
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
    
    
    if left:
        # case: particle is incident from left lead
        # propagate initial values backwards (from right to left lead)
        q = q[::-1]
    
    # else:
        # case: particle is incident from right lead
        # propagate initial values forwards (from left to right lead)
        # use vector q unmodified
    
    
    
    # initial values made independent of particle moving direction
    
    # case: particle incident from left lead
    # initial values in right lead with c = exp(-ik n*dx)
    
    # case: particle incident from right lead
    # initial values in left lead with c = exp(-ik dx)
    
    y0 = np.exp(1J*k*dx)
    y1 = 1.0
    
    
    # calculate full wave function in scattering region 
    # including two values on each side in the leads.
    y = numerov(q, y0, y1, dx, full=True)
    
    # extract last two values of wave function in right (left) lead
    y0, y1 = y[-2], y[-1]
    
    
    # reverse wave function if propagated backwards
    if left:
        y = y[::-1]
    
    
    # match numerical solution with free propagation ansatz
    # to normalize wave function (normalization a = 1)
    det = np.exp(1J*k*dx) - np.exp(-1J*k*dx)
    
    if left:
        a = (np.exp(2J*k*dx) * y0 - np.exp(1J*k*dx) * y1) / det
    else:
        a = (np.exp(1J*k*(n+1)*dx) * y0 - np.exp(1J*k*n*dx) * y1) / det
    
    # return wave function within scattering region
    # remove concatenated points and normalize
    return y[2:-2] / a
