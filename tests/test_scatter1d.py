# -*- coding: utf-8 -*-


import pytest
import numpy as np
from transport import scatter1d



@pytest.mark.parametrize(('e', 'l', 'n'), [
    (0.1, 10, 50), 
    (0.2, 10, 100), 
    (1.0, 10, 200), 
    (1.1, 10, 230), 
])
def test_amplitudes_zero_potential(e, l, n):
    '''checks reflection and transmission amplitudes for zero potential.'''
    
    # expected scattering amplitudes for vanishing potential
    # reflection amplitude r = 0
    # transmission amplitude t = 1
    
    # scattering potential
    v = np.zeros(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical reflection and transmission amplitudes
    r_num, t_num = scatter1d.amplitudes(e, v, dx)
    
    
    # compare numerical and analytical solutions
    assert np.isclose(r_num, 0)
    assert np.isclose(t_num, 1)



@pytest.mark.parametrize(('e', 'l', 'n'), [
    (0.1, 10, 50), 
    (0.2, 10, 100), 
    (1.0, 10, 200), 
    (1.1, 10, 230), 
])
def test_wavefunction_zero_potential(e, l, n):
    '''checks scattering wave function for zero potential.'''
    
    # scattering potential
    v = np.zeros(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical scattering wave function
    y_num = scatter1d.wavefunction(e, v, dx)
    
    
    # exact analytical wave function
    k = np.sqrt(e) # wave vector
    y_ex = np.exp(-1J*k*x)
    
    
    # compare numerical and analytical solutions
    assert np.allclose(y_num, y_ex)




@pytest.mark.parametrize(('e', 'l', 'n'), [
    (0.1, 1.0, 10000), 
    (0.2, 1.0, 10000), 
    (0.3, 1.0, 10000), 
    (0.4, 1.0, 10000), 
    (0.5, 1.0, 10000), 
    (0.6, 1.0, 10000), 
    (0.7, 1.0, 10000), 
    (0.8, 1.0, 10000), 
    (0.9, 1.0, 10000), 
    (1.0, 1.0, 10000), 
    (1.1, 1.0, 10000), 
    (1.2, 1.0, 10000), 
    (1.3, 1.0, 10000), 
    (1.4, 1.0, 10000), 
    (1.5, 1.0, 10000), 
    (1.6, 1.0, 10000), 
])
def test_amplitudes_rectangular_potential(e, l, n):
    '''checks reflection and transmission amplitudes for rectangular potential.'''
    
    # rectangular potential barrier
    v = np.ones(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical reflection and transmission amplitudes
    r_num, t_num = scatter1d.amplitudes(e, v, dx)
    
    # exact analytical reflection and transmission amplitudes
    r_ex, t_ex = rectangular_barrier(e, 1.0, l)
    
    
    # compare numerical and analytical solutions
    assert np.isclose(r_num, r_ex, atol=1e-4)
    assert np.isclose(t_num, t_ex, atol=1e-4)



def rectangular_barrier(e, v0, l):
    '''exact reflection and transmission amplitudes for rectangular potential.'''
    
    # exact solution for right incident particle
    
    if np.isclose(e, v0):
        # limiting case
        k0 = np.sqrt(v0)
        
        denominator = 2J + k0*l
        
        t = 2J * np.exp(-1J*k0*l) / denominator
        r = k0*l * np.exp(-2J*k0*l) / denominator
        
    else:
        # general case
        k0 = np.sqrt(e)
        k1 = np.sqrt(complex(e-v0))
        
        denominator = (k0+k1)**2 * np.exp(-1J*k1*l) - (k0-k1)**2 * np.exp(1J*k1*l)
        
        t = 4*k0*k1 * np.exp(-1J*k0*l) / denominator
        r = (k1*k1-k0*k0) * np.exp(-2J*k0*l) * \
            (np.exp(1J*k1*l) - np.exp(-1J*k1*l)) / denominator
    
    return r, t
