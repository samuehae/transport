# -*- coding: utf-8 -*-


import pytest
import numpy as np
from transport import scatter1d



@pytest.mark.parametrize('e', [0.1, 0.2, 1.0, 1.1])
@pytest.mark.parametrize('l', [10, ])
@pytest.mark.parametrize('n', [230, ])
@pytest.mark.parametrize('left', [False, True])

def test_amplitudes_zero_potential(e, l, n, left):
    '''checks reflection and transmission amplitudes for zero potential.'''
    
    # expected scattering amplitudes for vanishing potential
    # reflection amplitude r = 0
    # transmission amplitude t = 1
    
    # scattering potential
    v = np.zeros(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical reflection and transmission amplitudes
    r_num, t_num = scatter1d.amplitudes(e, v, dx, left)
    
    
    # compare numerical and analytical solutions
    assert np.isclose(r_num, 0)
    assert np.isclose(t_num, 1)



@pytest.mark.parametrize('e', [0.1, 0.2, 1.0, 1.1])
@pytest.mark.parametrize('l', [10, ])
@pytest.mark.parametrize('n', [230, ])
@pytest.mark.parametrize('left', [False, True])

def test_wavefunction_zero_potential(e, l, n, left):
    '''checks scattering wave function for zero potential.'''
    
    # scattering potential
    v = np.zeros(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical scattering wave function
    y_num = scatter1d.wavefunction(e, v, dx, left)
    
    # exact analytical wave function
    k = np.sqrt(e) # wave vector
    if left:
        y_ex = np.exp(1J*k*x)
    else:
        y_ex = np.exp(-1J*k*x)
    
    
    # compare numerical and analytical solutions
    assert np.allclose(y_num, y_ex)




@pytest.mark.parametrize('e', np.arange(0.1, 1.7, 0.1))
@pytest.mark.parametrize('v0', [1.0, -1j, 1.0-0.5j])
@pytest.mark.parametrize('l', [1.0, ])
@pytest.mark.parametrize('n', [10000, ])
@pytest.mark.parametrize('left', [False, True])

def test_amplitudes_rectangular_potential(e, v0, l, n, left):
    '''checks reflection and transmission amplitudes for rectangular potential.'''
    
    # rectangular potential barrier
    v = np.full(n, v0)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical reflection and transmission amplitudes
    r_num, t_num = scatter1d.amplitudes(e, v, dx, left)
    
    # exact analytical reflection and transmission amplitudes
    r_ex, t_ex, _ = rectangular_barrier(e, v0, l, x, left)
    
    # compare numerical and analytical solutions
    assert np.isclose(r_num, r_ex, atol=1e-4)
    assert np.isclose(t_num, t_ex, atol=1e-4)



@pytest.mark.parametrize('e', np.arange(0.1, 1.7, 0.1))
@pytest.mark.parametrize('v0', [1.0, -1j, 1.0-0.5j])
@pytest.mark.parametrize('l', [1.0, ])
@pytest.mark.parametrize('n', [10000, ])
@pytest.mark.parametrize('left', [False, True])

def test_wavefunction_rectangular_potential(e, v0, l, n, left):
    '''checks wave function for rectangular potential.'''
    
    # rectangular potential barrier
    v = np.full(n, v0)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical scattering wave function
    y_num = scatter1d.wavefunction(e, v, dx, left)
    
    # exact analytical wave function
    y_ex = rectangular_barrier(e, v0, l, x, left)[2]
    
    
    # compare numerical and analytical solutions
    assert np.allclose(y_num, y_ex, atol=1e-4)



def rectangular_barrier(e, v0, l, x, left):
    '''exact wave function and amplitudes for rectangular potential.'''
    
    
    if np.isclose(e, v0):
        # limiting case
        k0 = np.sqrt(v0)
        
        denominator = 2J + k0*l
        
        
        # transmission and reflection amplitudes (right incident)
        t = 2J * np.exp(-1J*k0*l) / denominator
        r = k0*l * np.exp(-2J*k0*l) / denominator
        
        if left:
            # correct reflection amplitude
            r *= np.exp(2J*k0*l)
            
            # wave function inside scattering region
            y = (2J - 2*k0*(x-l)) / denominator
            
        else:
            # wave function inside scattering region
            y = (2J + 2*k0*x) * np.exp(-1J*k0*l) / denominator
        
    else:
        # general case
        k0 = np.sqrt(e)
        k1 = np.sqrt(complex(e-v0))
        
        denominator = (k0+k1)**2 * np.exp(-1J*k1*l) - (k0-k1)**2 * np.exp(1J*k1*l)
        
        
        # transmission and reflection amplitudes (right incident)
        t = 4*k0*k1 * np.exp(-1J*k0*l) / denominator
        r = (k1*k1-k0*k0) * np.exp(-2J*k0*l) * \
            (np.exp(1J*k1*l) - np.exp(-1J*k1*l)) / denominator
        
        if left:
            # correct reflection amplitude
            r *= np.exp(2J*k0*l)
            
            # coefficients to calculate wave function inside scattering region
            beta0 = 2*k0*(k0+k1) * np.exp(-1J*k1*l) / denominator
            beta1 = -2*k0*(k0-k1) * np.exp(1J*k1*l) / denominator
            
        else:
            # coefficients to calculate wave function inside scattering region
            beta0 = -2*k0*(k0-k1) * np.exp(-1J*k0*l) / denominator
            beta1 = 2*k0*(k0+k1) * np.exp(-1J*k0*l) / denominator
        
        
        # wave function inside scattering region
        y = beta0 * np.exp(1J*k1*x) + beta1 * np.exp(-1J*k1*x)
    
    return r, t, y
