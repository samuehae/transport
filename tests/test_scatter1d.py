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
def test_scatter_zero(e, l, n):
    '''checks reflection and transmission amplitudes for zero potential.'''
    
    # expected scattering amplitudes for vanishing potential
    # reflection amplitude r = 0
    # transmission amplitude t = 1
    
    # scattering potential
    v = np.zeros(n)
    
    # sampling points
    x, dx = np.linspace(0, l, n, retstep=True)
    
    # numerical reflection and transmission amplitudes
    r_num, t_num = scatter1d.scatter(e, v, dx)
    
    
    # compare numerical and analytical solutions
    assert np.isclose(r_num, 0)
    assert np.isclose(t_num, 1)
