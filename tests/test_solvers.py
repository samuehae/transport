# -*- coding: utf-8 -*-


import pytest
import numpy as np
import transport.solvers as sol



@pytest.mark.parametrize(('a', 'b', 'q', 'fex'), [
    (0, 1, np.ones(11), np.sin), 
    (0, 1, np.ones(101), np.sin),
    (0, 1, np.ones(11), np.cos), 
    (0, 1, np.ones(101), np.cos),
    (0, 1, np.zeros(11), lambda x: x-1), 
    (-1, 0, np.zeros(11), lambda x: x-1),
    (1, 10, 1/np.linspace(1, 10, 501)**2, lambda x: np.sqrt(x)*np.cos(np.sqrt(3)/2.0*np.log(x)))
])
def test_numerov(a, b, q, fex):
    '''checks numerical solution of y''(x) + q(x)*y(x) = 0.'''
    
    # number of sampling points
    n = len(q)
    
    # sampling points
    x, dx = np.linspace(a, b, n, retstep=True)
    
    # numerical solution
    y_num = sol.numerov(q, fex(x[0]), fex(x[1]), dx, full=False)
    
    # exact analytical solution
    y_ex = fex(x[-2:])
    
    # compare numerical and exact solutions
    assert np.allclose(y_ex, y_num)
