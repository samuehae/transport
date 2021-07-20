# -*- coding: utf-8 -*-


import transport
import numpy as np

import matplotlib.pyplot as pp



'''compare transmission through a gaussian and a parabolic barrier.

a) gaussian potential
V(x) = V0 * exp(-2*(x/w)^2)

b) parabolic barrier: approximates gaussian potential
V(x) = V0 * (1 - 2*(x/w)^2)

quantities in normalized units
* position xi = x / w
* particle energy eps = e / [hbar^2 / (2*m*w^2)]
* peak potential v0 = V0 / [hbar^2 / (2*m*w^2)]
'''




def gaussian_potential(xi, v0):
    '''gaussian potential in normalized units.'''
    
    return v0 * np.exp(-2*xi*xi)



def parabolic_transmission(e, v0):
    '''analytical transmission through parabolic potential.
    
    sources:
    * L. Glazman et al., Zh. Eksp. Teor. Fiz. Pis'ma Red. 48, 218 (1988)
    * T. Ihn, Semiconductor Nanostructures (Oxford University, 2010)
    '''
    
    arg = -np.pi / np.sqrt(2.0 * v0) * (e - v0)
    return 1.0 / (1.0 + np.exp(arg))




# colors for plotting different potential heights
colors = ['#96ceb4', '#ffcc5c', '#ff6f69']


# dimensionless potential height
v0_list = [2, 10, 20]


# dimensionless particle energies
eps = np.linspace(0.1, 30, 300)


# dimensionless positions to discretize potentials
xi, dxi = np.linspace(-5, 5, 500, retstep=True)



for c, v0 in zip(colors, v0_list):
    
    # gaussian scattering potentials
    v_gauss = gaussian_potential(xi, v0)
    
    # calculate reflection and transmission amplitudes
    r_gauss, t_gauss = transport.amplitudes(eps, v_gauss, dxi, left=False)
    
    
    # calculate transmission probability through parabolic approximation
    t_parab = parabolic_transmission(eps, v0)
    
    
    # plot transmission probabilities
    pp.plot(eps, np.abs(t_gauss)**2, ls='solid', c=c, label=str(v0))
    pp.plot(eps, np.abs(t_parab)**2, ls='dashed', c=c)



pp.ylabel('transmission probability')
pp.xlabel(r'particle energy ($\hbar^2 / m w^2$)')

pp.legend(frameon=False, title=r'$V_0$ ($\hbar^2 / m w^2$)')


pp.show()
