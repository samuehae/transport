# -*- coding: utf-8 -*-


import transport
import numpy as np

import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages



'''transmission, reflection and loss probabilities through rectangular potential.

rectangular potential
V(x) = V0 for 0 <= x <= w
V(x) = 0 else

quantities in normalized units
* position xi = x / w
* particle energy eps = e / [hbar^2 / (2*m*w^2)]
* potential height v0 = V0 / [hbar^2 / (2*m*w^2)]
'''



# dimensionless potential height
v0_values = np.linspace(1, 1-1j, 21)

# dimensionless particle energies
eps = np.linspace(0.01, 5, 300)


# dimensionless positions to discretize potentials
n = 500 # number of sampling points
dxi = 1.0 / (n - 1) # separation between positions



# open pdf file
pdf_pages = PdfPages('rectangular_potential.pdf')


for v0 in v0_values:
    
    # gaussian scattering potentials
    v = np.full(n, v0)
    
    # calculate reflection and transmission amplitudes
    r, t = transport.amplitudes(eps, v, dxi, left=False)
    
    # transmission, reflection and loss probabilities
    T = np.abs(t)**2
    R = np.abs(r)**2
    L = 1 - T - R
    
    
    
    # create plot
    fig, ax = pp.subplots()
    
    # plot probabilities as areas
    ax.fill_between(eps, L + R, 1, color='#636bab', label='transmission')
    ax.fill_between(eps, L, L + R, color='#d6568f', label='reflection')
    ax.fill_between(eps, L, color='#f59a53', label='loss')
    
    # annotate plot axes
    ax.set_ylabel('probabilities')
    ax.set_xlabel(r'particle energy ($\hbar^2 / m w^2$)')
    
    # set plot limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 5)
    
    # display legend
    ax.legend(loc='upper right', facecolor='white', 
        framealpha=1.0, edgecolor='None')
    
    # annotate plot with potential height
    ax.set_title('$V_0$ = {:.2f} ($\hbar^2 / m w^2$)'.format(v0))
    
    # save and close figure
    pdf_pages.savefig(fig)
    pp.close(fig)



# close pdf file
pdf_pages.close()
