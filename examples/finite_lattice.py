# -*- coding: utf-8 -*-


import transport
import numpy as np

import matplotlib.pyplot as pp



'''calculate transmission and wave functions through finite lattice.

finite lattice potential
V(x) = V0 * sin(pi * x/d)^2 for 0 <= x <= n*d
V(x) = 0 else

quantities in normalized units
* position xi = x / d
* particle energy eps = e / [hbar^2 / (2*m*d^2)]
* peak potential v0 = V0 / [hbar^2 / (2*m*d^2)]
'''



def lattice_potential(xi, v0, n):
    '''lattice potential in normalized units.'''
    
    # condition for point inside lattice
    cond = np.logical_and(0 < xi, xi < n)
    
    # return potential
    return np.where(cond, v0 * np.sin(np.pi*xi)**2, 0)



def plot_basic(xi, v, eps, t):
    '''plot lattice potential and transmission.'''
    
    # create figure and axes
    fig, axs = pp.subplots(ncols=2, sharey=True, 
        gridspec_kw=dict(width_ratios=[3, 1]))
    
    
    # plot finite lattice potential
    axs[0].fill_between(xi, v, color='0.7')
    axs[0].plot(xi, v, color='k')
    
    
    # plot transmission probabilities
    axs[1].fill_betweenx(eps, np.abs(t)**2, color='0.7')
    axs[1].plot(np.abs(t)**2, eps, c='k')
    
    return fig, axs



def plot_wavefunction(fig, axs, xi, eps, psi, colors, func):
    '''plot wave functions.'''
    
    for i, c in enumerate(colors):
        
        # mark particle energy in axes
        axs[0].axhline(eps[i], ls='dashed', c=c)
        axs[1].axhline(eps[i], ls='dashed', c=c)
        
        # plot wave function with energy offset
        axs[0].plot(xi, eps[i] + func(psi[i]), c=c)
    
    return fig, axs




# colors for plotting different wave functions
colors = ['#d11141', '#00b159', '#00aedb', '#f37735']


# dimensionless potential height
v0 = 10.0

# dimensionless particle energies for transmission
eps_transmission = np.linspace(0.1, 25, 700)

# dimensionless particle energies for wave functions
eps_wavefunction = np.array([2.4, 7.05, 12, 19.2])


# dimensionless positions to discretize potential
xi, dxi = np.linspace(-3, 9, 500, retstep=True)



# calculate finite lattice potential
v = lattice_potential(xi, v0, n=6)


# calculate reflection and transmission amplitudes
r, t = transport.amplitudes(eps_transmission, v, dxi, left=True)


# array stores wave functions for each particle energy
psi = np.empty((eps_wavefunction.size, xi.size), dtype=complex)

for i, eps in enumerate(eps_wavefunction):
    
    # calculate scattering wave function
    psi[i] = transport.wavefunction(eps, v, dxi, left=True)



# function defines plotted quantity of wave functions
funcs = [np.real, lambda x: np.abs(x)**2]

# corresponding axis label for funcs
ylabels = [
    u'Re($\psi$) ($1/d^{1/2}$)\nenergy ($\hbar^2 / m d^2$)', 
    u'$|\psi|^2$ ($1/d$)\nenergy ($\hbar^2 / m d^2$)'
]


for func, ylabel in zip(funcs, ylabels):
    
    # plot lattice potential, transmission and wave functions
    fig, axs = plot_basic(xi, v, eps_transmission, t)
    plot_wavefunction(fig, axs, xi, eps_wavefunction, psi, colors, func)
    
    # set plotting limits
    axs[0].set_ylim(-0.4, 25.4)
    axs[1].set_xlim(-0.05, 1.05)
    
    # annotate plot axes
    axs[0].set_xlabel('position (d)')
    axs[1].set_xlabel('transmission')
    
    axs[0].set_ylabel(ylabel)


pp.show()
