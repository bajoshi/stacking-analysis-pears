"""
These MCMC codes are based on those from Mike Line's astrostatistics 
class. I'm using this to learn more about MCMC and fitting spectral data.
"""

import numpy as np
from scipy.stats import multivariate_normal
from emcee.autocorr import integrated_time
import corner

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def targ_dist(C, mean1, mean2, cov):

    rv1 = multivariate_normal(mean1, cov)
    rv2 = multivariate_normal(mean2, cov)

    g1 = rv1.pdf(C)
    g2 = rv2.pdf(C)

    return g1 + g2

def main():

    # Set up multivariate normal mean and covariance matrix
    cov = [[1.0, 0.0], [0.0, 1.0]]
    mean1 = [-2.0, -2.0]
    mean2 = [2.0, 2.0]

    # --------- Code block to check basic working of multivariate normal dist
    # Set up coordinates
    dx = 0.01
    x = np.arange(-10, 10, dx)
    y = np.arange(-10, 10, dx)

    X, Y = np.meshgrid(x, y)
    C = np.dstack((X, Y))

    # Get target distribution and plot to check
    Z = targ_dist(C, mean1, mean2, cov)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, Z)
    plt.show()
    """
    # ---------

    # ---------------------------- Metropolis Hastings ---------------------------- #
    N = 50000  # number of "timesteps"
    r = np.array([0, 0])  # defining parameter vector (X in the notes)

    cx, cy = np.meshgrid(r[0], r[1])
    c0 = np.dstack((cx, cy))

    p = targ_dist(c0, mean1, mean2, cov)  # evaluating the probability at the initial guess
    jump_size = 2.0 # size of multivariate normal proposal distribution
    
    print("Jump size [in units of parameter space coords??]:", jump_size)
    print("Initial guess probability:", p)
    
    samples = []  #creating array to hold parameter vector with time
    accept = 0.

    for i in range(N):#beginning the iteratitive loop

        print("Evaluating:", i, end='\r')

        rn = r + jump_size * np.random.normal(size=2)  #creating proposal vecotr ("Y" or X_t+1)

        cnx, cny = np.meshgrid(rn[0], rn[1])
        cn = np.dstack((cnx, cny))

        pn = targ_dist(cn, mean1, mean2, cov)  #evaluating probability of proposal vector

        # print("\nProposed movement vector:", rn)
        # print("Probability at new location:", pn)
        
        if pn >= p:   #always keep it if probability got higher
            # print("Probability increased. Will keep this point.")
            p = pn
            r = rn
            accept+=1
        
        else:  #only keep it based on acceptance probability
            # print("Probability decreased. Will decide to keep or not based on random.")
            u = np.random.rand()  #random number between 0 and 1
            if u < pn/p:  #only if proposal prob / previous prob is greater than u, then keep new proposed step
                p = pn
                r = rn
                accept+=1
                # print("Point kept.")
    
        samples.append(r)  #update

    # ---------------------------- END Metropolis Hastings ---------------------------- #

    #plotting
    samples = np.array(samples)
    print("Samples:", samples)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1, color='k')

    '''Plot target'''
    CS = ax.contour(X, Y, Z, 10, cmap=cm.inferno)
    ax.clabel(CS, inline=1, fontsize=8)

    # Trace plots and burn in?
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(samples[:,0], label='X', color='tab:blue')
    ax1.plot(samples[:,1], label='Y', color='tab:red')
    ax1.legend(loc=0)
    plt.show()

    # Plot basic corner plot
    corner.corner(samples)
    plt.show()
    
    # ----- autocorrelation
    cor_len1 = integrated_time(samples[:,0])
    cor_len2 = integrated_time(samples[:,1])

    print('Correlation Length:', cor_len1, cor_len2)
    print('Independent Samples:', N/cor_len1, N/cor_len2)
    
    # acceptance (want between 25 and 50%)
    # This is affected by jump size
    print('Acceptance Rate:', accept/N)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)