"""
These MCMC codes are based on those from Mike Line's astrostatistics 
class. I'm using this to learn more about MCMC and fitting spectral data.
"""

import numpy as np
import scipy.optimize as optimize
import scipy.linalg as linalg
import emcee
import corner

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

home = os.getenv('HOME')

def loglike(theta, x, data, err):
    
    v0, logT0 = theta

    y = model(x, v0, 10**logT0)
    lnLike = -0.5 * np.sum((y-data)**2/err**2)
    
    return lnLike

def logprior(theta):

    v0, logT0 = theta
    
    if ( 1. <= v0 <=70.  and  -1.0 <= logT0 <=2   ):
        return 0.0
    
    return -np.inf

def logpost(theta, x, data, err):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, x, data, err)
    
    return lp + lnL


def model(v, v0, T0):

    sig = 2.
    f = np.exp(-0.5*(v-v0)**2/sig**2)

    return T0 * f

def main():

    # ---- Load in data
    datadir = home + '/Documents/astro-statistics/mcmc_learning/'
    dat = np.genfromtxt(datadir + 'data.txt', dtype=None, names=['channel', 'flux'], skip_header=1, encoding='ascii')

    wav = dat['channel']
    flam = dat['flux']

    N = len(wav)  # number of data points
    
    err0 = 1.
    err = np.zeros(N) + err0

    # ---- Plot data if you want to check what it looks like
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, flam)
    plt.show()
    """

    # Below we will try three different fitting methods.
    # 1. Least squares using curve_fit from scipy.optimize
    # 2. Using our self-written MCMC with the MH algorithm
    # 3. Using the emcee module

    # ----------------------- 1. Using Scipy Optimize ----------------------- #
    # Call curve_fit with some initial guess
    theta0 = np.array([10,10])  # initial guess
    bestarr, covarr = optimize.curve_fit(model, wav, flam, theta0, err)
    print("Results from scipy curve fit:")
    print("Best-fit parameter values:", bestarr)

    # top left and bottom right are the errors^2 on the two parameters
    print("Covariance matrix for fitted parameters:", covarr)

    """
    #try a couple of different random intial guesses--boy, least squares sucks!
    for i in range(10):
        theta0 = np.array([np.random.uniform(1,70), np.random.uniform(0.1,100)])
        bestarr, covarr = optimize.curve_fit(model, wav, flam, theta0, err)
        print("\nIteration:", i, ";     best-fit parameter values:", bestarr)
        print("Covariance matrix for fitted parameters:\n", covarr)
    """

    """
    The above code block for using curve_fit from scipy.optimize is intended to 
    show how HIGHLY dependent the procedure is on the initial guess for the 
    parameters. It will give a different answer every time it is called with
    different initial guesses. For some initial guesses, it will fail altogether.
    """ 

    # ----------------------- 2. Using explicit MCMC with Metropolis-Hastings ----------------------- #
    #*******Metropolis Hastings********************************
    print("\nRunning explicit Metropolis-Hastings...")
    N = 500000   #number of "timesteps"

    r = np.array([10, 0])  # defining parameter vector (X in the notes)

    logp = logpost(r, wav, flam, err)  # evaluating the probability at the initial guess
    jump_size = 2.0 # size of multivariate normal proposal distribution
    
    print("Jump size [in units of parameter space coords??]:", jump_size)
    print("Initial guess log(probability):", logp)

    samples = []  #creating array to hold parameter vector with time
    accept = 0.

    for i in range(N): #beginning the iteratitive loop

        print("MH Iteration", i, end='\r')

        rn = r + jump_size * np.random.normal(size=2)  #creating proposal vecotr ("Y" or X_t+1)

        #print("Proposal parameter vector", rn)
        
        logpn = logpost(rn, wav, flam, err)  #evaluating probability of proposal vector
        #print("Proposed parameter vector log(probability):", logpn)
        dlogL = logpn - logp
        a = np.exp(dlogL)

        #print("Ratio of probabilities at proposed to current position:", a)

        if a >= 1:   #always keep it if probability got higher
            #print("Will accept point since probability increased.")
            logp = logpn
            r = rn
            accept+=1
        
        else:  #only keep it based on acceptance probability
            #print("Probability decreased. Will decide whether to keep point or not.")
            u = np.random.rand()  #random number between 0 and 1
            if u < a:  #only if proposal prob / previous prob is greater than u, then keep new proposed step
                logp = logpn
                r = rn
                accept+=1
                #print("Point kept.")

        samples.append(r)  #update
    print("Finished explicit Metropolis-Hastings.")
    # --------- End metropolis hastings

    # Plotting results from explicit MH
    samples = np.array(samples)

    # plot trace
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples[:,0], color='tab:red', label='v0')
    ax.plot(samples[:,1], color='tab:blue', label='log10(T0)')
    ax.legend(loc=0)
    plt.show()

    # using corner
    corner.corner(samples, bins=100, labels=[r'$v_0$',r'$log10(T_0)$'], \
        show_titles='True', plot_contours='True', truths=np.array([37., 0.]))
    plt.show()
    #  levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)), 

    print("Acceptance Rate:", accept/N)

    # ----------------------- 3. Using emcee ----------------------- #
    print("\nRunning emcee...")
    ndim, nwalkers = 2, 100  # setting up emcee params--number of params and number of walkers
    pos = [bestarr + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]  # generating "intial" ball of walkers about curvefit best fit
    pos = np.array(pos)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[wav, flam, err], threads=2)
    sampler.run_mcmc(pos, 1000)
    chains = sampler.chain
    print("Finished running emcee.")

    # plot trace
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for i in range(nwalkers): 
        ax1.plot(chains[i,:,0], 'tab:red', label='v0', alpha=0.1)
    for i in range(nwalkers): 
        ax1.plot(chains[i,:,1], 'tab:blue', label='log10(T0)', alpha=0.1)
    plt.show()

    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation
    burn_in = 400
    samples = sampler.chain[:, 400:, :].reshape((-1, ndim))

    # plot corner plot
    corner.corner(samples, bins=30, labels=[r'$v_0$',r'$log10(T_0)$'], levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)), \
        show_titles='True', plot_contours='True', truths=np.array([37., 0.]))
    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

