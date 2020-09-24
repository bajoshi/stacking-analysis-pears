"""
This code will fit, using Markov Chain Monte Carlo, the
supplied continuum divided stack of spectra. 
"""

import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from astropy.cosmology import Planck15

import os
import sys
from functools import reduce
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_figs_dir = datadir = home + '/Documents/pears_figs_data/'
datadir = home + '/Documents/pears_figs_data/data_spectra_only/'
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
stacking_analysis_dir = home + '/Documents/GitHub/stacking-analysis-pears/'
stacking_figures_dir = home + "/Documents/stacking_figures/"

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy', mmap_mode='r')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy', mmap_mode='r')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy', mmap_mode='r')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy', mmap_mode='r')

"""
Array ranges are:
1. Age: 7.02 to 10.114 (this is log of the age in years)
2. Metals: 0.0001 to 0.05 (absolute fraction of metals. All CSP models although are fixed at solar = 0.02)
3. Tau: 0.01 to 63.095 (this is in Gyr. SSP models get -99.0)
4. TauV: 0.0 to 2.8 (Visual dust extinction in magnitudes. SSP models get -99.0)
"""

def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap):

    """
    print("\nFinding closest model to --")
    print("Age [Gyr]:", 10**age / 1e9)
    print("Tau [Gyr]:", tau)
    print("Tau_v:", tauv)
    print("Metallicity [abs. frac.]:", metallicity)
    """

    # First find closest values and then indices corresponding to them
    # It has to be done this way because you typically wont find an exact match
    closest_age_idx = np.argmin(abs(log_age_arr - age))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(abs(tauv_arr - tauv))

    # Now get indices
    age_idx = np.where(log_age_arr == log_age_arr[closest_age_idx])[0]
    tau_idx = np.where(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])[0]
    tauv_idx = np.where(tauv_arr   ==    tauv_arr[closest_tauv_idx])[0]
    metal_idx = np.where(metal_arr == metallicity)[0]

    model_idx = int(reduce(np.intersect1d, (age_idx, tau_idx, tauv_idx, metal_idx)))

    model_llam = model_comp_spec_withlines_mmap[model_idx]

    chosen_age = 10**log_age_arr[model_idx] / 1e9
    chosen_tau = tau_gyr_arr[model_idx]
    chosen_av = 1.086 * tauv_arr[model_idx]
    chosen_metallicity = metal_arr[model_idx]

    """
    print("\nChosen model index:", model_idx)
    print("Chosen model parameters -- ")
    print("Age [Gyr]:", chosen_age)
    print("Tau [Gyr]:", chosen_tau)
    print("A_v:", chosen_av)
    print("Metallicity [abs. frac.]:", chosen_metallicity)
    """

    return model_llam

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = cosmo.luminosity_distance(redshift)  # returns dl in Mpc
    dl = dl * 3.09e24  # convert to cm

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def loglike(theta, x, data, err):
    
    age, tau, av, lsf_sigma = theta

    y = model(x, age, tau, av, lsf_sigma)
    #print("Model func result:", y)

    # ------- Vertical scaling factor
    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.nansum((y-data)**2/err**2)
    
    return lnLike

def logprior(theta):

    age, tau, av, lsf_sigma = theta
    
    # Make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    #age_at_z = Planck15.age(z).value  # in Gyr
    #age_lim = age_at_z - 0.1  # in Gyr

    if ( 0.01 <= age <= 12.0  and  0.01 <= tau <= 100.0  and  0.0 <= av <= 3.0  and  10.0 <= lsf_sigma <= 300.0  ):
        return 0.0
    
    return -np.inf

def logpost(theta, x, data, err):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, x, data, err)
    
    return lp + lnL

def model(x, age_gyr, tau_gyr, av, lsf_sigma):
    """
    This function will return the closest BC03 template 
    from a large grid of pre-generated templates.

    Expects to get the following arguments
    x: observed wavelength grid
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    av: visual dust extinction
    lsf_sigma: in angstroms
    """

    current_age = np.log10(age_gyr * 1e9)  # because the saved age parameter is the log(age[yr])
    current_tau = tau_gyr  # because the saved tau is in Gyr
    tauv = av / 1.086
    current_tauv = tauv
    current_metallicity = 0.02  # Force it to only choose from the solar metallicity CSP models

    model_llam = get_template(current_age, current_tau, current_tauv, current_metallicity, \
        log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, model_lam_grid, model_grid)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_llam, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((model_lam_grid >= x[0] - lam_step) & (model_lam_grid < x[0] + lam_step))[0]
    model_mod[0] = np.mean(model_lsfconv[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((model_lam_grid >= x[j-1]) & (model_lam_grid < x[j+1]))[0]
        model_mod[j] = np.mean(model_lsfconv[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((model_lam_grid >= x[-1] - lam_step) & (model_lam_grid < x[-1] + lam_step))[0]
    model_mod[-1] = np.mean(model_lsfconv[idx])

    # ----------------------- Using numpy polyfitting ----------------------- #
    # -------- The best-fit will be used to set the initial position for the MCMC walkers.
    pfit = np.ma.polyfit(x, model_mod, deg=3)
    np_polynomial = np.poly1d(pfit)

    model_div_polyfit = model_mod / np_polynomial(x)

    return model_div_polyfit

def main():

    print("Starting at:", datetime.datetime.now())

    print("\n* * * *   [WARNING]: the downgraded model is offset by delta_lambda/2 where delta_lambda is the grism wavelength sampling.   * * * *\n")
    print("\n* * * *   [WARNING]: not interpolating to find matching models in parameter space.   * * * *\n")
    print("\n* * * *   [WARNING]: using two different cosmologies for dl and Universe age at a redshift.   * * * *\n")

    # ---- Load in data
    # Define redshift range 
    z_low = 0.16
    z_high = 0.96

    stack = np.genfromtxt(stacking_analysis_dir + 'massive_stack_pears_' + str(z_low) + 'z' + str(z_high) + '.txt', \
                          dtype=None, names=['lam', 'flam', 'flam_err'], encoding='ascii')

    wav = stack['lam']
    flam = stack['flam']
    ferr = stack['flam_err']

    # ----------------------- Using explicit MCMC with Metropolis-Hastings ----------------------- #
    #*******Metropolis Hastings********************************
    mh_start = time.time()
    print("\nRunning explicit Metropolis-Hastings...")
    N = 10000   #number of "timesteps"

    # The parameter vector is (redshift, age, tau, av)
    # age in gyr and tau in gyr
    # last parameter is av not tauv
    r = np.array([1.0, 1.0, 0.5, 10.0])  # initial position
    print("Initial parameter vector:", r)

    # Set jump sizes
    jump_size_age = 0.1  # in gyr
    jump_size_tau = 0.1  # in gyr
    jump_size_av = 0.2  # magnitudes
    jump_size_lsf = 5.0  # angstroms

    label_list = [r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$', r'$LSF [\AA]$']

    """
    logp = logpost(r, wav, flam, ferr)  # evaluating the probability at the initial guess
    
    print("Initial guess log(probability):", logp)

    samples = []  #creating array to hold parameter vector with time
    accept = 0.

    for i in range(N): #beginning the iteratitive loop

        print("MH Iteration", i, end='\r')

        rn1 = float(r[0] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[1] + jump_size_tau * np.random.normal(size=1))
        rn3 = float(r[2] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[3] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn1, rn2, rn3, rn4])

        #print("Proposed parameter vector", rn)
        
        logpn = logpost(rn, wav, flam, ferr)  #evaluating probability of proposal vector
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

    mh_time = time.time() - mh_start
    mh_min, mh_sec = divmod(mh_time, 60.0)
    mh_hr, mh_min = divmod(mh_min, 60.0)
    print("Time taken for explicit Metropolis-Hastings:", \
        "{:.2f}".format(mh_hr), "hours", "{:.2f}".format(mh_min), "mins", "{:.2f}".format(mh_sec), "seconds.")

    # Plotting results from explicit MH
    samples = np.array(samples)

    # plot trace
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples[:,0], label='Age [Gyr]')
    ax.plot(samples[:,1], label='Tau [Gyr]')
    ax.plot(samples[:,2], label=r'$A_v [mag]$')
    ax.plot(samples[:,3], label=r'$LSF [\AA]$')
    ax.legend(loc=0)

    # using corner
    corner.corner(samples, bins=30, labels=label_list, \
        show_titles='True', plot_contours='True')
    plt.show()

    print("Acceptance Rate:", accept/N)
    """

    # ----------------------- Using emcee ----------------------- #
    print("\nRunning emcee...")
    ndim, nwalkers = 4, 100  # setting up emcee params--number of params and number of walkers

    # generating "intial" ball of walkers about best fit from min chi2
    pos = np.zeros(shape=(nwalkers, ndim))

    for i in range(nwalkers):

        rn1 = float(r[0] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[1] + jump_size_tau * np.random.normal(size=1))
        rn3 = float(r[2] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[3] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn1, rn2, rn3, rn4])

        pos[i] = rn


    from multiprocessing import Pool
    with Pool() as pool:
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[wav, flam, ferr], pool=pool)
        sampler.run_mcmc(pos, 2000, progress=True)

    chains = sampler.chain
    print("Finished running emcee.")

    # plot trace
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for i in range(nwalkers):
        for j in range(ndim):
            ax1.plot(chains[i,:,j], label=label_list[j], alpha=0.1)

    fig1.savefig(stacking_figures_dir + 'mcmc_stackfit_trace.pdf', dpi=300, bbox_inches='tight')

    ax1.set_yscale('log')

    plt.clf()
    plt.cla()
    plt.close()

    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    burn_in = 400
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

    # plot corner plot
    fig = corner.corner(samples, bins=30, plot_contours='True', labels=label_list, label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14})
    fig.savefig(stacking_figures_dir + 'mcmc_stackfit_corner.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
