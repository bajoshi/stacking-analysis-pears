"""
These MCMC codes are based on those from Mike Line's astrostatistics 
class. I'm using this to learn more about MCMC and fitting spectral data.

This particular code is used to see if I can fit a polynomial to one of
the grism spectra from the stacking project. 
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
    
    z, age, tau, av, lsf_sigma = theta

    y = model(x, z, age, tau, av, lsf_sigma)
    #print("Model func result:", y)

    # ------- Vertical scaling factor
    alpha = np.sum(data * y / err**2) / np.sum(y**2 / err**2)
    #print("Alpha:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.sum((y-data)**2/err**2)
    
    return lnLike

def logprior(theta):

    z, age, tau, av, lsf_sigma = theta
    
    # Make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = Planck15.age(z).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    if ( 0.01 <= z <= 6.0  and  0.01 <= age <= age_lim  and  0.01 <= tau <= 100.0  and  0.0 <= av <= 3.0  and  10.0 <= lsf_sigma <= 300.0  ):
        return 0.0
    
    return -np.inf

def logpost(theta, x, data, err):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, x, data, err)
    
    return lp + lnL

def model(x, z, age_gyr, tau_gyr, av, lsf_sigma):
    """
    This function will return the closest BC03 template 
    from a large grid of pre-generated templates.

    Expects to get the following arguments
    x: observed wavelength grid
    z: redshift to apply to template
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

    # ------ Apply redshift
    model_lam_z, model_flam_z = apply_redshift(model_lam_grid, model_llam, z)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((model_lam_z >= x[0] - lam_step) & (model_lam_z < x[0] + lam_step))[0]
    model_mod[0] = np.mean(model_lsfconv[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((model_lam_z >= x[j-1]) & (model_lam_z < x[j+1]))[0]
        model_mod[j] = np.mean(model_lsfconv[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((model_lam_z >= x[-1] - lam_step) & (model_lam_z < x[-1] + lam_step))[0]
    model_mod[-1] = np.mean(model_lsfconv[idx])

    return model_mod

def main():

    print("Starting at:", datetime.datetime.now())

    print("\n* * * *   [WARNING]: the downgraded model is offset by delta_lambda/2 where delta_lambda is the grism wavelength sampling.   * * * *\n")
    print("\n* * * *   [WARNING]: not interpolating to find matching models in parameter space.   * * * *\n")
    print("\n* * * *   [WARNING]: using two different cosmologies for dl and Universe age at a redshift.   * * * *\n")

    # ---- Load in data
    pears_id = 109151
    pears_field = 'GOODS-S'

    fname = pears_field + '_' + str(pears_id) + '_' + 'PAcomb.fits'

    spec_hdu = fits.open(datadir + fname)

    # Get data
    wav = spec_hdu[1].data
    flam = spec_hdu[2].data[0]
    ferr = spec_hdu[2].data[1]

    # Chop wavelength grid to 6000A -- 9500A
    arg_low = np.argmin(abs(wav - 5900))
    arg_high = np.argmin(abs(wav - 9700))

    wav  = wav[arg_low:arg_high+1]
    flam = flam[arg_low:arg_high+1]
    ferr = ferr[arg_low:arg_high+1]

    # ---- Plot data if you want to check what it looks like
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, flam)
    ax.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5)
    plt.show()
    """

    # ----------------------- Using numpy polyfitting ----------------------- #
    # -------- The best-fit will be used to set the initial position for the MCMC walkers.
    pfit = np.ma.polyfit(wav, flam, deg=3)
    np_polynomial = np.poly1d(pfit)

    bestarr = pfit

    print("Numpy polynomial fit:")
    print(np_polynomial)

    # Plotting fit and residuals
    """
    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(6,2)
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.7)

    ax1 = fig.add_subplot(gs[:4,:])
    ax2 = fig.add_subplot(gs[4:,:])

    ax1.plot(wav, flam, 'o-', markersize=3.0, color='turquoise', linewidth=1.5, label='PEARS obs data')
    ax1.plot(wav, np_polynomial(wav), color='crimson', zorder=1.0, label='NumPy polynomial fit')

    # Divide by polynomial fit
    flam = flam / np_polynomial(wav)
    ferr = ferr / np_polynomial(wav)

    # Plot "pure emission/absorption" spectrum
    ax2.axhline(y=1.0, ls='--', color='black', lw=1.5, zorder=1)
    ax2.plot(wav, flam, color='turquoise', linewidth=1.5, zorder=2)

    plt.show()
    """

    # ----------------------- Using explicit MCMC with Metropolis-Hastings ----------------------- #
    #*******Metropolis Hastings********************************
    mh_start = time.time()
    print("\nRunning explicit Metropolis-Hastings...")
    N = 50000   #number of "timesteps"

    # The parameter vector is (redshift, age, tau, av)
    # age in gyr and tau in gyr
    # last parameter is av not tauv
    r = np.array([0.1, 1.0, 1.0, 1.0, 10.0])  # initial position
    print("Initial parameter vector:", r)

    # Set jump sizes
    jump_size_z = 0.01  
    jump_size_age = 0.1  # in gyr
    jump_size_tau = 0.1  # in gyr
    jump_size_av = 0.2  # magnitudes
    jump_size_lsf = 5.0  # angstroms

    label_list = [r'$z$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$', r'$LSF [\AA]$']

    """
    logp = logpost(r, wav, flam, ferr)  # evaluating the probability at the initial guess
    
    print("Initial guess log(probability):", logp)

    samples = []  #creating array to hold parameter vector with time
    accept = 0.

    for i in range(N): #beginning the iteratitive loop

        print("MH Iteration", i, end='\r')

        rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))  #creating proposal vecotr ("Y" or X_t+1)
        rn1 = float(r[1] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[2] + jump_size_tau * np.random.normal(size=1))
        rn3 = float(r[3] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[4] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn0, rn1, rn2, rn3, rn4])

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
    ax.plot(samples[:,0], label='z')
    ax.plot(samples[:,1], label='Age [Gyr]')
    ax.plot(samples[:,2], label='Tau [Gyr]')
    ax.plot(samples[:,3], label=r'$A_v [mag]$')
    ax.plot(samples[:,4], label=r'$LSF [\AA]$')
    ax.legend(loc=0)

    # using corner
    corner.corner(samples, bins=30, labels=label_list, \
        show_titles='True', plot_contours='True')
    plt.show()

    print("Acceptance Rate:", accept/N)

    sys.exit(0)
    """

    # ----------------------- Using emcee ----------------------- #
    print("\nRunning emcee...")
    ndim, nwalkers = 5, 100  # setting up emcee params--number of params and number of walkers

    # generating "intial" ball of walkers about best fit from min chi2
    pos = np.zeros(shape=(nwalkers, ndim))

    for i in range(nwalkers):

        rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))
        rn1 = float(r[1] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[2] + jump_size_tau * np.random.normal(size=1))
        rn3 = float(r[3] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[4] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn0, rn1, rn2, rn3, rn4])

        pos[i] = rn

    from multiprocessing import Pool

    with Pool() as pool:
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[wav, flam, ferr], pool=pool)
        sampler.run_mcmc(pos, 5000, progress=True)

    print("Finished running emcee.")

    samples = sampler.get_chain()

    print("Samples shape:", samples.shape)

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.1)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    # Get autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
    except AutocorrError:
        print("Emcee AutocorrError occured.")
        print("The chain is shorter than 50 times the integrated autocorrelation time for 5 parameter(s).")
        print("Use this estimate with caution and run a longer chain!")
    print("Autocorrelation time (i.e., steps that walkers take in each dimension before they forget where they started):", tau)

    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    burn_in = int(3 * tau[0])
    print("Burn-in:", burn_in)

    thinning_steps = int(0.5 * tau[0])
    print("Thinning steps:", thinning_steps)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("Flat samples shape:", flat_samples.shape)

    # plot corner plot
    fig2 = corner.corner(flat_samples, plot_contours='True', labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14})

    plt.show()

    # Plot 100 random models from the parameter space
    inds = np.random.randint(len(flat_samples), size=100)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(wav, flam, color='k')
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=3)

    for ind in inds:
        sample = flat_samples[ind]
        m = model(wav, sample[0], sample[1], sample[2], sample[3], sample[4]) 
        ax3.plot(wav, m, color='tab:red', alpha=0.2, zorder=2)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


