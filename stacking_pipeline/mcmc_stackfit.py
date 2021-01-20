"""
This code will fit, using Markov Chain Monte Carlo, the
supplied continuum divided stack of spectra. 
"""

import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from scipy.interpolate import griddata
from scipy.interpolate import splev, splrep

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import os
import sys
from functools import reduce
import time
import datetime
import socket
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# assign directories and custom imports
home = os.getenv('HOME')
pears_figs_dir = datadir = home + '/Documents/pears_figs_data/'
datadir = home + '/Documents/pears_figs_data/data_spectra_only/'
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
stacking_analysis_dir = home + '/Documents/GitHub/stacking-analysis-pears/'
stacking_figures_dir = home + '/Documents/stacking_figures/'

emcee_diagnostics_dir = home + '/Documents/emcee_runs/emcee_diagnostics_stacking/'

sys.path.append(stacking_utils)
from dust_utils import get_dust_atten_model
import grid_coadd as gd

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

start = time.time()

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'

assert os.path.isdir(modeldir)

model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy", mmap_mode='r')
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy", mmap_mode='r')

all_m62_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m62_chab.npy', mmap_mode='r')
    all_m62_models.append(a)
    del a

# load models with large tau separately
all_m62_models.append(np.load(modeldir + 'bc03_all_tau20p000_m62_chab.npy', mmap_mode='r'))

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def loglike(theta, x, data, err):
    
    age, logtau, av, lsf_sigma, zscatter = theta

    y = model(x, age, logtau, av, lsf_sigma, zscatter)

    # ------- Clip all arrays to where the stack is believable
    # then get the log likelihood
    x0 = np.where( (x >= 3800) & (x <= 6300) )[0]

    y = y[x0]
    data = data[x0]
    err = err[x0]
    x = x[x0]

    lnLike = -0.5 * np.nansum((y-data)**2/err**2)

    #print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))
    #print("log likelihood:", lnLike)
    
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$L_\lambda\, [\mathrm{continuum\ divided}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    ax.set_xscale('log')
    ax.minorticks_on()
    plt.show()
    #sys.exit(0)
    """

    return lnLike

def logprior(theta):

    age, logtau, av, lsf_sigma, zscatter = theta

    if ( 0.01 <= age <= 13.0  and  -3.0 <= logtau <= 2.0  and  0.0 <= av <= 5.0 \
        and  70.0 <= lsf_sigma <= 140.0  and 0.00 <= zscatter <= 0.08):
        return 0.0
    
    return -np.inf

def logpost(theta, x, data, err):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, x, data, err)
    
    return lp + lnL

def model(x, age, logtau, av, lsf_sigma, zscatter):
    """
    This function will return the closest BC03 template 
    from a large grid of pre-generated templates.

    Expects to get the following arguments
    x: observed wavelength grid
    age: age of SED in Gyr
    logtau: log of exponential SFH timescale in Gyr
    av: visual dust extinction
    """

    tau = 10**logtau  # logtau is log of tau in gyr

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = tau_int_idx * len(model_ages)  +  age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

        #print("Tau int and age index:", tau_int_idx, age_idx)
        #print("Tau and age from index:", models_taurange_idx+tau_int_idx/1e3, model_ages[age_idx]/1e9)
        #print("Model tau range index:", models_taurange_idx)

    elif tau >= 20.0:
        
        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

        #print("logtau and age index:", logtau_idx, age_idx)
        #print("Tau and age from index:", 10**(logtau_arr[logtau_idx]), model_ages[age_idx]/1e9)

    #print("Model index:", model_idx)

    model_llam = models_arr[model_idx]

    # ------ Apply dust extinction
    model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_dusty_llam, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = griddata(points=model_lam, values=model_lsfconv, xi=x)

    """
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
    """

    # ----------------------- Using scipy spline fitting ----------------------- #
    model_err = np.zeros(len(x))
    model_cont_norm, model_err_cont_norm = divcont(x, model_mod, model_err)

    # Shift it to force stack value ~1.0 at ~4600A
    shift_idx = np.where((x >= 4600) & (x <= 4700))[0]
    scaling_fac = np.mean(model_cont_norm[shift_idx])
    model_cont_norm /= scaling_fac

    # ----------------------- Restack the same model using scatter in redshift ----------------------- #
    model_stack = gen_model_stack(x, model_cont_norm, zscatter)

    return model_stack

def gen_model_stack(w, m, zs):

    nstack = 500

    # Generate a random array of redshifts
    zs_arr = np.random.normal(loc=0.0, scale=zs, size=nstack)

    # Now stack the same model at all the redshifts given
    # Code same as the one in grid_coadd. make sure wavelength step and grid is the same

    lam_step = 25.0
    # Set the ends of the lambda grid
    # This is dependent on the redshift range being considered
    lam_grid_low = 3000
    lam_grid_high = 7600

    lam_grid = np.arange(lam_grid_low, lam_grid_high, lam_step)

    # Set up arrays to save stacks
    stack_ll = np.zeros(len(lam_grid))
    stack_le = np.zeros(len(lam_grid))
    # also convert to lists
    stack_ll = stack_ll.tolist()
    stack_le = stack_le.tolist()

    stack_numpoints = np.zeros(len(lam_grid))
    stack_numgalaxies = np.zeros(len(lam_grid))

    for i in range(len(zs_arr)):

        # This step should only be done on the first iteration within a grid cell
        # This converts every element (which are all 0 to begin with) 
        # in the flux and flux error arrays to an empty list
        # This is done so that function add_spec() can now append to every element
        if i == 0:
            for x in range(len(lam_grid)):
                stack_ll[x] = []
                stack_le[x] = []

        # Now scatter the model in redshift 
        # This only involves a shift in wavelength because 
        # it is the same model being stacked with itself
        current_w = w * (1 + zs_arr[i])

        # Add to stack
        stack_ll, stack_le, stack_numpoints, stack_numgalaxies = \
        gd.add_spec(current_w, m, np.zeros(len(m)), stack_ll, stack_le, \
            stack_numpoints, stack_numgalaxies, lam_grid, lam_step)

    # Now finalize the stack by computing medians
    stack_ll, stack_le = gd.take_median(stack_ll, stack_le, lam_grid)

    # convert to numpy array and return
    stack_ll = np.asarray(stack_ll)

    # make sure the returned stack has the same wavelength range as the model
    #idx = np.where((w >= 4000) & (w <= 6000))[0]
    #stack_ll_clip = stack_ll[idx]
    #lam_grid_clip = lam_grid[idx]
    stack_ll = griddata(points=lam_grid, values=stack_ll, xi=w)

    return stack_ll

def divcont(wav, flux, ferr, showplot=False):

    # Normalize flux levels to approx 1.0
    flux_norm = flux / np.mean(flux)
    ferr_norm = ferr / np.mean(flux)

    # Mask lines
    #mask_indices = get_mask_indices(wav, zprior)

    # Make sure masking indices are consistent with array to be masked
    #remove_mask_idx = np.where(mask_indices >= len(wav))[0]
    #mask_indices = np.delete(arr=mask_indices, obj=remove_mask_idx)

    #weights = np.ones(len(wav))
    #weights[mask_indices] = 0

    # SciPy smoothing spline fit
    spl = splrep(x=wav, y=flux_norm, k=3, s=0.1)
    wav_plt = np.arange(wav[0], wav[-1], 1.0)
    spl_eval = splev(wav_plt, spl)

    # Divide the given flux by the smooth spline fit and return
    cont_div_flux = flux_norm / splev(wav, spl)
    cont_div_err  = ferr_norm / splev(wav, spl)

    # Test figure showing fits
    if showplot:
        fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(5,1)
        gs.update(left=0.06, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.5)

        ax1 = fig.add_subplot(gs[:3,:])
        ax2 = fig.add_subplot(gs[3:,:])

        ax1.set_ylabel(r'$\mathrm{Flux\ [normalized]}$', fontsize=15)
        ax2.set_ylabel(r'$\mathrm{Continuum\ divided\ flux}$', fontsize=15)
        ax2.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=15)

        ax1.plot(wav, flux_norm, color='k')
        ax1.fill_between(wav, flux_norm - ferr_norm, flux_norm + ferr_norm, color='gray', alpha=0.5)
        ax1.plot(wav_plt, spl_eval, color='crimson', lw=3.0, label='SciPy smooth spline fit')

        ax2.plot(wav, cont_div_flux, color='teal', lw=2.0, label='Continuum divided flux')
        ax2.axhline(y=1.0, ls='--', color='k', lw=1.8)

        # Tick label sizes
        ax1.tick_params(which='both', labelsize=14)
        ax2.tick_params(which='both', labelsize=14)

        plt.show()

    return cont_div_flux, cont_div_err

def main():

    print("Starting at:", datetime.datetime.now())

    print(f"{bcolors.WARNING}")
    print("\n* * * *   [WARNING]: the downgraded model is offset by delta_lambda/2 where delta_lambda is the grism wavelength sampling.   * * * *\n")
    print(f"{bcolors.ENDC}")

    # ---- Load in data
    # Define redshift range 
    z_low = 0.16
    z_high = 0.96

    # Define mass range
    ms_lim_low = 10.5
    ms_lim_high = 12.0

    stack_filename = stacking_analysis_dir + 'massive_stack_pears_' \
    + str(ms_lim_low).replace('.','p') + '_Ms_' + str(ms_lim_high).replace('.','p') \
    + '_' + str(z_low).replace('.','p') + '_z_' + str(z_high).replace('.','p') + '.txt'

    stack = np.genfromtxt(stack_filename, dtype=None, names=['lam', 'flam', 'flam_err'], encoding='ascii')

    wav = stack['lam']
    flam = stack['flam']
    ferr = stack['flam_err']

    #x0 = np.where( (wav >= 4000) & (wav <= 6000) )[0]
    #wav = wav[x0]
    #flam = flam[x0]
    #ferr = ferr[x0]

    # ----------------------- Using explicit MCMC with Metropolis-Hastings ----------------------- #
    #*******Metropolis Hastings********************************
    #mh_start = time.time()
    #print("\nRunning explicit Metropolis-Hastings...")
    #N = 10000   #number of "timesteps"

    # The parameter vector is (redshift, age, tau, av)
    # age in gyr and tau in gyr
    # last parameter is av not tauv
    r = np.array([4.0, 0.1, 0.5, 95.0, 0.02])  # initial position
    print("Initial parameter vector:", r)

    # Set jump sizes
    jump_size_age = 0.5  # in gyr
    jump_size_logtau = 0.01  # in gyr
    jump_size_av = 0.2  # magnitudes
    jump_size_lsf = 5.0  # angstroms
    jump_size_zscatter = 0.002

    label_list = [r'$\mathrm{Age\ [Gyr]}$', r'$\mathrm{log(\tau\ [Gyr])}$', r'$\mathrm{A_V\ [mag]}$', \
    r'$\mathrm{LSF\ [\AA]}$', r'$\left< \frac{\Delta z}{1+z} \right>$']

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
    ndim, nwalkers = 5, 200  # setting up emcee params--number of params and number of walkers

    # generating "intial" ball of walkers about best fit from min chi2
    pos = np.zeros(shape=(nwalkers, ndim))

    for i in range(nwalkers):

        rn1 = float(r[0] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[1] + jump_size_logtau * np.random.normal(size=1))
        rn3 = float(r[2] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[3] + jump_size_lsf * np.random.normal(size=1))
        rn5 = float(r[4] + jump_size_zscatter * np.random.normal(size=1))

        rn = np.array([rn1, rn2, rn3, rn4, rn5])

        pos[i] = rn
    
    print("logpost at starting position:", logpost(r, wav, flam, ferr))

    # ----------- Set up the HDF5 file to incrementally save progress to
    emcee_savefile = emcee_diagnostics_dir + 'massive_stack_pears_' + str(z_low) + 'z' + str(z_high) + '_emcee_sampler.h5'
    backend = emcee.backends.HDFBackend(emcee_savefile)
    backend.reset(nwalkers, ndim)

    with Pool() as pool:
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[wav, flam, ferr], pool=pool, backend=backend)
        sampler.run_mcmc(pos, 1000, progress=True)

    print("Finished running emcee.")
    print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")
    sys.exit(0)

    # -------------------------------------------------------- # 
    # --------------------- plotting ------------------------- #
    sampler = emcee.backends.HDFBackend(emcee_savefile)

    samples = sampler.get_chain()
    print(f"{bcolors.CYAN}\nRead in sampler:", emcee_savefile, f"{bcolors.ENDC}")
    print("Samples shape:", samples.shape)

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(emcee_diagnostics_dir + 'mcmc_stackfit_trace.pdf', dpi=200, bbox_inches='tight')

    # Corner plot 
    # First get autocorrelation time and other stuff
    tau = sampler.get_autocorr_time(tol=0)
    burn_in = int(2 * np.max(tau))
    thinning_steps = int(0.5 * np.min(tau))

    print(f"{bcolors.CYAN}")
    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)
    print(f"{bcolors.ENDC}")

    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, \
        verbose=True, truth_color='tab:red', smooth=0.8, smooth1d=0.8)#, \
    #range=[(2.5, 7.0), (-0.3, 0.2), (0.0, 0.15)])

    fig.savefig(emcee_diagnostics_dir + 'mcmc_stackfit_corner.pdf', dpi=200, bbox_inches='tight')

    # Print corner estimates to screen 
    cq_age = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_tau = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])
    cq_lsf = corner.quantile(x=flat_samples[:, 3], q=[0.16, 0.5, 0.84])
    cq_zscatter = corner.quantile(x=flat_samples[:, 4], q=[0.16, 0.5, 0.84])

    # print parameter estimates
    print(f"{bcolors.CYAN}")
    print("Parameter estimates:")
    print("Age [Gyr]: ", cq_age)
    print("log SFH Timescale [Gyr]:", cq_tau)
    print("Visual extinction [mag]:", cq_av)
    print("LSF [Angstroms]:", cq_lsf)
    print("Average error in redshift for galaxies in stack:", cq_zscatter)
    print(f"{bcolors.ENDC}")

    # Plot 100 random models from the parameter space within +-1sigma of corner estimates
    fig3 = plt.figure(figsize=(10,5))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{L_\lambda\ [continuum\ divided]}$', fontsize=15)

    model_count = 0
    ind_list = []

    while model_count <= 100:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)

        sample = flat_samples[ind]
        sample = sample.reshape(ndim)
        print("On count:", model_count)
        print(sample)

        # Get the parameters of the sample
        model_age = sample[0]
        model_tau = sample[1]
        model_av = sample[2]
        model_lsf = sample[3]
        model_zscatter = sample[4]

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        if (model_age >= cq_age[0]) and (model_age <= cq_age[2]) and \
           (model_tau >= cq_tau[0]) and (model_tau <= cq_tau[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]) and \
           (model_lsf >= cq_lsf[0]) and (model_lsf <= cq_lsf[2]) and \
           (model_zscatter >= cq_zscatter[0]) and (model_zscatter <= cq_zscatter[2]):

            m = model(wav, sample[0], sample[1], sample[2], sample[3], sample[4])

            ax3.plot(wav, m, color='firebrick', lw=1.8, alpha=0.05, zorder=2)

            model_count += 1

    print("\nList of randomly chosen indices:", ind_list)

    ax3.plot(wav, flam, color='mediumblue', lw=2.2, zorder=1)
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=1)

    ax3.axhline(y=1.0, ls='--', color='k')

    ax3.set_ylim(0.9, 1.1)

    fig3.savefig(emcee_diagnostics_dir + 'mcmc_stackfit_overplot.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)









