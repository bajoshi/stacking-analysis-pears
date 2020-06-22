import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Desktop/FIGS/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"

stacking_utils_dir = stacking_analysis_dir + "util_codes/"
sys.path.append(stacking_utils_dir)
import proper_and_lum_dist as pl

def redshift_spectrum(template_wav, template_llam, redshift):

    # Redshift wavelengths
    redshifted_wav = template_wav * (1 + redshift)

    # Get luminosity distance to convert l_lambda to f_lambda
    dl = pl.luminosity_distance(redshift)  # in mpc
    mpc_to_cm = 3.086e24
    dl = dl * mpc_to_cm  # convert to cm
    redshifted_flux = template_llam / ((4 * np.pi * dl * dl) * (1 + redshift))

    return redshifted_wav, redshifted_flux

def add_stellar_vdisp(spec_wav, spec_flux, vdisp):

    # Now compute the broadened spectrum by numerically
    # integrating a Gaussian stellar velocity function
    # with the stellar vdisp.
    # Integration done numerically as a Riemann sum.

    speed_of_light = 299792.458  # km per second
    delta_v = 1.0

    vdisp_spec = np.zeros(len(spec_wav))

    #print(len(spec_wav))

    for w in range(len(spec_wav)):
        #print(w)
        
        lam = spec_wav[w]

        I = 0
        
        # Now compute the integrand numerically
        # between velocities that are within 
        # +- 3-sigma using the specified velocity dispersion.
        # Mean of all velocities should be 0,
        # of course since the avg vel of all stars within
        # a rotating disk or in an elliptical galaxy should be zero.
        for v in np.arange(-3*vdisp, 3*vdisp, delta_v):

            beta = 1 + (v/speed_of_light)
            new_lam = lam / beta
            new_lam_idx = np.argmin(abs(spec_wav - new_lam))

            flux_at_new_lam = spec_flux[new_lam_idx]

            gauss_exp_func = np.exp(-1*v*v/(2*vdisp*vdisp))

            prod = flux_at_new_lam * gauss_exp_func
            I += prod

        vdisp_spec[w] = I / (vdisp*np.sqrt(2*np.pi))

    return vdisp_spec

def lsf_convolve(spec):

    gauss_kernel_sigma = 8.0

    mock_lsf = Gaussian1DKernel(gauss_kernel_sigma)
    lsf_convolved_spectrum = convolve(spec, mock_lsf, boundary='extend')

    return lsf_convolved_spectrum

def add_statistical_noise(spec):

    # say the template has a 3-sigma measuremnt of each flux point
    avg_err = 0.33 * spec

    # put in random noise in the model
    spec_noise = np.zeros(len(spec))
    for k in range(len(spec)):

        mu = spec[k]
        sigma = avg_err[k]

        spec_noise[k] = np.random.normal(mu, sigma, 1)

    return spec_noise

def downsample(spec_wav, spec_flux):

    # First multiply by grism sensitivity curve

    # Downsample to grism resolution
    final_wav_grid = get_final_wav_grid()

    resampled_flux = np.zeros((len(final_wav_grid)))
    for k in range(len(final_wav_grid)):

        if k == 0:
            lam_step_high = final_wav_grid[k+1] - final_wav_grid[k]
            lam_step_low = lam_step_high

        elif k == len(final_wav_grid) - 1:
            lam_step_low = final_wav_grid[k] - final_wav_grid[k-1]
            lam_step_high = lam_step_low

        else:
            lam_step_high = final_wav_grid[k+1] - final_wav_grid[k]
            lam_step_low = final_wav_grid[k] - final_wav_grid[k-1]

        new_ind = np.where((spec_wav >= final_wav_grid[k] - lam_step_low) & \
            (spec_wav < final_wav_grid[k] + lam_step_high))[0]

        resampled_flux[k] = np.mean(spec_flux[new_ind])

    return resampled_flux

def chop_spectrum(spec_wav, spec_flux, chop_lim_low, chop_lim_high):

    # Chop to required wavlength range
    wav_start_idx = np.argmin(abs(spec_wav - chop_lim_low))
    wav_end_idx = np.argmin(abs(spec_wav - chop_lim_high))

    short_wav = spec_wav[wav_start_idx: wav_end_idx+1]
    short_spec = spec_flux[wav_start_idx: wav_end_idx+1]

    return short_wav, short_spec

def get_final_wav_grid():

    grism_low_wav = 6000
    grism_high_wav = 9500
    pears_spec_points = 88

    final_wav_grid = np.linspace(grism_low_wav, grism_high_wav, pears_spec_points)

    return final_wav_grid

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print("Starting template mods --", dt.now())

    # Read in templates and redshifts file 
    templates = np.genfromtxt('template_and_redshift_choices.txt', dtype=None, names=True, encoding='ascii')

    # Read in each template, modify it and store 
    # all modified templates in a large numpy array
    # First get the final wavelength grid
    final_wav_grid = get_final_wav_grid()
    templates_with_mods = np.zeros((len(templates), len(final_wav_grid)))

    i_init = 8000
    for i in range(i_init, len(templates)):

        current_template_name = templates['template_name'][i]
        current_redshift = templates['redshift'][i]

        print(i, ":  ", current_template_name, "  ", current_redshift)

        # Read in template
        tt = np.genfromtxt(current_template_name, dtype=None, names=True, encoding='ascii')
        current_template_wav = tt['wav']
        current_template_llam = tt['llam']

        # For now randomly assign a stellar velocity dispersion
        # value betwen 200 to 300 km/s 
        stellar_vdisp_arr = np.linspace(200, 300, 11)
        stellar_vdisp = np.random.choice(stellar_vdisp_arr, size=1)

        # Decide how to shorten the BC03 spectra
        # These wavelengths are in angstroms in the rest-frame
        chop_lim_low = 3000
        chop_lim_high = 10000

        # Mods. Steps:
        # * First chop the spectrum so that the remaining computationally expensive steps 
        # can be done relatively faster
        # * Redshift spectrum
        # * Convolve with a line spread function
        # * The BC03 models have no dust. Add in dust if needed according to the Calzetti prescription.
        # * Not all galaxies within a sample will be the same brightness so you need take this into account. 
        # * Add random noise to each flux point.
        # * Convolve model with grism sensitivity curve. Downsample to grism resolution, 
        # while also adding in systematic noise... i.e., correlated flux measurements.

        short_wav, short_spec = chop_spectrum(current_template_wav, current_template_llam, chop_lim_low, chop_lim_high)
        redshifted_wav, redshifted_flux = redshift_spectrum(short_wav, short_spec, current_redshift)
        vdisp_flux = add_stellar_vdisp(redshifted_wav, redshifted_flux, stellar_vdisp)
        # = add_dust()
        # = luminosity_func_mod()
        spec_noise = add_statistical_noise(vdisp_flux)
        lsf_convolved_spectrum = lsf_convolve(spec_noise)
        grism_spec = downsample(redshifted_wav, lsf_convolved_spectrum)
        
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(current_template_wav, current_template_llam, color='k')
        ax.plot(redshifted_wav, redshifted_flux)
        ax.plot(redshifted_wav, vdisp_flux)
        ax.plot(final_wav_grid, grism_spec)
        ax.set_xscale('log')
        ax.set_xlim(5000, 10000)
        plt.show()

        plt.cla()
        plt.clf()
        plt.close()

        if i > i_init+20: sys.exit(0)
        """

        # Add into numpy array
        templates_with_mods[i] = grism_spec

    np.save(figs_dir + "modified_templates.npy", templates_with_mods)

    # Total time taken
    print("Total time taken for all mods --", "{:.2f}".format((time.time() - start)/60.0), "minutes.")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)