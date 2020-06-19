import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Desktop/FIGS/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"

stacking_utils_dir = stacking_analysis_dir + "util_codes"
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

def lsf_convolve(spec):

    mock_lsf = Gaussian1DKernel(2.0)
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
    extended_grism_wav_grid = np.arange(4000, 12000 + 40, 40.0)

    resampled_flux = np.zeros((len(extended_grism_wav_grid)))
    for k in range(len(extended_grism_wav_grid)):

        if k == 0:
            lam_step_high = extended_grism_wav_grid[k+1] - extended_grism_wav_grid[k]
            lam_step_low = lam_step_high

        elif k == len(extended_grism_wav_grid) - 1:
            lam_step_low = extended_grism_wav_grid[k] - extended_grism_wav_grid[k-1]
            lam_step_high = lam_step_low

        else:
            lam_step_high = extended_grism_wav_grid[k+1] - extended_grism_wav_grid[k]
            lam_step_low = extended_grism_wav_grid[k] - extended_grism_wav_grid[k-1]

        new_ind = np.where((spec_wav >= extended_grism_wav_grid[k] - lam_step_low) & \
            (spec_wav < extended_grism_wav_grid[k] + lam_step_high))[0]

        resampled_flux[k] = np.mean(spec_flux[new_ind])

    return resampled_flux, extended_grism_wav_grid

def chop_spectrum(spec_wav, spec_flux, final_wav_grid):

    # Finally chop to required wavlength range
    wav_start_idx = np.argmin(abs(spec_wav - final_wav_grid[0]))
    wav_end_idx = np.argmin(abs(spec_wav - final_wav_grid[-1]))
    grism_spec = spec_flux[wav_start_idx: wav_end_idx+1]

    return grism_spec

def main():

    # Read in templates and redshifts file 
    templates = np.genfromtxt('template_and_redshift_choices.txt', dtype=None, names=True, encoding='ascii')

    # Read in each template, modify it and store 
    # all modified templates in a large numpy array
    # First define the final wavelength grid
    grism_low_wav = 6000
    grism_high_wav = 9500
    pears_spec_points = 88

    final_wav_grid = np.linspace(grism_low_wav, grism_high_wav, pears_spec_points)
    templates_with_mods = np.zeros((len(templates), len(final_wav_grid)))

    for i in range(len(templates)):

        current_template_name = templates['template_name'][i]
        current_redshift = templates['redshift'][i]

        print(i, ":  ", current_template_name, "  ", current_redshift)

        # Read in template
        tt = np.genfromtxt(current_template_name, dtype=None, names=True, encoding='ascii')
        current_template_wav = tt['wav']
        current_template_llam = tt['llam']

        # Mods. Steps:
        # * Redshift spectrum
        # * Convolve with a line spread function
        # * The BC03 models have no dust. Add in dust if needed according to the Calzetti prescription.
        # * Not all galaxies within a sample will be the same brightness so you need take this into account. 
        # * Add random noise to each flux point.
        # * Convolve model with grism sensitivity curve. Downsample to grism resolution, 
        # while also adding in systematic noise... i.e., correlated flux measurements.
        redshifted_wav, redshifted_flux = redshift_spectrum(current_template_wav, current_template_llam, current_redshift)
        # = add_dust()
        # = luminosity_func_mod()
        spec_noise = add_statistical_noise(redshifted_flux)
        lsf_convolved_spectrum = lsf_convolve(spec_noise)
        grism_spec, extended_grism_wav_grid = downsample(redshifted_wav, lsf_convolved_spectrum)
        final_spec = chop_spectrum(extended_grism_wav_grid, grism_spec, final_wav_grid)

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(final_wav_grid, final_spec)
        ax.set_xscale('log')
        ax.set_xlim(5000, 10000)
        plt.show()

        plt.cla()
        plt.clf()
        plt.close()

        if i > 7480: break
        """

        # Add into numpy array
        templates_with_mods[i] = final_spec

    np.save("modified_templates.npy", templates_with_mods)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)