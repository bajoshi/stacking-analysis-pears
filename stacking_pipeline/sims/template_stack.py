import numpy as np
from astropy.modeling import models, fitting

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Desktop/FIGS/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + "util_codes/")
sys.path.append(stacking_analysis_dir + "stacking_pipeline/")
sys.path.append(stacking_analysis_dir + "stacking_pipeline/sims/")
import grid_coadd as gd
import proper_and_lum_dist as pl
from template_mods import get_final_wav_grid

def main():

    # ----------------------------------------- Code config params ----------------------------------------- #
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 25  # somewhat arbitrarily chosen # pretty much trial and error

    # Set the ends of the lambda grid
    # This is dependent on the redshift range being considered
    lam_grid_low = 3000
    lam_grid_high = 7600

    lam_grid = np.arange(lam_grid_low, lam_grid_high, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.16 < z < 0.96

    # ----------------------------------------- Prep stuff ----------------------------------------- #
    # First get the final wavelength grid
    lam_obs = get_final_wav_grid()  # This is the same as lam_obs in the other code

    # Define empty arrays and lists for saving stacks
    template_old_llam = np.zeros(len(lam_grid))
    template_old_llamerr = np.zeros(len(lam_grid))
    template_old_llam = template_old_llam.tolist()
    template_old_llamerr = template_old_llamerr.tolist()

    template_num_points = np.zeros(len(lam_grid))
    template_num_galaxies = np.zeros(len(lam_grid))

    # Read in templates and redshifts file 
    templates = np.genfromtxt('template_and_redshift_choices.txt', dtype=None, names=True, encoding='ascii')

    # Read in all modified templates
    templates_with_mods = np.load(figs_dir + "modified_templates.npy")

    # Use the constant measurement significane that the modification code used
    # Confirm with that code that this number is the same in both places
    avg_template_meas_sigma = 3.0

    # Create figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    # --------------------------------------- Loop over all spectra and coadd them --------------------------------------- #
    for u in range(len(templates)):

        # This step should only be done on the first iteration within a grid cell
        # This converts every element (which are all 0 to begin with) 
        # in the flux and flux error arrays to an empty list
        # This is done so that function add_spec() can now append to every element
        if u == 0:
            for x in range(len(lam_grid)):
                template_old_llam[x] = []
                template_old_llamerr[x] = []

        current_template_name = templates['template_name'][u]
        current_redshift = templates['redshift'][u]

        print(u, ":  ", current_template_name, "  ", current_redshift)

        # Read in modified template flux
        current_template_flux = templates_with_mods[u]

        # Deredshift the observed data 
        dl = pl.luminosity_distance(current_redshift)  # in mpc
        mpc_to_cm = 3.086e24
        dl = dl * mpc_to_cm  # convert to cm 

        current_template_err = current_template_flux / avg_template_meas_sigma

        template_lam_em = lam_obs / (1 + current_redshift)
        template_llam_em = current_template_flux * (1 + current_redshift) * (4 * np.pi * dl * dl)
        template_lerr = current_template_err * (1 + current_redshift) * (4 * np.pi * dl * dl)

        # Subtract continuum by fitting a third degree polynomial
        # Continuum fitted with potential emission line areas masked
        p_init = models.Polynomial1D(degree=5)
        fit_p = fitting.LinearLSQFitter()

        # mask emission lines 
        template_llam_em_masked, template_mask_ind = gd.mask_em_lines(template_lam_em, template_llam_em)
        
        # Now fit
        p_template = fit_p(p_init, template_lam_em, template_llam_em_masked)

        # Now divide continuum
        template_llam_em = template_llam_em / p_template(template_lam_em)

        # Also divide errors 
        template_lerr = template_lerr / p_template(template_lam_em)

        # add the continuum subtracted spectrum
        template_old_llam, template_old_llamerr, template_num_points, template_num_galaxies = \
        gd.add_spec(template_lam_em, template_llam_em, template_lerr, template_old_llam, template_old_llamerr, \
            template_num_points, template_num_galaxies, lam_grid, lam_step)

        # ax.plot(template_lam_em, template_llam_em, ls='-', color='palegreen', linewidth=0.5, alpha=0.4)

    # Now take the median of all flux points appended within the list of lists
    # This function also does the 3-sigma clipping
    template_old_llam, template_old_llamerr = gd.take_median(template_old_llam, template_old_llamerr, lam_grid)

    template_old_llam = np.asarray(template_old_llam)
    template_old_llamerr = np.asarray(template_old_llamerr)

    # Shift it to force stack value =1.0 at 4500A
    #shift_idx = np.argmin(abs(lam_grid - 4500))
    #template_old_llam /= template_old_llam[shift_idx]

    # Plot stacks
    ax.plot(lam_grid, template_old_llam, '.-', color='tab:green', linewidth=1.5, \
        markeredgecolor='tab:green', markersize=1.0, zorder=5)
    ax.fill_between(lam_grid, template_old_llam - template_old_llamerr, template_old_llam + template_old_llamerr, \
        color='gray', alpha=0.5, zorder=5)

    ax.set_xlim(lam_grid_low, lam_grid_high)
    ax.set_ylim(0.9, 1.1)  # if dividing by the continuum instead of subtracting
    ax.axhline(y=1.0, ls='--', color='k')
    ax.minorticks_on()

    gd.add_line_labels(ax)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
