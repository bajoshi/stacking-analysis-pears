import numpy as np

import os
import sys

import matplotlib.pyplot as plt

def deredshift(template_wav, template_llam, redshift):

    

    return redshifted_wav, redshifted_flux

def main():

    # Read in templates and redshifts file 
    templates = np.genfromtxt('template_and_redshift_choices.txt', dtype=None, names=True, encoding='ascii')

    # Read in each template, modify it and store 
    # all modified templates in a large numpy array
    # First define the final wavelength grid
    final_wav_grid = 
    templates_with_mods = np.zeros(len(templates), len(final_wav_grid), len(final_wav_grid))

    for i in range(len(templates)):

        current_template_name = templates['template_name'][i]

        # Read in template
        tt = np.genfromtxt(current_template_name, dtype=None, names=True, encoding='ascii')
        current_template_wav = tt['wav']
        current_template_llam = tt['llam']

        # Mods # Steps:
        # 1. 
        # 2. 
        # 3. 
        # 4. Not all galaxies within a sample will be the same brightness so you need take this into account. 
        = deredshift(current_template_wav, current_template_llam)
        = lsf_convolve()
        = luminosity_func_mod()
        = add_statistical_noise()
        = downsample()
        = add_systematic_noise()

        # Add into numpy array

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)