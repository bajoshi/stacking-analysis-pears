import numpy as np
import scipy
from astropy.modeling import models

import os
import sys

import matplotlib.pyplot as plt

def main():
    """
    This program will check if the way we are currently applying the LSF to a spectrum 
    is correct. 
    
    1. 
    The fast one line solution we are currently using:    
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=lsf_sigma)
    
    for some specified lsf_sigma. The input spectrum is a redshifted high-res model template.
    The lsf_sigma is the 1-sigma width of the galaxy in "angstroms". Convert to pixels using
    the dispersion of the dispersive element. 
    This solution simply filters the given spectrum with a Gaussian that has a specified 
    1-sigma width of lsf_sigma.

    2. 
    A more involved but perhaps intutively easier to understand and also probably more
    accurate way of applying the LSF would be to specify the full width of the galaxy in 
    pixels and then assume that the specified model is emitted by each one of these pixels.
    Pretty sure this is how a slitless simulation software forward models the dispersed 
    images.

    This should produce the correct type of "broadening" instead of a simple Gaussian 
    filter which is not what happends in reality.

    Considerations:
    a. Now, the final spectrum will be a combination of the spectra coming from all these
    pixels along the dispersion direction and then somehow also collapsed from 2D to 1D. 
    
    b. Must also account for a typical galaxy being brighter in the center and dimming 
    radially outward. Can assume a Sersic profile? Or exponential disk + bulge? Or both.

    ----- Make sure to verify the following.
    - The higher the resolution of the grism the worse this effect is. (?)
    - The larger the galaxy along the dispersion direction the worse this effect is.

    This method also conveniently allows the user to assign different spectra to 
    different parts of the galaxy, e.g., a spectrum for the bulge and another one for 
    the disk.

    Algorithm:
    Step I --  

    """

    # Create a dummy spectrum
    spec = np.zeros(1000)
    wav = np.arange(len(spec))

    # Superimpose an emission line on the spectrum
    g = models.Gaussian1D(amplitude=3, mean=300.0, stddev=20.0)
    gauss_1d_emission = g(wav)

    spec = spec + gauss_1d_emission

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, spec)
    plt.show()
    """

    # Now assign some width to the dummy galaxy
    # in pixels
    # say 1-sigma if it is shaped like a Gaussian
    # will deal with Sersic and other complicated profiles later
    galaxy_width = 5
    num_sigma = 1  # e.g., going from -2 to +2 sigma of galaxy width

    # Assume that each time the spectrum shifts 
    # by some amount epsilon. I'm leaving this param
    # free for now.
    # in units of wavelength
    # this is saying that if a given pixel has the center of
    # the emission line at lambda then the adjacent pixel along
    # the dispersion direction will have the line center at
    # lambda + epsilon
    epsilon = 10.0
    # BE CAREFUL!! This currently means epsilon steps of wavelength
    # so that if the wavelenght is sampled at 1 A then epsilon is
    # 30 A but if the wavelength array is sampled at 10 A then epsilon
    # means 300 A.

    print("Width of galaxy provided:", galaxy_width)
    print("Will consider between +- these many sigmas:", num_sigma)
    print("Epsilon:", epsilon)

    exten = int(epsilon * num_sigma * galaxy_width)
    print("Extension:", exten)
    total_pix = num_sigma * galaxy_width
    print("Total pixels to consider:", total_pix)

    final_spec = np.zeros(shape=(total_pix, (len(spec) + exten)))
    print("Shape of final spectra array:", final_spec.shape)
    print("\n")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pix in range(total_pix):

        # Create shifted spectrum
        i = int(epsilon*(1+pix))  # starting index for shift
        shifted_spec = np.zeros(len(spec) + i)
        shifted_spec[i:] = spec

        print("\nAt pixel:", pix+1)
        print("Shifting to index:", i)
        print("len of shifted spec:", len(shifted_spec))

        # Now fold it into the final spectrum
        final_spec[pix, :len(shifted_spec)] = shifted_spec

        ax.plot(shifted_spec)

    final_spec_comb = np.sum(final_spec, axis=0)
    print("Shape of final LSF broadened spectrum:", final_spec_comb.shape)

    ax.plot(final_spec_comb, color='k')
    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


