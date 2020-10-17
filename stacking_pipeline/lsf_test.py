import numpy as np
import scipy

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
    spec = np.ones(1000)

    # Superimpose an emission line on the spectrum




    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


