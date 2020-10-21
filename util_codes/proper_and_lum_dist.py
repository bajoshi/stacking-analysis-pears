import scipy.integrate as spint
import numpy as np

# -------- Define cosmology -------- # 
# Planck 2018
H0 = 67.4  # km/s/Mpc
omega_m0 = 0.315
omega_r0 = 8.24e-5
omega_lam0 = 1.0 - omega_m0 - omega_r0

speed_of_light_kms = 299792.458  # km per s

def print_info():

    print("Plank 2018 cosmology assumed.")
    print("H0: ", H0, "km/s/Mpc")
    print("Omega_m:", omega_m0)
    print("Omega_lambda:", "{:.3f}".format(omega_lam0))
    print("Omega_r:", omega_r0)

    return None

def proper_distance(redshift):
    """
    This function will integrate 1/(a*a*H)
    between scale factor at emission to scale factor of 1.0.

    Will return proper distance in megaparsecs.
    """
    ae = 1 / (1 + redshift)

    p = lambda a: 1/(a*a*H0*np.sqrt((omega_m0/a**3) + (omega_r0/a**4) + omega_lam0 + ((1 - omega_m0 - omega_r0 - omega_lam0)/a**2)))
    dp = spint.quadrature(p, ae, 1.0)

    dp = dp[0] * speed_of_light_kms

    return dp

def luminosity_distance(redshift):
    """
    Returns luminosity distance in megaparsecs for a given redshift.
    """

    # Get proper distance and multiply by (1+z)
    dp = proper_distance(redshift)  # returns answer in Mpc
    dl = dp * (1+redshift)  # dl also in Mpc

    return dl

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = luminosity_distance(redshift)  # returns dl in Mpc
    dl = dl * 3.09e24  # convert to cm

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux


