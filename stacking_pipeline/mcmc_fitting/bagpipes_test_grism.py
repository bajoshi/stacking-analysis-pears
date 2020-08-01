import numpy as np
import bagpipes as pipes

import os
import sys

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Documents/pears_figs_data/"
stacking_analysis_dir = home + "/Documents/GitHub/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + "stacking_pipeline/")
import grid_coadd as gd

def load_pears_spec_for_bagpipes(pears_id):

    grism_lam_obs, grism_flam_obs, grism_ferr_obs, return_code = gd.get_pears_data(int(pears_id), 'GOODS-S')

    bagpipes_spec = np.c_[grism_lam_obs, grism_flam_obs, grism_ferr_obs]

    return bagpipes_spec

def bagpipes_test_grism_fit():

    # Get data
    pears_id = '109151'

    galaxy = pipes.galaxy(pears_id, load_pears_spec_for_bagpipes, photometry_exists=False)
    #fig = galaxy.plot()

    # Define fit instructions
    dblplaw = {}                        
    dblplaw["tau"] = (0., 15.)            
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta_prior"] = "log_10"
    dblplaw["massformed"] = (1., 15.)
    dblplaw["metallicity"] = (0.1, 2.)
    dblplaw["metallicity_prior"] = "log_10"
    
    nebular = {}
    nebular["logU"] = -3.
    
    dust = {}
    dust["type"] = "CF00"
    dust["eta"] = 2.
    dust["Av"] = (0., 2.0)
    dust["n"] = (0.3, 2.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3
    
    fit_instructions = {}
    fit_instructions["redshift"] = (0.75, 1.25)
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift_prior"] = "Gaussian"
    fit_instructions["redshift_prior_mu"] = 0.9
    fit_instructions["redshift_prior_sigma"] = 0.05
    fit_instructions["dblplaw"] = dblplaw 
    fit_instructions["nebular"] = nebular
    fit_instructions["dust"] = dust

    fit_instructions["veldisp"] = (1., 1000.)   #km/s
    fit_instructions["veldisp_prior"] = "log_10"

    calib = {}
    calib["type"] = "polynomial_bayesian"
    
    calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.0
    calib["0_prior_sigma"] = 0.25
    
    calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.25
    
    calib["2"] = (-0.5, 0.5)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.25
    
    fit_instructions["calib"] = calib

    # Now fit
    fit = pipes.fit(galaxy, fit_instructions, run='test')
    fit.fit(verbose=True)

    return None

def main():

    bagpipes_test_grism_fit()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)