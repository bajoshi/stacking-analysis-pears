from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
threedhst_datadir = home + "/Desktop/3dhst_data/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
from pears_and_3dhst import read_3dhst_cats
import fullfitting_grism_broadband_emlines as ff

speed_of_light = 299792458e10  # angsroms per second

if __name__ == '__main__':
    

    sys.exit(0)