import numpy as np
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u

from tqdm import tqdm
import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')

if __name__ == '__main__':

    data_dir = '/Volumes/Joshi_external_HDD/figs_raw_data/'

    obs_table = Observations.query_criteria(proposal_id='13779', \
        intentType='science', obs_collection=['HST'])

    data_products = Observations.get_product_list(obs_table)

    manifest = Observations.download_products(data_products, mrp_only=False, productType="SCIENCE", download_dir=data_dir)

    print(manifest)

    sys.exit(0)