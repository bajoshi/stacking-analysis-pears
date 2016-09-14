from __future__ import division

import numpy as np

import os

import matplotlib.pyplot as plt
import seaborn as sns

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"
figures_dir = stacking_analysis_dir + "figures/"

sns.set_style("white")
    
cat = np.genfromtxt('/Users/bhavinjoshi/Desktop/FIGS/new_codes/color_stellarmass.txt',
 dtype=None, names=['z'], usecols=(4), skip_header=3)

z_phot = cat['z']

print max(z_phot), min(z_phot)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(z_phot, 15, alpha=0.5)
ax.set_xlim(0.6, 1.25)

sns.despine()

fig.savefig(figures_dir + 'redshift_dist.eps', dpi=300)
plt.show()