from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
from matplotlib.backends.backend_pdf import PdfPages

sens_curve = np.genfromtxt('ACS.WFC.1st.sens.7.dat', dtype=None, names=True, skip_header=4)

senslam = sens_curve['Lambda']
sens = sens_curve['sensitivity']
senserr = sens_curve['error']

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.errorbar(senslam, sens, yerr=senserr,\
             fmt='o', color='k', ecolor='r', markersize=2, markeredgecolor='k', capsize=0)

# Normalize
sensmax = max(sens)
sens = sens / sensmax
senserr = senserr / sensmax

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.errorbar(senslam, sens, yerr=senserr,\
             fmt='o', color='k', ecolor='r', markersize=2, markeredgecolor='k', capsize=0)

plt.show()