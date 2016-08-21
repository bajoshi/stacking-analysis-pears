from __future__ import division
import numpy as np
import pyfits as pf

import sys, os
import logging

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from mpl_toolkits.mplot3d import Axes3D

def make3dfig():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	return ax

if __name__ == '__main__':

	import matplotlib
	print matplotlib.__version__

	