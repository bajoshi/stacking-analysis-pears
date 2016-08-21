from __future__ import division
import numpy as np
import pyfits as pf
import glob, os, sys
import warnings

data_path = "/Users/bhavinjoshi/Documents/PEARS/data_spectra_only/"

class Error(Exception):
    """
        Base class for errors in this program.
    """

class counter():
    """
        Class that initializes all counters at the start of the program to 0.
    """
    def __init__(self):
        self.galaxies = 0

class DivideError(Error):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def makeregions(catalog, catalogname, id=True, idcol=0, racol=1, deccol=2):
    """
        Creates a ds9 regions file for the supplied catalog.
        
        This function will assume that the RA and DEC columns in the supplied catalog
        are 1 and 2 respectively. If this is not the case then the user must give an 
        input that is not the default value.
        
        It also assumes that the user needs the region to have a text field and the text field
        is given by the 0 column in the catalog; by default the text is the id of the object. 
        This can also be changed by the user.
        
        Enter the FULL PATH of the catalog as the second argument.
        The given catalog should be a numpy recarray i.e. the catalog should come from a numpy
        genfromtxt statement.
        The function will place the region file in the same folder as the catalog.
        Also make sure that the catalog has the extension .cat or .txt.
    """

    print "The total number of sources in the catalog are ", len(catalog)

    ext = os.path.basename(catalogname).split('.')[-1]
    
    if ext == 'cat':
        regfilename = catalogname.replace('.cat','.reg')
    elif ext == 'txt':
        regfilename = catalogname.replace('.cat','.reg')
    else:
        print "Unrecognized extension for catalog."
        print "Please make sure that the extension is .cat or .txt."
        return

    regfile = open(regfilename, 'wa')

    if id:
        for i in range(len(catalog)):
            regfile.write('fk5;circle(' + str(catalog[i][racol]) + ',' + str(catalog[i][deccol]) +\
                          ',0.5") # color=green width=1 text = {' + str(catalog[i][idcol]) + '};' + '\n')
    else:
        for i in range(len(catalog)):
            regfile.write('fk5;circle(' + str(catalog[i][racol]) + ',' + str(catalog[i][deccol]) +\
                          ',0.5") # color=green width=1;' + '\n')

    regfile.close()
    print "Created regions file -- ", os.path.basename(regfilename), "in the same folder as catalog."


if __name__ == '__main__':
    
    # testing...
    catname = '/Users/baj/Desktop/FIGS/GS2/cat_v023/GS2_prelim_science_v0.23.cat'
    #makeregions(cat, catname)