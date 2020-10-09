import matplotlib.pyplot as plt

import sys

def main():

    print("-------------------")
    print("This code is intended to test the working of BC03.")
    print("We will perform two tests.\n")
    print("1: This code will test that calls to csp_galaxev")
    print("from within python provide identical results to calls made")
    print("through the command line. You will need to do the command line")
    print("part and provide the path to those spectra here.\n")
    print("2: A much more IMPORTANT test!")
    print("Given the spectra of a handful of nearby NGC/IC galaxies")
    print("(probably through NED or some other literature sources and preferably UV to FIR)")
    print("and their corresponding stellar population parameters, which were")
    print("hopefully derived independently NOT using BC03 SPS models,")
    print("this code will test that the spectra given by csp_galaxev")
    print("are \"reasonably\" similar to those from the literature.")
    print("The stellar population parameters required are -- stellar mass, ")
    print("age, SFH, Av (and dust law), metallicity, and redshift for the galaxy.")
    print("Applying dust attenuation and redshift is done outside of the BC03 code.")
    print("-------------------\n")


    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
