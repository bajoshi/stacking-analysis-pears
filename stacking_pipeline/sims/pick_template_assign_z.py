import numpy as np

import os
import sys

import matplotlib.pyplot as plt

def generate_redshift_dist(num_galaxies, avg_z_err, zlow, zhigh):

    # Uniform distribution of redshifts
    redshift_dist = np.linspace(zlow, zhigh, num_galaxies)

    # Now assume some redshift error
    # Then randomly sample assuming the redshift error is gaussian
    redshift_gauss_vals = []

    for i in range(len(redshift_dist)):
        mu = redshift_dist[i]
        sigma = avg_z_err
        redshift_gauss_vals.append(float(np.random.normal(mu, sigma, size=1)))

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts, bins, _trash = ax.hist(redshift_gauss_vals, 30, density=True)
    ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), \
        color='tab:red', lw=2.0)
    plt.show()
    """

    redshift_gauss_vals = np.asarray(redshift_gauss_vals)

    return redshift_gauss_vals

def main():

    # Parameters for this part of the simulation
    num_galaxies = 1e4
    avg_z_err = 0.03
    zlow = 0.16
    zhigh = 0.96

    # First generate a random redshift distribution within the chosen redshift range
    final_redshifts = generate_redshift_dist(num_galaxies, avg_z_err, zlow, zhigh)

    # Now chose models at random from the existing set
    model_set = ['bc03_template_0.10_gyr.txt', 'bc03_template_0.30_gyr.txt', \
    'bc03_template_0.50_gyr.txt', 'bc03_template_1.00_gyr.txt', \
    'bc03_template_2.00_gyr.txt', 'bc03_template_4.00_gyr.txt', \
    'bc03_template_6.00_gyr.txt']

    fh = open('template_and_redshift_choices.txt', 'w')
    fh.write("#  template_name  redshift" + "\n")

    for i in range(int(num_galaxies)):
        chosen_model = np.random.choice(model_set, size=1)
        chosen_model = chosen_model[0]
        fh.write(str(chosen_model) + "  " + "{:.3f}".format(final_redshifts[i]) + "\n")

    fh.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)