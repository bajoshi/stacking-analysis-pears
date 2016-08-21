# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pyfits as pf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, time
pgf_preamble = {"pgf.texsystem": "pdflatex"}
mpl.rcParams.update(pgf_preamble)

def plot_spectrum(flux, flux_err, lam, zp):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [erg/cm^2/s/\AA]$')
    ax.axhline(y=0,linestyle='--')
    ax.set_xlim(5700, 9500)
    ax.errorbar(lam,flux,yerr=flux_err,color='black', linewidth=1)
    
    lam_break = 4000 * (1 + zp)
    l4100 = 4100 * (1 + zp)
    l4000 = 4000 * (1 + zp)
    l3950 = 3950 * (1 + zp)
    l3850 = 3850 * (1 + zp)
    
    ax.axvline(x = lam_break, linestyle='--', color='r', linewidth=1)
    
    #ax.axvline(x = l4100, linestyle='--', color='g', linewidth=1)
    #ax.axvline(x = l4000, linestyle='--', color='g', linewidth=1)
    #ax.axvline(x = l3950, linestyle='--', color='g', linewidth=1)
    #ax.axvline(x = l3850, linestyle='--', color='g', linewidth=1)
    
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    
    plt.show()

#z_low = 0.6
#z_high = 1.235

# read in all matched files
cdfn1 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn1.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn2 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn2.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn3 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn3.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn4 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn4.txt',\
                      dtype=None, skip_header=3, names=True)

cdfs1 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs1.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs2 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs2.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs3 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs3.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs4 = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs4.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs_new = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_new.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs_udf = np.genfromtxt('/Users/bhavinjoshi/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_udf.txt',\
                      dtype=None, skip_header=3, names=True)

all_fields_north = [cdfn1, cdfn2, cdfn3, cdfn4]
all_fields_south = [cdfs1, cdfs2, cdfs3, cdfs4, cdfs_new, cdfs_udf]
# 2032 total objects

data_path = "/Users/bhavinjoshi/Documents/PEARS/data_spectra_only/"

redshift = []
d4000_index = []
d4000_err = []
stellar_mass = []

cdfn1_skip = [33944, 35090, 35700, 36830, 37116, 37285, 36739, 38146, 37644, 38345, 39394, 40010, 39842, 39987, 41172, 41064, 41499, 42132, 41931, 41546, 42058, 42245, 41540, 41932, 42014, 42526, 42430, 43079, 44207, 46348, 45117, 46694, 47644, 47252, 46687, 48681, 49098, 48216, 48250, 50542, 51485, 52241, 52591, 54384, 54327, 54815, 54689, 55392, 57038, 58650, 59121, 59472, 59873, 60049, 61113, 62562, 62543, 63169, 63410, 64394, 65347, 66045, 62376]
dontbelieve_cdfn1 = [35309,37044,36569,37541,37670,37908,38344,38812,38392,39311,41018,41630,44367,44060,44562,44835,45163,\
                     44021,46085,45150,46952,47511,48514,48954,48229,49001,50265,50210,49569,51988,51220,52497,53198,52869,\
                     53566,55063,57178,58124,58595,59445,59531,60003,60666,59863,60675,61065,61418,62388,63084,64178,65170,\
                     65589,66217,46914,46626,53403]
maybecheckagain_cdfn1 = [41031,44678,44001,45506,46214,45958,52948,52896,56190,60858]

cdfn2_skip = [64394, 65347, 66045, 66218, 65831, 67460, 68134, 68057, 68430, 68518, 68941, 69195, 69850, 72081, 72168, 72809, 73331, 73362, 72684, 74943, 76107, 77076, 77232, 77570, 76936, 78220, 76773, 78906, 78345, 80090, 79986, 80645, 82035, 82391, 82693, 84233, 84028, 84502, 84467, 84143, 84543, 84626, 84617, 85879, 86941, 88045, 88207, 90145, 90359, 90526, 90879, 90907, 91695, 91795, 93012, 92509, 94234]
dontbelieve_cdfn2 = [64178,65170,65589,66217,67081,66873,67551,67343,68706,69509,69819,69419,70082,69106,70677,70974,71675,\
                     71974,71252,71891,72261,72986,73160,73626,74125,73635,75179,75429,74794,76123,75917,75659,76359,77037,\
                     76917,77819,78263,78349,77596,78180,80098,79903,81946,82400,82523,82368,82799,83434,84321,84033,85023,\
                     85556,85385,85372,85489,86234,87608,88056,87634,88175,89030,89189,89239,90095,90064,90248,90517,91367,\
                     91611,91469,91947,92233,93154,93137,93208,93405,93394,93719,93968,94073,94364,81988,91162]
maybecheckagain_cdfn2 = [64523,68696,70874,75367,75556,79928,83432,85356,88566,88858,94262]



# maybewrongredshift cdfn3 112486
cdfn3_skip = [84077, 85141, 85410, 84974, 84715, 86697, 88448, 88768, 89877, 89845, 90194, 91474, 92989, 92316, 93328, 93646, 94117, 94458, 97904, 100950, 100857, 100776, 102160, 102719, 102575, 103192, 105674, 105709, 105902, 106694, 106752, 107999, 107654, 108932, 110708, 112235, 112486, 115830]
dontbelieve_cdfn3 = [84577, 84551, 85100, 85399, 88265, 88171, 88591, 89683, 89383, 90143, 91587, 91377, 91951, 91724, 92450, 92603, 93025, 94782, 95898, 96352, 95449, 98420, 98209, 100202, 102600, 104213, 108010, 108239, 107452, 108721, 109561, 110223, 111351, 111054, 112301, 113696, 115636]
maybecheckagain_cdfn3 = [91612, 100486, 105729]



cdfn4_skip = [100728, 101185, 104023, 103443, 108279, 108871, 109867, 110403, 110727, 111096, 112383, 113600, 113595, 116527, 118319, 118335, 118864, 119684, 119723, 120750, 120710, 121225, 121718, 121834, 121428, 122303, 122257, 122855, 122827, 123269, 123162, 123296, 123504, 124893, 125212]
dontbelieve_cdfn4 = [100137, 100435, 102530, 102547, 102134, 104306, 104879, 105303, 105461, 107152, 108321, 108529, 108799, 108448, 108774, 110145, 110595, 111648, 112100, 112264, 113506, 113746, 117715, 117847, 118394, 118534, 119616, 120572, 120691, 121093, 121635, 121917, 122475, 122616, 122757, 123008, 123054, 124177, 124678, 125809, 125934, 125948]
maybecheckagain_cdfn4 = [114628, 118148, 123476]
#####################

cdfs1_skip = [10593, 12230, 13011, 13988, 14093, 15281, 15336, 15513, 15093, 14990, 15793, 15921, 16183, 16371, 16488, 16683, 17321, 17587, 17494, 17950, 18610, 18213, 18882, 18972, 19003, 18337, 19569, 19652, 19733, 20072, 20265, 20387, 20694, 20877, 21363, 21777, 21605, 21730, 22233, 22266, 22153, 22513, 22486, 22897, 23034, 23093, 23136, 23177, 23920, 23992, 24360, 24609, 25164, 25205, 25687, 26450, 26728, 26160, 27183, 27652, 28280, 28000, 29417, 29252, 29733, 29626, 29801, 29564, 30032, 29922, 30887, 31086, 31227, 31736, 31325, 31861, 32847, 30492, 32905, 33074, 32879, 33422, 33294, 33013, 34245, 34620, 34880, 35134, 35692, 35783, 34803, 35989, 35993, 35255, 35929, 36001, 34789, 36277, 36084, 35762, 36143, 35878]
dontbelieve_cdfs1 = [10429, 10465, 11099, 12350, 11507, 12457, 12949, 12926, 13283, 13732, 14405, 15324, 16039, 15929, 16179, 16788, 17024, 17055, 17022, 16290, 17163, 17264, 17368, 17413, 17468, 17669, 18065, 17999, 18484, 19435, 19473, 19585, 19591, 19683, 19952, 20714, 20681, 21023, 20901, 21891, 21647, 21819, 22051, 22217, 22316, 22514, 22563, 22784, 23059, 23155, 23246, 23413, 23524, 24599, 24717, 24671, 24605, 24521, 24919, 25253, 25462, 25474, 25860, 26614, 27304, 27562, 27862, 28310, 28588, 29515, 29630, 30763, 31772, 31570, 32566, 32224, 32173, 33247, 32885, 33498, 33949, 34000, 34243, 33734, 35116, 34128, 35657, 36026, 36213, 36172, 35589, 36385, 35579, 35475, 36110]
maybecheckagain_cdfs1 = [14215, 24707, 25184]

cdfs2_skip = [37959, 39130, 39266, 39752, 38350, 40048, 39116, 40117, 40286, 40243, 40565, 39835, 40295, 40899, 41078, 41807, 41822, 41615, 41299, 42425, 43079, 43478, 44047, 44627, 44893, 45223, 44769, 44581, 44163, 45314, 45410, 45997, 45919, 46498, 46284, 46520, 47448, 47883, 47324, 46479, 47469, 47996, 47997, 48126, 47029, 49165, 49434, 49366, 50298, 52034, 51522, 52445, 52961, 53700, 53632, 53811, 51976, 54056, 54227, 54232, 54742, 54005, 53248, 52945, 54809, 55736, 54553, 55145, 56362, 56391, 57258, 54977, 56602, 57308, 57524, 57685, 58743, 59160, 59595, 59873, 59571, 60438, 61078, 60555, 60722, 61413, 61980, 62188, 61598, 64393, 64963, 65246, 65179, 64769, 65539, 65471, 65708, 65746, 66106, 65981, 66708, 66853, 67053, 67558, 67542, 67654, 68115, 68104, 68241, 68918, 69044, 68944, 68982, 69168, 69341, 69713, 69234]

maybewrongredshift_cdfs2 = [39266,38350,43478,44163,47324,46479,47469,53248,54553,\
                            54977,57685,61598,65981,38854, 65825, 68675]
# If you correct these then take these ids out of the skip array
# I'm almost sure that all of these are wrong. And in every case the redshift has been overestimated.
dontbelieve_cdfs2 = [38744, 38053, 39177, 38417, 38571, 39235, 40063, 40564, 40163, 38769, 41075, 40488, 41338, 41327, 40875, 42022, 41677, 41994, 43202, 42395, 43170, 43909, 44204, 43803, 44843, 45022, 45502, 45727, 45462, 46988, 46037, 46562, 47990, 48447, 47488, 48934, 48857, 50348, 50261, 51293, 52096, 51533, 52529, 52259, 49771, 51356, 52204, 52774, 53691, 53743, 53588, 54150, 54666, 55759, 54631, 55951, 56575, 54922, 56442, 56565, 57146, 58615, 59905, 59792, 60709, 62122, 62841, 63360, 63098, 62788, 63339, 63612, 63861, 64376, 64503, 64797, 65335, 65620, 65572, 65609, 66149, 66190, 66620, 67929, 67996, 68033, 68414, 68852, 69170, 69260, 69686]


cdfs3_skip = [93128, 93102, 91651, 93627, 92959, 93992, 93808, 94128, 94873, 94900, 94858, 95513, 96579, 96671, 96942, 97327, 96481, 98046, 98242, 98951, 99252, 99381, 99014, 100181, 99156, 99400, 100393, 100652, 100679, 100951, 100543, 101299, 101176, 101634, 101780, 101633, 101795, 102037, 102205, 103116, 103421, 103422, 103683, 103982, 103877, 104478, 104498, 104641, 104868, 105328, 104446, 105400, 105622, 107060, 106026, 105015, 106446, 106641, 104516, 107052, 107055, 107036, 106993, 107680, 107668, 107858, 108274, 107997, 108145, 107754, 107778, 108366, 108336, 108456, 108716, 108561, 108620, 108322, 108779, 109264, 109091, 109422, 109167, 109438, 110198, 110215, 110892, 110712, 111122, 110664, 112386, 112614, 112745, 112747, 113107, 113169, 113558, 113482, 113919, 114244, 113895, 113877, 116185, 115683, 115928, 116423, 115877, 115462, 117560, 117548, 117748, 117770, 118182, 117575, 117222, 117997, 117790, 118014, 118459, 119880, 120439, 119504, 120415, 120644, 120099, 120725, 120950, 120926, 121473, 121302, 120898, 121678, 121911]

maybewrongredshift_cdfs3 = [93128,93102,94128,99014,100543,101634,108322,115462,\
                            94290, 93984, 97487, 99892, 102153, 102847, 106859, 111194, 111700, 115331]
# Again...
# I'm almost sure that all of these are wrong. And in every case the redshift has been overestimated.

dontbelieve_cdfs3 = [91095, 92455, 93923, 93811, 94425, 94854, 95330, 95800, 96378, 96907, 96778, 97043, 96494, 96927, 98198, 98703, 98849, 99129, 99275, 97568, 99598, 99490, 99589, 100332, 100119, 100338, 100526, 101064, 101091, 101093, 102163, 102547, 102708, 102735, 102957, 103107, 102027, 103811, 103969, 104125, 104731, 104622, 104801, 104514, 104408, 104981, 105016, 105522, 105570, 106132, 106073, 106061, 106606, 106953, 107208, 107111, 107670, 107823, 107776, 107863, 108028, 108174, 108874, 109250, 109043, 109396, 109107, 109609, 109596, 109824, 109889, 110235, 109900, 110258, 110962, 111055, 111151, 110733, 110870, 110765, 111806, 112746, 112840, 112989, 113409, 112943, 113730, 113434, 113184, 113815, 114024, 114960, 113555, 115210, 115491, 114998, 115761, 114932, 116364, 116124, 115775, 116174, 116154, 116507, 117225, 117865, 118077, 116284, 118171, 118251, 117066, 118024, 119137, 119164, 119076, 119260, 118925, 119747, 119470, 119278, 119873, 120445, 120466, 120798, 120859, 121574, 121785, 120845]

cdfs4_skip = [109770, 110170, 109872, 110031, 109948, 110400, 111108, 110065, 110950, 111492, 111366, 111622, 111901, 109886, 110795, 113842, 113279, 113685, 114057, 114077, 112617, 114779, 114644, 114978, 116260, 115684, 116031, 117202, 116653, 117002, 116802, 117521, 117714, 117864, 117686, 117648, 117138, 118196, 117429, 118673, 118193, 118772, 119341, 118526, 119722, 120268, 120286, 119893, 119997, 119088, 119702, 120576, 120689, 120229, 120974, 121350, 121149, 121195, 121904, 121733, 121864, 122099, 122206, 121974, 122039, 121837, 122929, 122600, 123390, 122710, 123008, 123168, 122766, 122949, 123802, 123779, 124741, 124152, 125699, 125541, 126193, 125733, 125828, 127195, 126769, 127541, 128062, 128352, 128198, 128379, 127741, 128541, 129605, 130087, 130387, 130552, 130885, 131467, 131381, 131724, 131864, 133306, 133586, 134389]
maybewrongredshift_cdfs4 = [110031,112617,114644,116653,117138,125828,134389,\
                            111741, 115331, 119612, 126790, 133150, 133802]

dontbelieve_cdfs4 = [109511, 109766, 109435, 110187, 109885, 109992, 110891, 110801, 111030, 112095, 111643, 112881, 112960, 112927, 112133, 113548, 113364, 113474, 113434, 113815, 114298, 114115, 114221, 115024, 115210, 115258, 115449, 115589, 115319, 115544, 114932, 115422, 116370, 117116, 117333, 118438, 118434, 118687, 118660, 118448, 119493, 119951, 118455, 119666, 119744, 119193, 119927, 120095, 119489, 119774, 120843, 119960, 121020, 120967, 120840, 120441, 120559, 121011, 121785, 120803, 121506, 121227, 121718, 122404, 122309, 122854, 122421, 122646, 122735, 122913, 122961, 123236, 123533, 123066, 124313, 124266, 124410, 124606, 123477, 124248, 124945, 124964, 126281, 126415, 126934, 126750, 126958, 128313, 128268, 129369, 129968, 129047, 130877, 131112, 131938, 131716, 132474, 132786, 133763, 134044, 134467, 135692]


cdfs_new_skip = [8941, 10593, 11488, 12230, 13011, 13988, 14093, 14386, 14958, 15281, 15336, 15513, 15093, 14990, 15793, 15921, 16183, 16371, 16488, 16683, 17321, 17587, 17494, 17950, 18882, 18972, 19003, 18337, 19569, 19652, 19733, 20072, 20265, 20387, 20694, 20877, 21363, 21614, 21777, 21605, 21730, 22233, 22266, 22153, 22513, 22486, 22897, 23034, 23093, 23136, 23177, 23433, 23511, 23920, 23992, 24360, 24609, 24770, 25164, 25205, 25687, 26450, 26728, 26160, 27183, 27652, 28280, 28000, 29417, 29252, 29733, 29626, 29801, 29726, 29564, 30032, 29922, 30887, 31086, 31227, 31325, 31861, 32847, 30492, 32905, 33074, 32879, 33422, 33294, 33013, 34245, 34620, 35134, 35692, 34803, 35255]

maybewrongredshift_cdfs_new = [14386,25205,31325]

dontbelieve_cdfs_new = [10075, 10429, 10465, 10810, 11099, 12350, 11507, 12457, 12949, 12926, 13732, 14405, 15324, 15868, 16039, 15929, 16788, 17024, 16290, 17163, 17264, 17368, 17413, 17669, 18065, 17999, 16836, 18484, 19435, 19473, 19591, 19683, 19952, 20714, 20681, 21023, 20901, 21196, 21647, 21819, 22217, 22316, 22514, 22563, 22784, 22695, 23059, 23155, 23246, 23413, 23524, 23929, 24367, 24599, 24717, 24671, 24605, 24521, 24919, 24707, 25253, 25462, 25474, 25316, 25860, 26909, 27304, 27562, 27862, 28310, 28588, 29598, 29515, 29630, 30763, 31393, 31772, 31570, 32566, 32224, 32173, 32858, 33247, 32885, 33498, 33949, 34000, 33734, 35068, 35116, 35657]

cdfs_udf_skip = [66708, 67053, 67558, 67542, 68115, 66729, 68241, 68918, 69044, 68982, 68941, 69168, 69117, 69341, 70232, 69472, 69713, 70635, 70340, 70264, 70859, 70973, 71081, 71305, 71890, 72302, 72024, 72853, 74011, 73908, 74166, 74900, 74950, 75397, 74039, 75495, 76706, 76612, 77853, 77668, 78036, 78273, 78527, 77902, 78761, 78771, 78360, 78692, 79154, 80058, 80434, 80618, 81296, 80076, 80543, 81609, 82112, 82307, 82075, 82314, 81384, 82356, 81973, 82239, 83749, 83804, 83735, 83686, 83789, 83834, 84991, 88069, 87658, 87492, 87611, 86719, 87735, 87679, 87783, 88791, 86051, 89122, 90321, 90198, 90809, 91429, 90325, 91382, 92025, 92248, 93198, 92376, 92399, 92860, 93218, 93719, 93759, 94128, 94364, 94873, 94858, 95997]

dontbelieve_cdfs_udf = [65620, 67929, 67599, 67752, 68303, 67996, 68033, 68675, 68852, 69170, 69260, 69686, 70225, 70128, 70141, 70518, 70682, 70821, 71858, 72113, 72048, 71524, 71864, 72319, 72692, 72082, 73166, 73385, 73549, 74234, 73423, 74217, 74352, 74385, 74418, 75733, 75938, 76125, 75971, 76150, 76806, 76154, 76924, 78045, 78417, 77539, 78224, 78343, 77522, 78140, 79058, 79520, 78710, 79891, 77579, 80446, 80297, 79895, 79756, 80892, 80115, 80662, 80500, 81021, 81277, 81636, 81328, 82729, 83293, 85374, 85517, 84197, 84173, 83479, 85692, 85844, 85552, 86109, 85861, 85918, 86604, 86962, 87420, 88542, 89030, 79645, 89063, 90078, 88897, 90246, 90936, 90740, 91584, 91869, 92495, 92468, 91517, 92715, 93563, 93242, 93484, 93849, 93827, 93811, 94200, 94425, 94792, 94632, 95449, 95469, 96013]

maybewrongredshift_cdfs_udf = [66729,68982,70264,74039,75495,77902,78360,83686,88069,87658,86719,90325,94128]

# North

def measure_north(matchedfile, field):
    
    skip = []
    okay = []
    notokay = []
    maybe = []
    
    for i in range(len(matchedfile)):
        cat = matchedfile
        current_id = cat['pearsid'][i]
        """
        if (field == 1) and (current_id in cdfn1_skip): continue
        if (field == 2) and (current_id in cdfn2_skip): continue
        if (field == 3) and (current_id in cdfn3_skip): continue
        if (field == 4) and (current_id in cdfn4_skip): continue
        
        if (field == 1) and (current_id in dontbelieve_cdfn1): continue
        if (field == 2) and (current_id in dontbelieve_cdfn2): continue
        if (field == 3) and (current_id in dontbelieve_cdfn3): continue
        if (field == 4) and (current_id in dontbelieve_cdfn4): continue
        """
        file = data_path + "h_pears_n_id" + str(cat['pearsid'][i]) + ".fits"
        fitsfile = pf.open(file)
        fitsdata = fitsfile[1].data
        flux = fitsdata['FLUX'] - fitsdata['CONTAM']
        flux_err = fitsdata['FERROR']
        lam = fitsdata['LAMBDA']
        flux_err_sqr = flux_err**2

        zp = cat['threed_zphot'][i]
        rest_lam = fitsdata['LAMBDA'] / (1 + zp)

        arg4100 = np.argmin(abs(rest_lam - 4100))
        arg4000 = np.argmin(abs(rest_lam - 4000))
        arg3950 = np.argmin(abs(rest_lam - 3950))
        arg3850 = np.argmin(abs(rest_lam - 3850))

        sum_up = 2 * sum(flux[arg4000+1:arg4100]) + flux[arg4000] + flux[arg4100]
        sum_low = 2 * sum(flux[arg3850+1:arg3950]) + flux[arg3850] + flux[arg3950]
        
        #sum_up_t = np.trapz(flux[arg4000:arg4100+1])
        #sum_low_t = np.trapz(flux[arg3850:arg3950+1])
        sum_up_err = np.sqrt(sum(4 * flux_err_sqr[arg4000+1:arg4100+1]) + flux_err_sqr[arg4000] + flux_err_sqr[arg4100])
        sum_low_err = np.sqrt(sum(4 * flux_err_sqr[arg3850+1:arg3950+1]) + flux_err_sqr[arg3850] + flux_err_sqr[arg3950])
        break_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
        
        
        #if (sum_up/sum_low > 10) or (sum_up/sum_low < 1):
        #    print field, current_id, sum_up/sum_low
            #skip.append(current_id)
            #plot_spectrum(flux[5:100:1], flux_err[5:100:1], lam[5:100:1], zp)
        """
            answer = raw_input('%d ' % current_id)
            if answer == 'y':
                okay.append(current_id)
            elif answer == 'n':
                notokay.append(current_id)
            elif answer == 'maybe':
                maybe.append(current_id)
        """

        d4000 = sum_up/sum_low
        redshift.append(zp)
        d4000_index.append(d4000)
        d4000_err.append(break_err)
        stellar_mass.append(cat['threed_mstellar'][i])
    """
    print skip
    print okay
    print notokay
    print maybe
    plt.close("all")
    """

# South
def measure_south(matchedfile, field):
    
    skip = []
    okay = []
    notokay = []
    maybe = []
    
    for i in range(len(matchedfile)):
        cat = matchedfile
        current_id = cat['pearsid'][i]
        """
        if (field == 5) and (current_id in cdfs1_skip): continue
        if (field == 6) and (current_id in cdfs2_skip): continue
        if (field == 7) and (current_id in cdfs3_skip): continue
        if (field == 8) and (current_id in cdfs4_skip): continue
        if (field == 9) and (current_id in cdfs_new_skip): continue
        if (field == 10) and (current_id in cdfs_udf_skip): continue
        
        if (field == 5) and (current_id in dontbelieve_cdfs1): continue
        if (field == 6) and (current_id in dontbelieve_cdfs2): continue
        if (field == 7) and (current_id in dontbelieve_cdfs3): continue
        if (field == 8) and (current_id in dontbelieve_cdfs4): continue
        if (field == 9) and (current_id in dontbelieve_cdfs_new): continue
        if (field == 10) and (current_id in dontbelieve_cdfs_udf): continue
        
        if (field == 6) and (current_id in maybewrongredshift_cdfs2): continue
        if (field == 7) and (current_id in maybewrongredshift_cdfs3): continue
        if (field == 8) and (current_id in maybewrongredshift_cdfs4): continue
        if (field == 9) and (current_id in maybewrongredshift_cdfs_new): continue
        if (field == 10) and (current_id in maybewrongredshift_cdfs_udf): continue
        """
        file = data_path + "h_pears_s_id" + str(cat['pearsid'][i]) + ".fits"
        fitsfile = pf.open(file)
        fitsdata = fitsfile[1].data
        flux = fitsdata['FLUX'] - fitsdata['CONTAM']
        flux_err = fitsdata['FERROR']
        lam = fitsdata['LAMBDA']
        flux_err_sqr = flux_err**2
    
        zp = cat['threed_zphot'][i]
        rest_lam = fitsdata['LAMBDA'] / (1 + zp)
    
        arg4100 = np.argmin(abs(rest_lam - 4100))
        arg4000 = np.argmin(abs(rest_lam - 4000))
        arg3950 = np.argmin(abs(rest_lam - 3950))
        arg3850 = np.argmin(abs(rest_lam - 3850))
    
        lam_break_obs = 4000 * (1 + zp)
    
        sum_up = 2 * sum(flux[arg4000+1:arg4100]) + flux[arg4000] + flux[arg4100]
        sum_low = 2 * sum(flux[arg3850+1:arg3950]) + flux[arg3850] + flux[arg3950]
        
        #sum_up_t = np.trapz(flux[arg4000:arg4100+1])
        #sum_low_t = np.trapz(flux[arg3850:arg3950+1])
        sum_up_err = np.sqrt(sum(4 * flux_err_sqr[arg4000+1:arg4100+1]) + flux_err_sqr[arg4000] + flux_err_sqr[arg4100])
        sum_low_err = np.sqrt(sum(4 * flux_err_sqr[arg3850+1:arg3950+1]) + flux_err_sqr[arg3850] + flux_err_sqr[arg3950])
        break_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
        
        
        #if (sum_up/sum_low > 10) or (sum_up/sum_low < 1):
        #    print field, current_id, sum_up/sum_low
            #skip.append(current_id)
            #plot_spectrum(flux[5:100:1], flux_err[5:100:1], lam[5:100:1], zp)
        """
            answer = raw_input('%d ' % current_id)
            if answer == 'y':
                okay.append(current_id)
            elif answer == 'n':
                notokay.append(current_id)
            elif answer == 'maybe':
                maybe.append(current_id)
        """
        #print cat['pearsra'][i], cat['pearsdec'][i]
        
        
        d4000 = sum_up/sum_low
        redshift.append(zp)
        d4000_index.append(d4000)
        d4000_err.append(break_err)
        stellar_mass.append(cat['threed_mstellar'][i])

    """
    print skip
    print okay
    print notokay
    print maybe
    plt.close("all")
    """

measure_north(cdfn1, 1)
measure_north(cdfn2, 2)
measure_north(cdfn3, 3)
measure_north(cdfn4, 4)

measure_south(cdfs1, 5)
measure_south(cdfs2, 6)
measure_south(cdfs3, 7)
measure_south(cdfs4, 8)

measure_south(cdfs_new, 9)
measure_south(cdfs_udf, 10)

print len(d4000_index), "galaxies in final sample."

d4000_index = np.array(d4000_index)
d4000_err = np.array(d4000_err)
stellar_mass = np.array(stellar_mass)
redshift = np.array(redshift)

d4000_del_indices = np.where((d4000_index < 0.5) | (d4000_index > 5))

d4000_index = np.delete(d4000_index, d4000_del_indices)
d4000_err = np.delete(d4000_err, d4000_del_indices)
stellar_mass = np.delete(stellar_mass, d4000_del_indices)
redshift = np.delete(redshift, d4000_del_indices)

# stellar mass vs d4000 fit
slope, y_intercept = np.polyfit(stellar_mass, d4000_index, 1, w = 1/d4000_err)
print slope, y_intercept, "are the slope and y-intercept of the best fit line to the D_n(4000) vs stellar mass relation."

xfit = np.arange(7.5, 11.5, 0.1)
yfit = xfit * slope + y_intercept

medianvals = []
medianerr_pos = []
medianerr_neg = []
for i in np.arange(8.75, 11.75, 0.5):
    med_indices = np.where((stellar_mass < i) & (stellar_mass < i + 0.5))
    medianvals.append(np.median(d4000_index[med_indices[0]]))
    medianerr = np.std(d4000_index[med_indices[0]]) # this is not the error on the median !!!! Need to fix this.
    
    medianerr_pos.append(np.median(d4000_index[med_indices[0]]) + medianerr)
    medianerr_neg.append(np.median(d4000_index[med_indices[0]]) - medianerr)

# redshift vs d4000 fit
slope_z, y_intercept_z = np.polyfit(redshift, d4000_index, 1, w = 1/d4000_err)
print slope_z, y_intercept_z, "are the slope and y-intercept of the best fit line to the D_n(4000) vs redshift relation."

xfit_z = np.arange(0.5, 1.3, 0.1)
yfit_z = xfit_z * slope_z + y_intercept_z

# d4000 histogram
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('$\mathrm{D_n}(4000)$')
ax1.set_ylabel('$\mathrm{N}$')

iqr = np.std(d4000_index, dtype=np.float64)
binsize = 2*iqr*np.power(len(d4000_index),-1/3)
totalbins = np.floor((max(d4000_index) - min(d4000_index))/binsize)

ax1.hist(d4000_index, totalbins, alpha=0.5)

ax1.minorticks_on()
ax1.tick_params('both', width=1, length=3, which='minor')
ax1.tick_params('both', width=1, length=4.7, which='major')

fig1.savefig('d4000_hist_pears', dpi=300)

# d4000 vs z
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('$\mathrm{z}$', fontsize=14)
ax2.set_ylabel('$\mathrm{D_n}(4000)$', fontsize=14)

#ax2.errorbar(redshift, d4000_index, yerr=d4000_err, fmt='o', color='k', markersize=3, elinewidth=1, capsize=0)
ax2.plot(redshift, d4000_index, 'o', color='k', markersize=3)
ax2.plot(xfit_z, yfit_z, '--', color='r', linewidth=3)

ax2.minorticks_on()
ax2.tick_params('both', width=1, length=3, which='minor')
ax2.tick_params('both', width=1, length=4.7, which='major')

ax2.axhline(y=1, linestyle='--')
ax2.set_ylim(0.3, 3.0)

#fig2.savefig('d4000_z_pears_fit', dpi=300)
fig2.savefig('d4000_z_pears_fit_noerrbar', dpi=300)

# d4000 vs stellar mass
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$\mathrm{log}\left(\frac{M_*}{M_\odot}\right)$', fontsize=14)
ax3.set_ylabel('$\mathrm{D_n}(4000)$', fontsize=14)

#ax3.errorbar(stellar_mass, d4000_index, yerr=d4000_err, fmt='o', color='k', markersize=3, elinewidth=1, capsize=0)
ax3.plot(stellar_mass, d4000_index, 'o', color='k', markersize=3)
ax3.plot(xfit, yfit, '--', color='r', linewidth=3)

ax3.plot(np.arange(8.75, 11.75, 0.5), medianvals, '-d', color='b', markersize=7, markeredgecolor='b')
ax3.plot(np.arange(8.75, 11.75, 0.5), medianerr_pos, '--d', color='b', markersize=5, markeredgecolor='b')
ax3.plot(np.arange(8.75, 11.75, 0.5), medianerr_neg, '--d', color='b', markersize=5, markeredgecolor='b')

ax3.minorticks_on()
ax3.tick_params('both', width=1, length=3, which='minor')
ax3.tick_params('both', width=1, length=4.7, which='major')

ax3.axhline(y=1, linestyle='--')
ax3.set_ylim(0.3, 3.0)

fig3.savefig('d4000_stellarmass_fit_noerrbar', dpi=300)
#fig3.savefig('d4000_stellarmass_fit', dpi=300)

plt.show()

# No selection; these are all galaxies that matched between 3D-HST and PEARS
"""
    2032 galaxies in final sample.
    0.096180990241 0.0991770475838 are the slope and y-intercept of the best fit line to the D_n(4000) vs stellar mass relation.
    -0.0038010856091 1.0439006481 are the slope and y-intercept of the best fit line to the D_n(4000) vs redshift relation.
"""
# With the narrow definition of the break strength the selection by break will have to be done again
# Changing D(4000) to D_n(4000) now puts points below the flat spectrum line

