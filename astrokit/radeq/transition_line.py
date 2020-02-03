#############################################################
#
#  emission_cii.py defines the 12cii and 13cii emission
#  profile
#
#  CHAOS (Cologne Hacked Astrophysical Software)
#
#  Created by Slawa Kabanovic
#
#############################################################

#############################################################
#
# impot standard python libraries
#
#############################################################

import numpy as np

#############################################################
#
# import chaos libraries
#
#############################################################

from astrokit_py.radeq import const

#############################################################
#
# model constant
#
# in SI units
#
#############################################################

# transition line rest frequency [Hz]

# 12CO (4-3)
freq_0 = 461.0406e9

#############################################################

#############################################################
#
# line profile
#
#############################################################

# line profile of the [12CII] and [13CII] transition lines
def profile(vel, vel_0, width):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # [12CII] line profile
    # line = (np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3))*np.exp(-(vel-vel_0)**2*beta/width**2)
    line = np.exp(-(vel-vel_0)**2*beta/width**2)

    return line

#############################################################
