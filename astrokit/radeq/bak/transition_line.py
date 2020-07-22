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

#from astrokit.radeq import const

#############################################################
#
# model constant
#
# in SI units
#
#############################################################

# transition line rest frequency [Hz]

# 12CO (4-3)
#freq_0 = 461.0406e9

#############################################################

#############################################################
#
# line profiles
#
#############################################################

def gauss_profile(vel, vel_0, width):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # [12CII] line profile
    # line = (np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3))*np.exp(-(vel-vel_0)**2*beta/width**2)
    line = np.exp(-(vel-vel_0)**2*beta/width**2)

    return line


# line profile of the [12CII] and [13CII] transition lines
def profile_cii(vel, vel_0, width):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
    # transition lines with respect to the [12CII] line [km/s]
    vel_13cii_f21 = -11.2
    vel_13cii_f10 =  65.2
    vel_13cii_f11 = -63.3

    # [12CII] line profile
    line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

    # [13CII] F(2-1) line profile
    line_13cii_f21 = (line_f21_norm/abundace_ratio) * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

    # [13CII] F(1-0) line profile
    line_13cii_f10 = (line_f10_norm/abundace_ratio) * \
                     np.exp(-(vel+vel_13cii_f10-vel_0)**2*beta/width**2)

    # [13CII] F(1-1) line profile
    line_13cii_f11 = (line_f11_norm/abundace_ratio) * \
                     np.exp(-(vel+vel_13cii_f11-vel_0)**2*beta/width**2)


    return (line_12cii + line_13cii_f21+ line_13cii_f10 + line_13cii_f11 )

#############################################################
