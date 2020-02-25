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

from astrokit.radeq import const

#############################################################
#
# model constant
#
# in SI units
#
#############################################################

# statistical weight of the level
wei_up = 4.
wei_low = 2.

# Einstein's coefficient [1/s]
einstein_coef = 2.29e-6

# [12CII] rest frequency of [Hz]
freq_0 = 1900.5369e9

# abundance ratio between [12CII]/[13CII] in Orion A
abundace_ratio = 67.
# abundace_ratio = 65

# relative line strength of the 13CII satellites
line_f21_norm = 0.625
line_f10_norm = 0.250
line_f11_norm = 0.125

#############################################################

#############################################################
#
# line profile
#
#############################################################

# line profile of the [12CII] and [13CII] transition lines
def profile_norm(vel, vel_0, width):

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

    return (np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)) * (line_12cii + line_13cii_f21+ line_13cii_f10 + line_13cii_f11 )

#############################################################

# line profile of the [12CII] and [13CII] transition lines
def profile(vel, vel_0, width):

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

#############################################################

# line profile of the [12CII] and [13CII] transition lines
def spectrum(vel, amp, vel_0, width):

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


    return amp*(line_12cii + line_13cii_f21+ line_13cii_f10 + line_13cii_f11 )


#############################################################

#############################################################

# line profile of the [12CII] and [13CII] transition lines
def spectrum_wing(vel, amp, vel_0, width):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
    # transition lines with respect to the [12CII] line [km/s]
    vel_13cii_f21 = -11.2

    # [12CII] line profile
    line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

    # [13CII] F(2-1) line profile
    line_13cii_f21 = (line_f21_norm/abundace_ratio) * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

    return amp*(line_12cii + line_13cii_f21)

#############################################################

#############################################################

# line profile of the [12CII] and [13CII] transition lines
def hidden_wing(vel, amp, vel_0, width, hidden_amp, hidden_vel_0, hidden_width):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
    # transition lines with respect to the [12CII] line [km/s]
    vel_13cii_f21 = -11.2

    # [12CII] line profile
    line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

    # [13CII] F(2-1) line profile
    line_13cii_f21 = (line_f21_norm/abundace_ratio) * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

    line_hidden = np.exp(-(vel-hidden_vel_0)**2*beta/hidden_width**2)

    return amp*(line_12cii + line_13cii_f21) + hidden_amp*line_hidden

#############################################################

#############################################################

# line profile of the [12CII] and [13CII] transition lines
def spectrum_wing_multi(vel, *param):

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
    # transition lines with respect to the [12CII] line [km/s]
    vel_13cii_f21 = -11.2

    spect_sum = np.zeros_like(vel)

    for comp in range(0, len(param), 3):

        amp   = param[comp]
        vel_0 = param[comp+1]
        width = param[comp+2]

        # [12CII] line profile
        line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

        # [13CII] F(2-1) line profile
        line_13cii_f21 = (line_f21_norm/abundace_ratio) * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

        spect_sum = spect_sum + amp*(line_12cii + line_13cii_f21)

    return spect_sum

#############################################################
