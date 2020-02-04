#############################################################
#
#  radeq.py includes the two layer multi component
#  radiative transfer equations.
#
#  CHAOS (Cologne Hacked Astrophysical Software)
#
#  Created by Slawa Kabanovic
#
#############################################################

import numpy as np

#############################################################
#
# import chaos libraries
#
#############################################################

# import line profile library
from astrokit.radeq import transition_line_cii as line

# imposrt physical constants
from astrokit.radeq import const


#############################################################
#
# determine: optical depth for each component
#
# the fuction takes as an input:
#
# vel = velocity axis of the spectrum [km/s]
# vel_0 = mean velocity of each componet [km/s]
# width = width of each component [km/s]
# temp_ex = exitation temperatur of each component [K]
# col_den = column density of each component [1/cm^2]
#
#############################################################

def optical_depth(vel, vel_0, width, tau_0):

    tau = line.profile(vel, vel_0, width) * tau_0

    return tau

#############################################################
#
# determine: brightness temperatur for each component
#
# the fuction takes as an input:
#
# temp_ex = exitation temperatur of each component [K]
#
#############################################################

def brightness_temperatur(temp_ex):

    temp_0 = (const.plank * line.freq_0)/const.boltzmann

    return (temp_0/(np.exp(temp_0/temp_ex)-1.))

#############################################################
#
# determine: excitation temperatur for each componet
#
# the fuction takes as an input:
#
# temp_mb = main beam temperatur of each component [K]
#
#############################################################

def excitation_temperatur(temp_mb):

    temp_0 = (const.plank * line.freq_0)/const.boltzmann

    return (temp_0/(np.log((temp_0/temp_mb) + 1.)))

#############################################################
#
# determine: main beam temperatur for each componet
#
# the fuction takes as an input:
#
# forg_num = number of foreground components
# bakg_num = number of background components
# vel = velocity axis of the spectrum
# params[i, i+4, ..] = exitation temperatur of every component [K]
# params[i+1, i+5, ..] = column density of every component [1/cm^2]
# params[i+2, i+6, ..] = mean velocity of every component [km/s]
# params[i+3, i+7, ..] = width of every component [km/s]
#
#############################################################

def two_layer(bakg_num, forg_num, vel, *param):

    forg_sum = np.zeros_like(vel)
    bakg_sum = np.zeros_like(vel)
    forg_tau = np.zeros_like(vel)


    if (forg_num > 0):

        for i in range(0, (4*bakg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = optical_depth(vel, vel_0, width, tau_0)

            bakg_sum += brightness_temperatur(temp_ex)*(1. - np.exp(-tau))

        for i in range((4*bakg_num), (4*bakg_num+4*forg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = optical_depth(vel, vel_0, width, tau_0)

            forg_tau +=  tau

            forg_sum += brightness_temperatur(temp_ex)*(1.-np.exp(-tau))

        return (forg_sum + bakg_sum * np.exp(-forg_tau))

    else:
        for i in range(0, (4*bakg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = optical_depth(vel, vel_0, width, tau_0)

            bakg_sum += brightness_temperatur(temp_ex)*(1. - np.exp(-tau))

        return bakg_sum
