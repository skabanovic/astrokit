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

from astropy import constants as const

def line_profile(vel = None,
                 amp = None,
                 vel_0 = None,
                 width = None,
                 abundace_ratio = 67.,
                 profile = 'gauss',
                 line = 'cii',
                 norm = False):

    if line == 'cii':

        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

    elif line == '12co(3-2)':

        # 12CO (3-2) rest frequency [Hz]

        freq_0 = 345.79598990e9

    elif line == '12co(4-3)':

        # 12CO (4-3) rest frequency [Hz]
        freq_0 = 461.0406e9

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    if profile == 'gauss':

        line = amp*np.exp(-(vel-vel_0)**2*beta/width**2)

    elif profile == 'cii' or profile == 'cii_wing':

        # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
        # transition lines with respect to the [12CII] line [km/s]
        vel_13cii_f21 = -11.2
        vel_13cii_f10 =  65.2
        vel_13cii_f11 = -63.3

        # relative line strength of the 13CII satellites
        line_f21_norm = 0.625
        line_f10_norm = 0.250
        line_f11_norm = 0.125

        # [12CII] line profile
        line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

        # [13CII] F(2-1) line profile
        line_13cii_f21 = (line_f21_norm/abundace_ratio)\
                       * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

        # [13CII] F(1-0) line profile
        line_13cii_f10 = (line_f10_norm/abundace_ratio)\
                       * np.exp(-(vel+vel_13cii_f10-vel_0)**2*beta/width**2)

        # [13CII] F(1-1) line profile
        line_13cii_f11 = (line_f11_norm/abundace_ratio)\
                       * np.exp(-(vel+vel_13cii_f11-vel_0)**2*beta/width**2)

        if norm:

            line_norm = np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)

            line = line_norm\
                 * (line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)
        else:

            line = amp*(line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)

    return line

def multi_line_profile(vel,
                       *param,
                       abundace_ratio,
                       profile,
                       line,
                       norm):

    if line == 'cii':

        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

    elif line == '12co4-3':

        # 12CO (4-3) rest frequency [Hz]
        freq_0 = 461.0406e9

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    line_sum = np.zeros_like(vel)

    for comp in range(0, len(param), 3):

        amp   = param[comp]
        vel_0 = param[comp+1]
        width = param[comp+2]

        if profile == 'gauss':

            line = amp*np.exp(-(vel-vel_0)**2*beta/width**2)

        elif profile == 'cii' or profile == 'cii_wing':

            # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
            # transition lines with respect to the [12CII] line [km/s]
            vel_13cii_f21 = -11.2
            vel_13cii_f10 =  65.2
            vel_13cii_f11 = -63.3

            # relative line strength of the 13CII satellites
            line_f21_norm = 0.625
            line_f10_norm = 0.250
            line_f11_norm = 0.125

            # [12CII] line profile
            line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

            # [13CII] F(2-1) line profile
            line_13cii_f21 = (line_f21_norm/abundace_ratio)\
                           * np.exp(-(vel+vel_13cii_f21-vel_0)**2*beta/width**2)

            # [13CII] F(1-0) line profile
            line_13cii_f10 = (line_f10_norm/abundace_ratio)\
                           * np.exp(-(vel+vel_13cii_f10-vel_0)**2*beta/width**2)

            # [13CII] F(1-1) line profile
            line_13cii_f11 = (line_f11_norm/abundace_ratio)\
                           * np.exp(-(vel+vel_13cii_f11-vel_0)**2*beta/width**2)

            if norm:

                line_norm = np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)

                line = line_norm*amp\
                     * (line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)
            else:

                line = amp*(line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)

            line_sum = line_sum + line

    return line_sum

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

def optical_depth(vel,
                  vel_0,
                  width,
                  temp_ex,
                  colden,
                  abundace_ratio = 67.,
                  profile = 'cii',
                  line = 'cii'):

    if line == 'cii':
        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

        # statistical weight of the level
        g_u = 4.
        g_l = 2.

        # Einstein's coefficient [1/s]
        A_ul = 2.29e-6

        temp_0 = (const.h.value * freq_0)/const.k_B.value

        amp = ((g_u * A_ul * const.c.value**3)/(g_l * 8. * np.pi * freq_0**3))\
            * colden * 1.0e4 * (1. - np.exp(-temp_0/temp_ex))\
            / (1. + (g_u/g_l) * np.exp(-temp_0/temp_ex))

        tau = line_profile(vel,
                           amp,
                           vel_0,
                           width,
                           abundace_ratio,
                           profile,
                           line,
                           norm = True)


    return tau

def column_density(vel_0,
                   width,
                   tau_0,
                   temp_ex,
                   abundace_ratio = 67.,
                   profile = 'cii',
                   line = 'cii'):


    if line == 'cii':
        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

        # statistical weight of the level
        g_u = 4.
        g_l = 2.

        # Einstein's coefficient [1/s]
        A_ul = 2.29e-6

        temp_0 = (const.h.value * freq_0)/const.k_B.value

        amp=1.0

        colden = tau_0/line_profile(vel_0,
                                    amp,
                                    vel_0,
                                    width,
                                    abundace_ratio,
                                    profile,
                                    line,
                                    norm = True)\
               * (g_l*8.*np.pi*freq_0**3)/(g_u*A_ul*const.c.value**3)\
               * (1. + (g_u/g_l) * np.exp(-temp_0/temp_ex))\
               / (1. - np.exp(-temp_0/temp_ex))

    return colden

#############################################################
#
# determine: brightness temperatur for each component
#
# the fuction takes as an input:
#
# temp_ex = exitation temperatur of each component [K]
#
#############################################################

def brightness_temperatur(temp_ex, line = 'cii'):

    if line=='cii':

        freq_0 = 1900.5369e9

    if line=='12co(3-2)':
        
        freq_0 = 345.79598990e9


    temp_0 = (const.h.value * freq_0)/const.k_B.value

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

def excitation_temperatur(temp_mb, line = 'cii'):

    if line=='cii':

        freq_0 = 1900.5369e9

    temp_0 = (const.h.value * freq_0)/const.k_B.value

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

def two_layer(bakg_num,
              forg_num,
              vel,
              *param,
              profile,
              line,
              abundace_ratio):

    forg_sum = np.zeros_like(vel)
    bakg_sum = np.zeros_like(vel)
    forg_tau = np.zeros_like(vel)


    if (forg_num > 0):

        for i in range(0, (4*bakg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = line_profile(vel,
                               tau_0,
                               vel_0,
                               width,
                               abundace_ratio,
                               profile,
                               line,
                               norm = False)

            bakg_sum += brightness_temperatur(temp_ex, line)*(1. - np.exp(-tau))

        for i in range((4*bakg_num), (4*bakg_num+4*forg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = line_profile(vel,
                               tau_0,
                               vel_0,
                               width,
                               abundace_ratio,
                               profile,
                               line,
                               norm = False)

            forg_tau +=  tau

            forg_sum += brightness_temperatur(temp_ex, line)*(1.-np.exp(-tau))

        return (forg_sum + bakg_sum * np.exp(-forg_tau))

    else:
        for i in range(0, (4*bakg_num), 4):

            temp_ex  = param[i]
            tau_0    = param[i+1]
            vel_0    = param[i+2]
            width    = param[i+3]

            tau = line_profile(vel,
                               tau_0,
                               vel_0,
                               width,
                               abundace_ratio,
                               profile,
                               line,
                               norm = False)

            bakg_sum += brightness_temperatur(temp_ex, line)*(1. - np.exp(-tau))

        return bakg_sum

def temp_mb_error(temp_ex,
                  tau,
                  temp_ex_err,
                  tau_err,
                  line = 'cii'):

    if line=='cii':

        freq_0 = 1900.5369e9

    temp_0 = (const.h.value * freq_0)/const.k_B.value

    term_temp = (temp_0/temp_ex)**2\
              * np.exp(temp_0/temp_ex) * (1.-np.exp(-tau))\
              / (np.exp(temp_0/temp_ex)-1.)**2

    term_tau = temp_0*np.exp(-tau)/(np.exp(temp_0/temp_ex)-1.)

    temp_mp_err = np.sqrt( (term_temp*temp_ex_err)**2
                          +(term_tau*tau_err)**2 )

    return temp_mp_err
