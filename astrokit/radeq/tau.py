#############################################################
#
#  tau.py
#
#  Created by Slawa Kabanovic
#
#############################################################

from sympy.solvers import solve
from sympy import Symbol, exp
import numpy as np

import astrokit
from astrokit.num import solver

def galactic_carbon_ratio(dist, dist_err):

    ratio_const1 = 6.21

    ratio_const2 = 18.71

    carbon_ratio = ratio_const1 * dist * 1e-3 + ratio_const2

    ratio_const2_err = 7.37

    carbon_ratio_err = np.sqrt((dist * 1e-3)**2 \
                               + ratio_const1**2*(dist_err * 1e-3)**2\
                               + ratio_const2_err**2)
    return carbon_ratio, carbon_ratio_err

def tau_ratio_eq(tau, iso_ratio, ratio_norm, obs_ratio):

    return ((1.-np.exp(-tau))/tau - (ratio_norm/iso_ratio)*obs_ratio)

def optical_depth_ratio(iso_ratio, ratio_norm, main_comp, iso_comp,\
error_ratio=0, error_main=0, error_iso=0, method = "symbolic", tau_max = 30):

    obs_ratio=main_comp/iso_comp

    if method == "symbolic":

        optical_depth = 0.

        tau = Symbol('tau')
        optical_depth_arr = solve((1.-exp(-tau))/tau - (ratio_norm/iso_ratio)*obs_ratio, tau)

        #print(optical_depth_arr)
        if not optical_depth_arr:

            optical_depth = 0.

        else:

            optical_depth = float(optical_depth_arr[0])

    elif method == "bisection":

        optical_depth =\
        solver.bisection(lambda tau : tau_ratio_eq(tau, iso_ratio=iso_ratio, ratio_norm=ratio_norm, obs_ratio=obs_ratio),\
        pos_1=1e-6, pos_2=tau_max, resolution = 1e-6, max_step = 1e6)

        if not optical_depth:

            optical_depth=0

    else:

        print("error: the chosen method is not valid. chose between sybolic, numeric")

    #error = 0
    if optical_depth == 0:

        error_tau = 0.

    else:

        #print("optical_depth: "+ str(optical_depth))

        error_tau = abs(optical_depth**2 /((1. + optical_depth ) * np.exp(-optical_depth) -1. ))\
        * (ratio_norm/iso_ratio)\
        * np.sqrt( (error_main/iso_comp)**2 \
        + (main_comp*error_iso/iso_comp**2)**2 \
        + (main_comp*error_ratio/(iso_comp*iso_ratio))**2 )

    return optical_depth, error_tau

def tau_spect(spect, vel,
              vel_min, vel_max,
              iso_shift, iso_ratio, iso_norm,
              spect_rms, error_ratio):

    idx_min = astrokit.get_idx(vel_min, vel)
    idx_max = astrokit.get_idx(vel_max, vel)

    vel_iso   = np.zeros_like(vel[idx_min:idx_max])
    Tmb_iso  =  np.zeros_like(vel[idx_min:idx_max])

    tau       = np.zeros_like(vel[idx_min:idx_max])
    error_tau = np.zeros_like(vel[idx_min:idx_max])

    vel_iso = vel[idx_min:idx_max] + iso_shift

    for vel_idx in range(len(vel_iso)):

        Tmb_iso[vel_idx] = astrokit.get_value(vel_iso[vel_idx], vel, spect)

        if (Tmb_iso[vel_idx] < spect_rms):

            tau[vel_idx], error_tau[vel_idx] = \
            astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                         spect[idx_min+vel_idx], spect_rms,
                                         error_ratio = error_ratio,
                                         error_main= spect_rms,
                                         error_iso = spect_rms,
                                         method = "bisection")

        else:

            tau[vel_idx], error_tau[vel_idx] = \
            astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                         spect[idx_min+vel_idx], Tmb_iso[vel_idx],
                                         error_ratio = error_ratio,
                                         error_main= spect_rms,
                                         error_iso = spect_rms,
                                         method = "bisection")


    return tau, error_tau
