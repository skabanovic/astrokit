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

from astrokit.num import solver

def tau_ratio_eq(tau, iso_ratio, ratio_norm, obs_ratio):

    return ((1.-np.exp(-tau))/tau - (ratio_norm/iso_ratio)*obs_ratio)

def optical_depth_ratio(iso_ratio, ratio_norm, main_comp, iso_comp,\
error_ratio=0, error_main=0, error_iso=0, method = "symbolic", tau_max = 20):

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
