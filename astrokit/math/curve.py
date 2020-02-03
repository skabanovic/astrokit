#############################################################
#
#  curve.py provides multiple tools to analyse
#  spectral data
#
#  Created by Slawa Kabanovic
#
#############################################################

import numpy as np
import astropy
import astropy.units as u

def gauss(ax, amp, ax_0, width):
    beta = 4.*np.log(2.)
    return amp*np.exp(-(ax-ax_0)**2*beta/(width**2))

def gauss_multi(ax, *param):

    gauss_sum = np.zeros_like(ax)

    beta = 4.*np.log(2.)

    for comp in range(0, len(param), 3):

        amp   = param[comp]
        ax_0  = param[comp+1]
        width = param[comp+2]

        gauss_sum = gauss_sum + amp*np.exp(-(ax-ax_0)**2*beta/(width**2))

    return gauss_sum

def exp_func(ax, amp, var, off):
    return amp*np.exp(-var*ax)+off

def wing_fit(ax, amp, ax_0, width, freq_offset, amp_scaling):

    wing_profile = gauss(ax, amp*amp_scaling, ax_0-freq_offset, width)+gauss(ax, amp, ax_0, width)

    return wing_profile

def heaviside(ax, ax_0, version = "normal"):

    if version == "normal":

        return 1. * (ax > ax_0)

    elif version == "inverse":
        return 1. * (ax <= ax_0)

    else:
        print("error: allowed verions are normal or inverse")

def gauss_asymmetric(ax, amp, ax_0, width, width_asym):

    asymmetric_profile = gauss(ax, amp, ax_0, width) * heaviside(ax, ax_0, version = "normal")\
        + gauss(ax, amp, ax_0, width*width_asym)  * heaviside(ax, ax_0, version = "inverse")

    return asymmetric_profile

def gauss_2d(beam, ax1_0, ax2_0, ax1, ax2):

    pow2beam=(beam/(2.*np.sqrt(2.*np.log(2.))))*(beam/(2.*np.sqrt(2.*np.log(2.))))*2.
    pow2ax1=(ax1-ax1_0)*(ax1-ax1_0)
    pow2ax2=(ax2-ax2_0)*(ax2-ax2_0)

    return ((1./(pow2beam*np.pi))*np.exp(-pow2ax1/pow2beam - pow2ax2/pow2beam))
