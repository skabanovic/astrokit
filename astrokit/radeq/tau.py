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
import astropy.units as u
from astropy import constants as const

import astrokit
from astrokit.num import solver

from astropy.coordinates import Angle


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

def optical_depth_ratio(iso_ratio,
                        ratio_norm,
                        main_comp,
                        iso_comp,
                        error_ratio=0,
                        error_main=0,
                        error_iso=0,
                        method = "symbolic",
                        tau_max = 30):

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


def tau_spect(spect,
              vel,
              vel_min = None,
              vel_max = None,
              iso_shift = 0.,
              iso_ratio = 67.,
              iso_norm = 1.,
              spect_rms = 0.,
              error_ratio = 0.,
              spect_iso = [],
              vel_iso = [],
              use_axis = 'none'):


    if len(spect_iso) == 0:
        spect_iso = spect

    if len(vel_iso) == 0:
        vel_iso = vel

    if vel_min and vel_max:

        idx_min = astrokit.get_idx(vel_min, vel, method = 'closer')
        idx_max = astrokit.get_idx(vel_max, vel, method = 'closer')

        vel_shift = np.zeros_like(vel[idx_min:idx_max])

        tau = np.zeros_like(vel[idx_min:idx_max])
        error_tau = np.zeros_like(vel[idx_min:idx_max])

        vel_shift = vel[idx_min:idx_max] + iso_shift

    elif use_axis == 'main':

        idx_min = 0

        vel_shift = np.zeros_like(vel)

        tau = np.zeros_like(vel)
        error_tau = np.zeros_like(vel)

        vel_shift = vel + iso_shift

    elif use_axis == 'iso':

        vel_shift = np.zeros_like(vel_iso)

        tau = np.zeros_like(vel_iso)
        error_tau = np.zeros_like(vel_iso)

        vel_shift = vel_iso + iso_shift

    for vel_idx in range(len(vel_shift)):

        if use_axis == 'iso':

            Tmb = astrokit.get_value(vel_shift[vel_idx], vel, spect)

        else:

            Tmb = astrokit.get_value(vel_shift[vel_idx], vel_iso, spect_iso)

        if (Tmb < spect_rms):

            if use_axis == 'iso':

                tau[vel_idx], error_tau[vel_idx] = \
                astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                             Tmb, spect_rms,
                                             error_ratio = error_ratio,
                                             error_main= spect_rms,
                                             error_iso = spect_rms,
                                             method = "bisection")



            else:

                tau[vel_idx], error_tau[vel_idx] = \
                astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                             spect[idx_min+vel_idx], spect_rms,
                                             error_ratio = error_ratio,
                                             error_main= spect_rms,
                                             error_iso = spect_rms,
                                             method = "bisection")

        else:

            if use_axis == 'iso':

                tau[vel_idx], error_tau[vel_idx] = \
                astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                             Tmb, spect_iso[vel_idx],
                                             error_ratio = error_ratio,
                                             error_main= spect_rms,
                                             error_iso = spect_rms,
                                             method = "bisection")

            else:


                tau[vel_idx], error_tau[vel_idx] = \
                astrokit.optical_depth_ratio(iso_ratio, iso_norm,
                                             spect[idx_min+vel_idx], Tmb,
                                             error_ratio = error_ratio,
                                             error_main= spect_rms,
                                             error_iso = spect_rms,
                                             method = "bisection")


    return tau, error_tau


def hyperfine_avarage(spect, vel,
                      vel_range, vel_shift, weight):

    num_of_hyper = len(vel_shift)

    Tmb_temp = np.zeros(num_of_hyper)

    idx_min = astrokit.get_idx(vel_range[0], vel)
    idx_max = astrokit.get_idx(vel_range[1], vel)

    vel_hyper = np.zeros_like(vel[idx_min:idx_max])
    Tmb_hyper = np.zeros_like(vel[idx_min:idx_max])

    vel_hyper = vel[idx_min:idx_max]

    for vel_idx in range(len(vel_hyper)):

        for hyper_line in range(num_of_hyper):

            Tmb_temp[hyper_line] = astrokit.get_value(vel[idx_min+vel_idx]+vel_shift[hyper_line], vel, spect)\
                                 * weight[hyper_line]

        Tmb_hyper[vel_idx] = sum(Tmb_temp)/sum(weight**2)


    return Tmb_hyper, vel_hyper


def inten_to_colden(inten, Tex=None, inten_err = 0, line = 'cii'):

    if line == 'cii':

        # cii rest frequency in [Hz]
        freq_cii = 1900.5369e9

        # Einstein's coefficient [1/s]
        einstein_coef = 2.29e-6

       # equivalent temperature of the excited level
        T_0 = (const.h.value*freq_cii)/const.k_B.value

        if Tex:

            const_term = (1+(2./4.)*np.exp(T_0/Tex))* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)

            colden = const_term*inten


        else:

            const_term = (3./2.)* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)

            colden = const_term*inten

        colden_err = const_term*inten_err

    return colden, colden_err

def colden_to_mass(colden,
                   area,
                   colden_err=0,
                   area_err=None,
                   line = 'hi',
                   abundance_ratio = 1.2e-4):


    if line == 'hi':

        # hydrogen mass
        element_mass = 1.6735575e-27

    elif line == 'cii':

        # carbon mass
        element_mass = 1.9944235e-26

    # convert size of the area to m
    area = area.to(u.meter**2)
    if area_err:
        area_err = area_err.to(u.meter**2)
    else:
        area_err = (0.0*u.pc**2).to(u.meter**2)

    mass = (area.value * colden * element_mass)/abundance_ratio

    mass_err = element_mass/abundance_ratio\
             * np.sqrt(area_err.value**2*colden**2
                       + area.value**2*colden_err**2)

    return mass, mass_err

def inten_to_flux(inten, line, inten_err=0):

    if line == 'cii':
        # cii rest frequency in [Hz]
        freq = 1900.5369e9

    const_flux = 2.*const.k_B.value*(freq**3/const.c.value**3)

    flux = inten*const_flux
    flux_err = inten_err*const_flux

    return flux, flux_err

def line_luminosity(inten, line, dist = None, area = None,
                    dist_err=0, area_err=None, inten_err=0):

        flux, flux_err = inten_to_flux(inten, line, inten_err )

        if dist:
            # if intput id flux
            lum = flux * 4.*np.pi*dist**2

            lum_err = 4.*np.pi*np.sqrt( (flux_err*dist**2)**2
                                       +(dist_err*flux*2.*dist)**2)

        elif area:

            area=area.to(u.meter**2)

            if area_err:
                area_err = area_err.to(u.meter**2)
            else:
                area_err = (0.0*u.pc**2).to(u.meter**2)

            # if input is surface brightness
            lum = flux * 4.*np.pi*area.value

            lum_err = 4.*np.pi*np.sqrt( (area.value*flux_err)**2
                                       +(flux*area_err.value)**2)

        return lum, lum_err


def colden_map(inten,
               inten_err = None,
               Tex = None,
               line = 'cii'):


    colden     = astrokit.zeros_map(inten)
    colden_err = astrokit.zeros_map(inten)

    if line == 'cii':

        # cii rest frequency in [Hz]
        freq_cii = 1900.5369e9

        # Einstein's coefficient [1/s]
        einstein_coef = 2.29e-6

       # equivalent temperature of the excited level
        T_0 = (const.h.value*freq_cii)/const.k_B.value

        if Tex:

            const_term = (1+(2./4.)*np.exp(T_0/Tex))* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)

            colden[0].data = const_term*inten[0].data


        else:

            const_term = (3./2.)* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)

            colden[0].data = const_term*inten[0].data

        if inten_err:
            colden_err[0].data = const_term*inten_err[0].data
        else:
            colden_err[0].data = np.nan

    return colden, colden_err


def mass_map(colden,
             colden_err = None,
             abundance_ratio = 1.2e-4,
             dist = 0,
             dist_err = 0,
             shape=None,
             pos = None,
             size_1 = None,
             size_2 = None):

    res_ra = abs(colden[0].header["CDELT1"])
    res_dec = abs(colden[0].header["CDELT2"])

    pixel_size = dist**2 * Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad

    pixel_size_err = 2. * dist * dist_err * Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad

    hydro_mass = 1.6735575e-27

    mass = astrokit.zeros_map(colden)
    mass_err = astrokit.zeros_map(colden)

    grid2d_ra, grid2d_dec = astrokit.get_grid(colden)

    if shape == "circle" or shape == "sphere":

        # determie the distance from the line points to every grid point
        if shape == "circle":

            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2 + (grid2d_ra-pos[0])**2)

        elif shape == "sphere":

            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2 + ((grid2d_ra-pos[0])*np.cos(Angle(pos[1]*u.deg)))**2)


        if size_2:

            grid2d_sig3=grid2d_r[np.where( np.logical_and(grid2d_r>=size_1, grid2d_r<=size_2))]

        else:
            # find relavant coordinat values
            grid2d_sig3=grid2d_r[np.where( grid2d_r <= size_1 )]

        bool_sig3 = np.isin(grid2d_r, grid2d_sig3)
        idx_sig3=np.asarray(np.where(bool_sig3))

        for idx_mask in range(len(idx_sig3[0][:])):


            ####################################################################
            #
            #pixel_size = dist**2\
            #           * (Angle( abs((grid2d_dec[idx_sig3[0][idx_mask]+1, idx_sig3[1][idx_mask]+1]-
            #                         grid2d_dec[idx_sig3[0][idx_mask]-1, idx_sig3[1][idx_mask]-1])/2.)*u.deg).rad*
            #             Angle( abs((grid2d_ra[idx_sig3[0][idx_mask]+1, idx_sig3[1][idx_mask]+1]-
            #                         grid2d_ra[idx_sig3[0][idx_mask]-1, idx_sig3[1][idx_mask]-1])/2.)*u.deg).rad*
            #             np.cos( Angle(grid2d_dec[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]*u.deg).rad ))
            #
            #
            #pixel_size_err = 2.*dist*dist_err\
            #               * (Angle( abs((grid2d_dec[idx_sig3[0][idx_mask]+1, idx_sig3[1][idx_mask]+1]-
            #                             grid2d_dec[idx_sig3[0][idx_mask]-1, idx_sig3[1][idx_mask]-1])/2.)*u.deg).rad*
            #                 Angle( abs((grid2d_ra[idx_sig3[0][idx_mask]+1, idx_sig3[1][idx_mask]+1]-
            #                             grid2d_ra[idx_sig3[0][idx_mask]-1, idx_sig3[1][idx_mask]-1])/2.)*u.deg).rad*
            #                 np.cos( Angle(grid2d_dec[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]*u.deg).rad ))
            #
            ####################################################################

            mass[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]] = (pixel_size*colden[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]*hydro_mass)/abundance_ratio

            mass_err[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]] = hydro_mass/abundance_ratio\
                                                                           * np.sqrt(  pixel_size_err**2*colden[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]**2
                                                                           + pixel_size**2*colden_err[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]**2)


    else:

        for idx_dec in range(len(mass[0].data[:,0])):
            for idx_ra in range(len(mass[0].data[0,:])):

                if colden[0].data[idx_dec, idx_ra]>0:

                    ###########################################################
                    #
                    #pix_size = dist**2\
                    #         * (Angle(abs((grid2d_dec[idx_dec+1, idx_ra+1]-
                    #                       grid2d_dec[idx_dec-1, idx_ra-1])/2.)*u.deg).rad*
                    #            Angle(abs((grid2d_ra[idx_dec+1, idx_ra+1]-
                    #                       grid2d_ra[idx_dec-1, idx_ra-1])/2.)*u.deg).rad*
                    #            np.cos( Angle(grid2d_dec[idx_dec, idx_ra]*u.deg).rad ))
                    #
                    #pix_size_err = 2.*dist*dist_err\
                    #             * (Angle(abs((grid2d_dec[idx_dec+1, idx_ra+1]-
                    #                           grid2d_dec[idx_dec-1, idx_ra-1])/2.)*u.deg).rad*
                    #                Angle(abs((grid2d_ra[idx_dec+1, idx_ra+1]-
                    #                           grid2d_ra[idx_dec-1, idx_ra-1])/2.)*u.deg).rad*
                    #                np.cos( Angle(grid2d_dec[idx_dec, idx_ra]*u.deg).rad ))
                    #
                    ############################################################

                    mass[0].data[idx_dec, idx_ra] = (pixel_size*colden[0].data[idx_dec, idx_ra]*hydro_mass)/abundance_ratio

                    mass_err[0].data[idx_dec, idx_ra] = hydro_mass/abundance_ratio\
                                                      * np.sqrt(pixel_size_err**2*colden[0].data[idx_dec, idx_ra]**2
                                                      + pixel_size**2*colden_err[0].data[idx_dec, idx_ra]**2)

    return mass, mass_err


def optical_depth_map(main_map,
                      iso_map,
                      main_err_map,
                      iso_err_map,
                      iso_ratio,
                      ratio_norm,
                      error_ratio = 0,
                      method = 'bisection'):

    tau_map = astrokit.zeros_map(main_map)
    tau_err_map = astrokit.zeros_map(main_map)

    len_ax1 = main_map[0].header['NAXIS1']
    len_ax2 = main_map[0].header['NAXIS2']

    for idx_ax1 in range(len_ax1):
        for idx_ax2 in range(len_ax2):

            check_1 = (iso_map[0].data[idx_ax2, idx_ax1]*(iso_ratio/ratio_norm)) - main_map[0].data[idx_ax2, idx_ax1]
            check_2 = iso_map[0].data[idx_ax2, idx_ax1] - (3.*iso_err_map[0].data[idx_ax2, idx_ax1])

            if ((check_1 > 0.) and (check_2 > 0.)):

                tau_map[0].data[idx_ax2, idx_ax1], tau_err_map[0].data[idx_ax2, idx_ax1] =\
                astrokit.optical_depth_ratio(iso_ratio,
                                             ratio_norm,
                                             main_map[0].data[idx_ax2, idx_ax1],
                                             iso_map[0].data[idx_ax2, idx_ax1],
                                             error_ratio = error_ratio,
                                             error_main = main_err_map[0].data[idx_ax2, idx_ax1],
                                             error_iso = iso_err_map[0].data[idx_ax2, idx_ax1],
                                             method = method,
                                             tau_max = 100)

            else:

                tau_map[0].data[idx_ax2, idx_ax1] = 0
                tau_err_map[0].data[idx_ax2, idx_ax1] = 0


    return tau_map, tau_err_map
