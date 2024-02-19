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
import copy

import astrokit
from astrokit.num import solver

from astropy.coordinates import Angle

from scipy.optimize import curve_fit
from IPython.display import clear_output

from scipy.optimize import brentq

from joblib import Parallel, delayed

def black_body(
    input_quantity,
    temp,
    input_unit='frequency',
    system_of_units = 'SI',
    input_scale = 'linear'

):

    if input_scale == 'log10':

        input_quantity = 10**input_quantity

    if system_of_units == 'SI':

        plank_const = const.h.si.value
        light_speed = const.c.si.value
        boltzmann_const = const.k_B.si.value

    elif system_of_units == 'cgs':

        plank_const = const.h.cgs.value
        light_speed = const.c.cgs.value
        boltzmann_const = const.k_B.cgs.value

    if input_unit == 'frequency':

         spectral_radiance = 2.*plank_const*input_quantity**3/light_speed**2/(np.exp(plank_const*input_quantity/boltzmann_const/temp)-1.)

    elif input_unit == 'wavelength':

        spectral_radiance = 2.*plank_const*light_speed**2/input_quantity**5/(np.exp(plank_const*light_speed/input_quantity/boltzmann_const/temp)-1.)

    else:
        print('error: unit is not correct')

    return spectral_radiance


def specific_dust_opacity(

    wave,
    beta = 2.,
    kappa_0 = 0.1,
    unit = 'frequency',
    system_of_units = 'cgs',
):

    if system_of_units == 'cgs':

        if unit == 'frequency':

            kappa = kappa_0 *(wave/1e12)**beta

        elif unit == 'wavelength':

            kappa = kappa_0 *(3e-4/wave)**beta

        else:

            print('error: unit is not correct')
    else:
        print('error: not supported system of unit chosen')

    return kappa

def gray_body(

    wave,
    temp,
    colden,
    beta = 2.,
    kappa_0 = 0.1,
    unit='frequency',
    system_of_units = 'cgs',
    version = 'Hildebrand+1983',
    colden_scale = 'log10',
    aver_mol_wei = 2.33

):

    if version == 'Hildebrand+1983':

        hydrogen_mass = aver_mol_wei*const.m_p.cgs.value

        if colden_scale == 'log10':
            colden_transform = 10**colden
        else:
            colden_transform = colden

        intensity = colden_transform*hydrogen_mass \
            *specific_dust_opacity(
                wave,
                beta = beta,
                kappa_0 = kappa_0,
                unit = unit,
                system_of_units = system_of_units
            )\
            *black_body(
                wave,
                temp,
                input_unit = unit,
                system_of_units = system_of_units
            )

    return intensity

def galactic_carbon_ratio(dist, dist_err):

    ratio_const1 = 6.21

    ratio_const2 = 18.71

    carbon_ratio = ratio_const1 * dist * 1e-3 + ratio_const2

    ratio_const1_err = 1.0

    ratio_const2_err = 7.37

    carbon_ratio_err = np.sqrt((dist * 1e-3 * ratio_const1_err)**2 \
                               + ratio_const1**2*(dist_err * 1e-3)**2\
                               + ratio_const2_err**2)

    return carbon_ratio, carbon_ratio_err


def isotopic_ratio(

    isotop_primary,
    isotop_secondary,
    rms_primary = None,
    rms_secondary = None
):

    iso_ratio = isotop_primary/isotop_secondary

    if rms_primary is not None:

        error_term_1 = rms_primary/isotop_secondary
        error_term_2 = rms_secondary*isotop_primary/isotop_secondary**2

        iso_ratio_error = np.sqrt(error_term_1**2 + error_term_2**2)

        return iso_ratio, iso_ratio_error

    else:

        return  iso_ratio

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
                        tau_min = 1e-6,
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
        pos_1=tau_min, pos_2=tau_max, resolution = 1e-6, max_step = 1e6)

        if not optical_depth:

            optical_depth = 1e-6

    else:

        print("error: the chosen method is not valid. chose between sybolic, numeric")

    #error = 0
    #if optical_depth == 0:

    #    error_tau = 0.

    #else:

        #print("optical_depth: "+ str(optical_depth))

    error_tau = abs(optical_depth**2 /((1. + optical_depth ) * np.exp(-optical_depth) -1. ))\
    * (ratio_norm/iso_ratio)\
    * np.sqrt( (error_main/iso_comp)**2 \
    + (main_comp*error_iso/iso_comp**2)**2 \
    + (main_comp*error_ratio/(iso_comp*iso_ratio))**2 )

    return optical_depth, error_tau


def tau_spect(
    
    spect_main,
    spect_iso,
    vel_main,
    vel_iso = None,
    rms_main = 0,
    rms_iso = 0,
    shift_iso = 0.,
    ratio_iso = 67.,
    ratio_err = 0.,
    norm_iso = 1.,
    vel_min = None,
    vel_max = None,
    vel_interp = None,
    use_axis = 'main',
    tau_max = 30
    
):

    if vel_iso is None:
        vel_iso = []


    if len(vel_iso) == 0:

        vel_iso[:] = copy.deepcopy(vel_main[:]) - shift_iso

    if use_axis == 'main':

        if vel_min and vel_max:

            idx_min = astrokit.get_idx(vel_min, vel_main, method = 'closer')
            idx_max = astrokit.get_idx(vel_max, vel_main, method = 'closer') +1 # +1 is to include the value at index 'idx_max' in the calculation

            vel = vel_main[idx_min : idx_max]

            temp_main = spect_main[idx_min : idx_max]

            temp_iso = np.interp(
                
                vel,
                vel_iso,
                spect_iso,
                left = 0.0,
                right = 0.0
                
            )

        else:

            vel = vel_main
            temp_main = spect_main
            temp_iso = np.interp(
                
                vel,
                vel_iso,
                spect_iso,
                left = 0.0,
                right = 0.0

            )

    elif use_axis == 'iso':

        if vel_min and vel_max:

            idx_min = astrokit.get_idx(vel_min, vel_iso, method = 'closer')
            idx_max = astrokit.get_idx(vel_max, vel_iso, method = 'closer')+1 # +1 is to include the value at index 'idx_max' in the calculation

            vel = vel_iso[idx_min : idx_max]

            temp_iso = spect_iso[idx_min : idx_max]

            temp_main = np.interp(vel,
                                  vel_main,
                                  spect_main,
                                  left = 0.0,
                                  right = 0.0)
        else:

            vel = vel_iso

            temp_iso = spect_iso

            temp_main = np.interp(vel,
                                  vel_main,
                                  spect_main,
                                  left = 0.0,
                                  right = 0.0)

    elif use_axis == 'interp':

        vel = vel_inpter

        temp_main = np.interp(vel,
                              vel_main,
                              spect_main,
                              left = 0.0,
                              right = 0.0)

        temp_iso = np.interp(vel,
                             vel_iso,
                             spect_iso,
                             left = 0.0,
                             right = 0.0)


    tau = np.zeros_like(vel)

    tau_err = np.zeros_like(vel)

    for channel in range(len(vel)):

        tau[channel], tau_err[channel] = \
            optical_depth_ratio(ratio_iso,
                                norm_iso,
                                temp_main[channel],
                                temp_iso[channel],
                                error_ratio = ratio_err,
                                error_main = rms_main,
                                error_iso = rms_iso,
                                method = "bisection",
                                tau_max = tau_max)

    return tau, tau_err

def tau_correction(value,
                   tau,
                   value_err = 0 ,
                   tau_err = 0 ):

    correction = (tau/(1-np.exp(-tau)))

    value_corr = value * correction

    err_1 = (correction*value_err)**2

    err_2 =(value * tau_err * ((1.-np.exp(-tau)*(1.+tau))/(1.-np.exp(tau))**2 ))**2

    value_corr_err = np.sqrt(err_1  +  err_2)

    return value_corr, value_corr_err

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

        Tmb_hyper[vel_idx] = sum(Tmb_temp)/sum(weight)


    return Tmb_hyper, vel_hyper


def inten_to_colden(
    inten,
    Tex=None,
    inten_err = 0,
    line = 'cii',
    unit = 'custom'
):

    molecular_line = False

    if line == 'cii':

        # cii rest frequency in [Hz]
        freq_cii = 1900.5369e9

        # Einstein's coefficient [1/s]
        einstein_coef = 2.29e-6

       # equivalent temperature of the excited level
        T_0 = (const.h.value*freq_cii)/const.k_B.value

        if unit == 'custom':

            # input unit: K km/s
            # output unit 1/cm^2

            conversion_factor = 1e-1

        elif unit == 'SI':

            # input unit: K m/s
            # output unit: 1/m^2

            conversion_factor = 1.

        if Tex:

            const_term = (1+(2./4.)*np.exp(T_0/Tex))* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)*conversion_factor


        else:

            const_term = (3./2.)* (8.*np.pi*freq_cii**3)/(T_0*const.c.value**3*einstein_coef)*conversion_factor


    if line == '12co(1-0)':

        J_up = 1

        dipole_const = 0.11011e-18

        freq_0 = 115.271202e9

        rot_const = 57.635968e9

        molecular_line = True

    elif line == '13co(1-0)':

        J_up = 1

        dipole_const = 0.11046e-18

        freq_0 = 110.201359e9

        rot_const = 55.101012e9

        molecular_line = True

    elif line == 'c18o(1-0)':

        J_up = 1

        dipole_const = 0.11049e-18

        freq_0 = 109.782176e9

        rot_const = 54.891421e9

        molecular_line = True


    elif line == '12co(2-1)':

        J_up = 2

        dipole_const = 0.11011e-18

        freq_0 = 230.538000e9

        rot_const = 57.635968e9

        molecular_line = True

    elif line == '13co(2-1)':

        J_up = 2

        dipole_const = 0.11046e-18

        freq_0 = 220.398686e9

        rot_const = 55.101012e9

        molecular_line = True

    elif line == 'c18o(2-1)':

        J_up = 2

        dipole_const = 0.11049e-18

        freq_0 = 219.560358e9

        rot_const = 54.891421e9

        molecular_line = True


    elif line == '12co(3-2)':

        J_up = 3

        dipole_const = 0.11011e-18

        freq_0 = 345.79598990e9

        rot_const = 57.635968e9

        molecular_line = True


    elif line == '13co(3-2)':

        J_up = 3

        dipole_const = 0.11046e-18

        freq_0 = 330.58796500e9

        rot_const = 55.101012e9

        molecular_line = True


    elif line == 'c18o(3-2)':

        J_up = 3

        dipole_const = 0.11049e-18

        freq_0 = 329.330e9

        rot_const = 54.891421e9

        molecular_line = True


    if molecular_line:

        #rot_const = freq_0/(2.*J_up)

        if unit == 'custom':

            # input unit: K km/s
            # output unit 1/cm^2

            conversion_factor = 1e5

        elif unit == 'SI':

            # input unit: K m/s
            # output unit: 1/m^2

            conversion_factor = 1e6



        T_0 = (const.h.cgs.value * freq_0)/const.k_B.cgs.value

        E_up = const.h.cgs.value*rot_const*J_up*(J_up+1.)

        const_term =\
            (3.*const.h.cgs.value*conversion_factor)/(8.*np.pi**3*dipole_const**2*J_up)\
            *( (const.k_B.cgs.value * Tex)/(const.h.cgs.value*rot_const)+1./3.)\
            * np.exp(E_up/(const.k_B.cgs.value*Tex))\
            /((np.exp(T_0/Tex)-1.)*(astrokit.brightness_temperatur(Tex, line = line) - astrokit.brightness_temperatur(2.7, line = line)) )

    colden = const_term*inten

    colden_err = const_term*inten_err

    return colden, colden_err

def colden_map(
    inten,
    inten_err = None,
    temp_ex = None,
    line = 'cii',
    unit = 'custom'
):


    colden     = astrokit.zeros_map(inten)
    colden_err = astrokit.zeros_map(inten)

    if inten_err:

        colden[0].data, colden_err[0].data = inten_to_colden(
            inten[0].data,
            Tex=temp_ex,
            inten_err = inten_err[0].data,
            line = line,
            unit = unit
        )
    else:

        colden[0].data, colden_err[0].data = inten_to_colden(
            inten[0].data,
            Tex=temp_ex,
            line = line,
            unit = unit
        )


###############################################################################
#    if line == 'cii':
#
#        # cii rest frequency in [Hz]
#        freq_cii = 1900.5369e9
#
#        # Einstein's coefficient [1/s]
#        einstein_coef = 2.29e-6
#
#       # equivalent temperature of the excited level
#        temp_0 = (const.h.value*freq_cii)/const.k_B.value
#
#        if temp_ex:
#
#            const_term = (1+(2./4.)*np.exp(temp_0/temp_ex))* (8.*np.pi*freq_cii**3)/(temp_0*const.c.value**3*einstein_coef)
#
#            colden[0].data = const_term*inten[0].data
#
#
#        else:
#
#            const_term = (3./2.)* (8.*np.pi*freq_cii**3)/(temp_0*const.c.value**3*einstein_coef)
#
#            colden[0].data = const_term*inten[0].data
#
#        if inten_err:
#
#            colden_err[0].data = const_term*inten_err[0].data
#
#        else:
#            colden_err[0].data = np.nan
#
#    elif line == 'co(3-2)':
#
#        colden[0].data = 1.58e13*(temp_ex+0.88)*np.exp(31.7/temp_ex)*inten[0].data
#
#        if inten_err:
#
#            colden_err[0].data = 1.58e13*(temp_ex+0.88)*np.exp(31.7/temp_ex)*inten_err[0].data
#
#        else:
#
#            colden_err[0].data = np.nan
#
###############################################################################

    return colden, colden_err


def colden_to_mass(colden,
                   area,
                   colden_err=0,
                   area_err=None,
                   line = 'hi',
                   abundance_ratio = 1.6e-4):


    if line == 'hi':

        # hydrogen mass
        element_mass = 1.6735575e-27

    elif line == 'h2':

        element_mass = 2.*1.6735575e-27

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

def inten_to_flux(inten, line, inten_err=0.):

    if line == 'cii':
        # cii rest frequency in [Hz]
        freq = 1900.5369e9

    elif line == '12co(1-0)':

        freq = 115.271202e9

    elif line == '13co(1-0)':

        freq = 110.201359e9

    elif line == '12co(3-2)':

        # 12CO (3-2) rest frequency [Hz]

        freq = 345.79598990e9

    elif line == '13co(3-2)':

        # 13CO (3-2) rest frequency [Hz]

        freq = 330.58796500e9

    elif line == '12co(4-3)':

        # 12CO (4-3) rest frequency [Hz]
        freq = 461.0406e9

    const_flux = 2.*const.k_B.value*(freq**3/const.c.value**3)

    flux = inten*const_flux

    flux_err = inten_err*const_flux

    return flux, flux_err

def flux_map(
    map_inten,
    line,
    map_inten_err = None

):

    if map_inten_err is None:

        out_err = False

        map_inten_err = astrokit.zeros_map(map_inten)

    else:

        out_err = True

    map_flux = astrokit.zeros_map(map_inten)
    map_flux_err = astrokit.zeros_map(map_inten)

    naxis_ra = map_inten[0].header['NAXIS1']
    naxis_dec = map_inten[0].header['NAXIS2']

    for ra in range(naxis_ra):
        for dec in range(naxis_dec):

            map_flux[0].data[dec, ra], map_flux_err[0].data[dec, ra] = inten_to_flux(
                inten = map_inten[0].data[dec, ra],
                line = line,
                inten_err = map_inten_err[0].data[dec, ra]
            )


    if out_err:

        return map_flux, map_flux_err

    else:

        return map_flux

    

def line_luminosity(inten,
                    line,
                    dist = None,
                    area = None,
                    dist_err=0,
                    area_err=None,
                    inten_err=0):

        flux, flux_err = inten_to_flux(inten, line, inten_err)

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

def mass_map(colden,
             colden_err = None,
             abundance_ratio = 1.2e-4,
             element = 'atom',
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

    if element == 'atom':

        element_mass = 1.6735575e-27

    elif element == 'molecule':

        element_mass = 2.*1.6735575e-27

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



            mass[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]] =\
                (pixel_size*colden[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]*element_mass)/abundance_ratio

            mass_err[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]] =\
                element_mass/abundance_ratio\
                * np.sqrt( pixel_size_err**2*colden[0].data[idx_sig3[0][idx_mask], idx_sig3[1][idx_mask]]**2
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

                    mass[0].data[idx_dec, idx_ra] = (pixel_size*colden[0].data[idx_dec, idx_ra]*element_mass)/abundance_ratio

                    mass_err[0].data[idx_dec, idx_ra] = element_mass/abundance_ratio\
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
                      noise_limit = 3.,
                      tau_max = 100,
                      method = 'bisection'):

    tau_map = astrokit.zeros_map(main_map)
    tau_err_map = astrokit.zeros_map(main_map)

    len_ax1 = main_map[0].header['NAXIS1']
    len_ax2 = main_map[0].header['NAXIS2']

    for idx_ax1 in range(len_ax1):
        for idx_ax2 in range(len_ax2):

            check_1 = (iso_map[0].data[idx_ax2, idx_ax1]*(iso_ratio/ratio_norm)) - main_map[0].data[idx_ax2, idx_ax1]
            check_2 = iso_map[0].data[idx_ax2, idx_ax1] - (noise_limit*iso_err_map[0].data[idx_ax2, idx_ax1])
            check_3 = main_map[0].data[idx_ax2, idx_ax1] - (noise_limit*main_err_map[0].data[idx_ax2, idx_ax1])

            if ((check_1 > 0.) and (check_2 > 0.) and (check_3 > 0.)):

                tau_map[0].data[idx_ax2, idx_ax1], tau_err_map[0].data[idx_ax2, idx_ax1] =\
                astrokit.optical_depth_ratio(iso_ratio,
                                             ratio_norm,
                                             main_map[0].data[idx_ax2, idx_ax1],
                                             iso_map[0].data[idx_ax2, idx_ax1],
                                             error_ratio = error_ratio,
                                             error_main = main_err_map[0].data[idx_ax2, idx_ax1],
                                             error_iso = iso_err_map[0].data[idx_ax2, idx_ax1],
                                             method = method,
                                             tau_max = tau_max)

            else:

                tau_map[0].data[idx_ax2, idx_ax1] = 0
                tau_err_map[0].data[idx_ax2, idx_ax1] = 0


    return tau_map, tau_err_map


def excitation_temperatur(
    temp_mb,
    tau = 0.,
    temp_continuum = 0.,
    temp_mb_err = 0,
    tau_err = 0,
    line = 'cii',
    optically_thick = True,
    calc_error = False,
    ):

    if line=='cii':

        freq_0 = 1900.5369e9

    elif line=='12co(3-2)':

        freq_0 = 345.79598990e9

    elif line == '13co(3-2)':

        freq_0 = 330.58796500e9

    temp_0 = (const.h.value*freq_0)/const.k_B.value

    if optically_thick:

        temp_ex = temp_0/np.log(temp_0/(temp_mb+temp_continuum) + 1.)

    else:

        temp_ex = temp_0/np.log(temp_0/(temp_mb+temp_continuum) * (1.-np.exp(-tau)) +1.)

    if calc_error:

        term_tau = -temp_0**2*np.exp(-tau)/temp_mb/(temp_0/temp_mb*(1.-np.exp(-tau))+1.)/np.log(temp_0/temp_mb*(1.-np.exp(-tau))+1.)**2

        term_temp = temp_0**2 *(1.-np.exp(-tau))/temp_mb**2/(temp_0/temp_mb*(1.-np.exp(-tau))+1.)/np.log(temp_0/temp_mb*(1.-np.exp(-tau))+1.)**2

        temp_ex_err = np.sqrt((term_tau*tau_err)**2+(term_temp*temp_mb_err)**2)

        return temp_ex, temp_ex_err

    else:

        return temp_ex

def jansky_to_kelvin(

    inten_jy, # intensity input in Jy/beam
    freq,     # frequency in Hz
    bmaj,     # beam size in arcsec
    bmin      # beam size in arcsec

):

    inten_k = 1.222e24 * inten_jy / freq**2 / bmaj / bmin

    return inten_k


def optical_depth_hisa(

    temp_diff,
    temp_hisa,
    temp_off,
    temp_cont,
    p = 1., # background to foreground ratio, with p=1 no foreground

):

    tau_hisa = -np.log(1.-temp_diff/(temp_hisa-p*temp_off-temp_cont))

    return tau_hisa


def column_density_hisa(

    tau_hisa,
    vel_hisa, # in km/s
    temp_spin, # in K
    channel_width = None, # in km/s
    methode = 'integral'
):

    if methode == 'integral':

        inten =  np.trapz(tau_hisa, vel_hisa)

    elif methode == 'channel':

        if channel_width is None:

            channel_width = abs(vel_hisa[1] - vel_hisa[0])

        inten = tau_hisa*channel_width

    colden_hisa = 1.8224e18 * temp_spin * inten

    return colden_hisa


def temp_hisa(

    temp_diff,
    tau_hisa,
    temp_off,
    temp_cont,
    p = 1.,

):

    temp_hisa = temp_diff/(1-np.exp(-tau_hisa)) +p*temp_off+temp_cont

    return temp_hisa

def SED_fit(

    dust_intensity,
    frequency,
    guess,
    bound_min,
    bound_max,
    sigma = None,
    maxfev = 100,
    aver_mol_wei = 2.33,
    beta = 2.,
    kappa_0 = 0.1,

):

    hydrogen_mass = aver_mol_wei*const.m_p.cgs.value

    try:

        if np.sum(dust_intensity) == 0:

            temp_dust = 0

            colden_dust = 0

        else:

            popt, pcov = curve_fit(

                lambda wave, temp, colden: astrokit.gray_body(

                    wave,
                    temp,
                    colden,
                    beta,
                    kappa_0,
                    unit='frequency',
                    system_of_units = 'cgs',
                    colden_scale = 'log10',
                    version = 'Hildebrand+1983'

                ),

                frequency,
                dust_intensity,
                sigma = sigma,
                p0 = guess,
                bounds = (bound_min, bound_max),
                maxfev = maxfev
            )

            temp_dust = popt[0]
            colden_dust = 10**popt[1]#/hydrogen_mass

            #print(popt)
        gray_body_fit = gray_body(

            frequency,
            popt[0],
            popt[1],
            beta = beta,
            kappa_0 = kappa_0,
            unit='frequency',
            system_of_units = 'cgs',
            version = 'Hildebrand+1983',
            colden_scale = 'log10',
            aver_mol_wei = aver_mol_wei

        )

        chi_squared = np.sum((dust_intensity - gray_body_fit)**2/gray_body_fit)


    except: #RuntimeError:

        temp_dust = 0

        colden_dust = 0

        chi_squared = 0


    return temp_dust, colden_dust, chi_squared

def SED_map(

    input_maps,
    frequency = None,
    wavelength = None,
    guess = [10, 1e21],
    bound_min = [2.7, 0.0],
    bound_max = [100, 1e24],
    sigma = None,
    num_cores = 1,
    maxfev = 100,
    input_map_unit = 'MJy/sr',
    aver_mol_wei = 2.33,
    beta = 2.,
    kappa_0 = 0.1,
):

    if frequency is None:
        frequency = []

    if wavelength is None:
        wavelength = []

    if sigma is None:
        sigma = []

    #hydrogen_mass = aver_mol_wei*const.m_p.cgs.value

    if guess[1] == 0:
        guess[1] = 0 
    else:
        guess[1] = np.log10(guess[1])#*hydrogen_mass

    if bound_min[1] == 0:
        bound_min[1] = 0
    else:
        bound_min[1] = np.log10(bound_min[1])#*hydrogen_mass

    if bound_max[1] == 0:
        bound_max[1] = 0
    else:
        bound_max[1] = np.log10(bound_max[1])#*hydrogen_mass

    if input_map_unit == 'MJy/sr':

        unit_to_cgs = 1e-17


    if len(frequency) == 0:

        frequency = const.c.value/wavelength


    if len(frequency) == 4:

        print('started 500 um map')

    elif len(frequency) == 3:

        print('started 350 um map')

    colden_dust_map = astrokit.zeros_map(input_maps[-1])
    temp_dust_map = astrokit.zeros_map(input_maps[-1])
    chi_squared_map = astrokit.zeros_map(input_maps[-1])


    if len(sigma) > 0:

        sigma = sigma*unit_to_cgs

    else:

        sigma = np.zeros(len(frequency))

        sigma[:] = unit_to_cgs


    len_ra = colden_dust_map[0].header['NAXIS1']
    len_dec = colden_dust_map[0].header['NAXIS2']

    dust_intensity_cube = np.zeros([len(frequency), len_dec, len_ra])

    for ra in range(len_ra):

        for dec in range(len_dec):

            for freq in range(len(frequency)):

                dust_intensity_cube[freq, dec, ra] =  np.nan_to_num(input_maps[freq][0].data[dec, ra])*unit_to_cgs


    dust_intensity_list = dust_intensity_cube.reshape(len(frequency), len_dec*len_ra).transpose()

    check_beta = isinstance(beta, list)
    check_kappa_0 = isinstance(kappa_0, list)
        
    if check_beta:
        
        list_beta = beta[0].data.reshape(len_dec*len_ra).transpose()
        
    if check_kappa_0:
        
        list_kappa_0 = kappa_0[0].data.reshape(len_dec*len_ra).transpose()

    #output_list  = Parallel(n_jobs=num_cores)(delayed(SED_fit)(dust_intensity, frequency, guess, bound_min, bound_max, sigma, maxfev = maxfev, aver_mol_wei = aver_mol_wei, beta = beta, kappa_0 = kappa_0) for dust_intensity in dust_intensity_list)
    if check_beta and check_kappa_0:
        
        output_list  = Parallel(n_jobs=num_cores)(delayed(astrokit.SED_fit)(dust_intensity_list[pixel], frequency, guess, bound_min, bound_max, sigma, maxfev = maxfev, aver_mol_wei = aver_mol_wei, beta = list_beta[pixel], kappa_0 = list_kappa_0[pixel]) for pixel in range(len(dust_intensity_list)))

    elif check_beta:
        
        output_list  = Parallel(n_jobs=num_cores)(delayed(astrokit.SED_fit)(dust_intensity_list[pixel], frequency, guess, bound_min, bound_max, sigma, maxfev = maxfev, aver_mol_wei = aver_mol_wei, beta = list_beta[pixel], kappa_0 = kappa_0) for pixel in range(len(dust_intensity_list)))
        
    elif check_kappa_0:
        
        output_list  = Parallel(n_jobs=num_cores)(delayed(astrokit.SED_fit)(dust_intensity_list[pixel], frequency, guess, bound_min, bound_max, sigma, maxfev = maxfev, aver_mol_wei = aver_mol_wei, beta = beta, kappa_0 = list_kappa_0[pixel]) for pixel in range(len(dust_intensity_list)))
        
    else:    
    
        output_list  = Parallel(n_jobs=num_cores)(delayed(astrokit.SED_fit)(dust_intensity_list[pixel], frequency, guess, bound_min, bound_max, sigma, maxfev = maxfev, aver_mol_wei = aver_mol_wei, beta = beta, kappa_0 = kappa_0) for pixel in range(len(dust_intensity_list)))
    

    temp_dust_list = np.zeros(len_dec*len_ra)
    colden_dust_list = np.zeros(len_dec*len_ra)
    chi_squared_list = np.zeros(len_dec*len_ra)

    for list_entry in range(len(output_list)):

        temp_dust_list[list_entry] = output_list[list_entry][0]
        colden_dust_list[list_entry] = output_list[list_entry][1]
        chi_squared_list[list_entry] = output_list[list_entry][2]


    temp_dust_map[0].data = temp_dust_list.reshape(len_dec,len_ra)
    colden_dust_map[0].data = colden_dust_list.reshape(len_dec,len_ra)
    chi_squared_map[0].data = chi_squared_list.reshape(len_dec,len_ra)

    return temp_dust_map, colden_dust_map, chi_squared_map

def dust_flux_ratio(

    temp,
    observed_ratio,
    beta,
    frequency = None,
    wavelength = None,

):

    if frequency is None:
        frequency = []

    if wavelength is None:
        wavelength = []

    if len(frequency) == 0:

        frequency = const.c.value/wavelength

    if len(wavelength) == 0:

        wavelength = const.c.value/frequency


    flux = np.zeros(2)

    flux[0] = astrokit.black_body(

        frequency[0],
        temp,
        input_unit='frequency',
        system_of_units = 'SI'

    )

    flux[1] = astrokit.black_body(

        frequency[1],
        temp,
        input_unit='frequency',
        system_of_units = 'SI'

    )

    flux_ratio = flux[0]/flux[1]*(wavelength[1]/wavelength[0])**beta

    return observed_ratio - flux_ratio


def dust_colden_map(

    input_maps,
    frequency = None,
    wavelength = None,
    temp_range = None,
    beta = 2,
    kappa_0 = 0.1,
    method = 'bisection',
    input_map_unit = 'MJy/sr',
    aver_mol_wei = 2.33
):

    if frequency is None:
        frequency = []

    if wavelength is None:
        wavelength = []

    if temp_range is None:
        temp_range = []



    if input_map_unit == 'MJy/sr':

        unit_to_cgs = 1e-17

    if len(frequency) == 0:

        frequency = const.c.value/wavelength

    if len(wavelength) == 0:

        wavelength = const.c.value/frequency

    if len(temp_range) == 0:

        temp_range = [2.7, 100]


    colden_dust_map = astrokit.zeros_map(input_maps[-1])
    temp_dust_map = astrokit.zeros_map(input_maps[-1])


    len_ra = colden_dust_map[0].header['NAXIS1']
    len_dec = colden_dust_map[0].header['NAXIS2']

    observed_flux = np.zeros(2)

    for ra in range(len_ra):

        clear_output(wait=True)
        print('Progress: ' + '#'* int((ra/len_ra)*30+1) + ' '+ str(round((ra/(len_ra-1))*100, 1))+'%')

        for dec in range(len_dec):

            observed_flux[0] = np.nan_to_num(input_maps[0][0].data[dec, ra]) * unit_to_cgs
            observed_flux[1] = np.nan_to_num(input_maps[1][0].data[dec, ra]) * unit_to_cgs

            if ((observed_flux[0]>0) and (observed_flux[1]>0)):

                observed_ratio = observed_flux[0]/observed_flux[1]

                if method == 'bisection':

                    temp_dust_map[0].data[dec, ra] = astrokit.solver.bisection(
                        lambda temp : dust_flux_ratio(

                            temp,
                            observed_ratio,
                            beta,
                            frequency,
                            wavelength
                        ),
                        pos_1 = temp_range[0],
                        pos_2 = temp_range[1],
                        resolution = 1e-6,
                        max_step = 1e6
                    )

                elif method == 'brentq':

                    try:

                        temp_dust_map[0].data[dec, ra] = brentq(
                            lambda temp : dust_flux_ratio(

                                temp,
                                observed_ratio,
                                beta,
                                frequency,
                                wavelength
                            ),
                            a = temp_range[0],
                            b = temp_range[1]
                        )

                    except:
                        pass


            if (temp_dust_map[0].data[dec, ra] > temp_range[1]):

                temp_dust_map[0].data[dec, ra] = 0

            elif (temp_dust_map[0].data[dec, ra] < temp_range[0]):

                temp_dust_map[0].data[dec, ra] = temp_range[0]

                colden_dust_map[0].data[dec, ra] = observed_flux[1] \
                                                 / astrokit.specific_dust_opacity(frequency[1], beta, kappa_0, unit = 'frequency', system_of_units = 'cgs') \
                                                 /  astrokit.black_body(frequency[1], temp_dust_map[0].data[dec, ra], input_unit='frequency', system_of_units = 'cgs') \
                                                 /  const.m_p.cgs.value / aver_mol_wei

            else :

                colden_dust_map[0].data[dec, ra] = observed_flux[1] \
                                                 / astrokit.specific_dust_opacity(frequency[1], beta, kappa_0, unit = 'frequency', system_of_units = 'cgs') \
                                                 /  astrokit.black_body(frequency[1], temp_dust_map[0].data[dec, ra], input_unit='frequency', system_of_units = 'cgs') \
                                                 /  const.m_p.cgs.value / aver_mol_wei


    return temp_dust_map, colden_dust_map


def highres_dust_colden(

    input_map_250,
    input_map_350,
    input_map_500

    ):

    highres_colden = astrokit.zeros_map(input_map_250)

    beam_500 = (input_map_500[0].header['BMAJ'] + input_map_500[0].header['BMIN'])*3600/2.
    beam_350 = (input_map_350[0].header['BMAJ'] + input_map_350[0].header['BMIN'])*3600/2.
    beam_250 = (input_map_250[0].header['BMAJ'] + input_map_250[0].header['BMIN'])*3600/2.

    #print('sum map_250: '+ str(np.sum(input_map_250[0].data)))
    #print('sum map_350: '+ str(np.sum(input_map_350[0].data)))
    #print('sum map_500: '+ str(np.sum(input_map_500[0].data)))

    map_500_350 = astrokit.regrid_spectral_cube(

        input_map_350,
        input_map_350,
        input_beam = beam_350,
        target_beam = beam_500,
        input_frame = 'icrs',
        target_frame = 'icrs',
        target_header = False

    )

    map_350_250 = astrokit.regrid_spectral_cube(

        input_map_250,
        input_map_250,
        input_beam = beam_250,
        target_beam = beam_350,
        input_frame = 'icrs',
        target_frame = 'icrs',
        target_header = False

    )

    #print('sum map_350_250: '+ str(np.sum(map_350_250[0].data)))
    #print('sum map_500_350: '+ str(np.sum(map_500_350[0].data)))

    len_ra = input_map_500[0].header['NAXIS1']
    len_dec = input_map_500[0].header['NAXIS2']

    for ra in range(len_ra):
        for dec in range(len_dec):

            #if np.isnan(input_map_350[0].data[dec, ra]) or np.isinf(input_map_350[0].data[dec, ra]) \
            #or np.isnan(input_map_250[0].data[dec, ra]) or np.isinf(input_map_250[0].data[dec, ra]) \
            #or np.isnan(map_500_350[0].data[dec, ra]) or np.isinf(map_500_350[0].data[dec, ra]) \
            #or np.isnan(map_350_250[0].data[dec, ra]) or np.isinf(map_350_250[0].data[dec, ra]):

            #    highres_colden[0].data[dec, ra] = input_map_500[0].data[dec, ra]

            #else:

            if ((input_map_500[0].data[dec, ra] == 0) or (input_map_350[0].data[dec, ra] == 0)):

                highres_colden[0].data[dec, ra] = input_map_250[0].data[dec, ra]

            else:

                highres_colden[0].data[dec, ra] = input_map_500[0].data[dec, ra] \
                                                + (input_map_350[0].data[dec, ra] - map_500_350[0].data[dec, ra]) \
                                                + (input_map_250[0].data[dec, ra] - map_350_250[0].data[dec, ra])

    return highres_colden

def kinetic2exitation_temperatur(
    temp_kin, 
    density_hydrogen, 
    tau_peak = None,
    temp_cont = 2.7
):
    # [CII] Einstein A coefficients
    Aul = 2.29e-6

     # [CII] transition frequency
    freq_cii = 1900.5369e9
        
    temp_0 = (const.h.value*freq_cii)/const.k_B.value

    if tau_peak is None:

        beta = 1
    
    else:

        beta = (1.-np.exp(-tau_peak))/tau_peak

 
    Rul = 7.6e-10*(temp_kin/100)**(0.14)
        
    Cul = density_hydrogen*Rul

    term_continuum = 1./(np.exp(temp_0/temp_cont)-1.)
        
    temp_ex = temp_0/np.log( (Cul + beta*(1+term_continuum)*Aul)/( term_continuum*beta*Aul + Cul*np.exp(-temp_0/temp_kin) )  )
        
    return temp_ex
        
def shell_profile(

    radius_out = 1,
    radius_in = None,
    shell_width = 0.8,
    colden = 1e21,
    grid_points = 1e3,
    colden_profile = False
): 
    
    if radius_in is None:
    
        radius_in = radius_out - shell_width
    
    else:
    
        shell_width = radius_out - radius_in
        
    dist_res = radius_out/grid_points
        
    dist_cent = np.arange(0, radius_out+dist_res, dist_res)

    path_out = np.sqrt(radius_out**2 - dist_cent**2)
    path_in = np.sqrt(radius_in**2 - dist_cent**2)

    colden_los = np.zeros_like(dist_cent)
    path_los = np.zeros_like(dist_cent)
    
    path_los[dist_cent >= radius_in] = path_out[dist_cent >= radius_in]
    path_los[dist_cent < radius_in] = path_out[dist_cent < radius_in] - path_in[dist_cent < radius_in]

    colden_los = colden * path_los/shell_width

    path_los_max = np.sqrt(radius_out**2 - radius_in**2)
    
    limb_brightening = path_los_max/shell_width

    path_max_half = (path_los_max+shell_width)/2.

    path_fwhm_out = np.sqrt((radius_out)**2 - path_max_half**2)
    path_fwhm_in = 1./(2.*path_max_half) * np.sqrt(2.*(radius_in**2*path_max_half**2 + radius_in**2*radius_out**2 + radius_out**2*path_max_half**2) - (radius_in**4 + radius_out**4 + path_max_half**4))

    fwhm_los = path_fwhm_out - path_fwhm_in
    
    if colden_profile:
    
        return limb_brightening, fwhm_los, colden_los, path_los 
    
    else:
    
        return limb_brightening, fwhm_los 