#############################################################
#
#  radeq.py includes the two layer multi componentk
#  radiative transfer equations.
#
#  CHAOS (Cologne Hacked Astrophysical Software)
#
#  Created by Slawa Kabanovic
#
#############################################################

import numpy as np
import astrokit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib

from astropy import constants as const

from IPython.display import clear_output

def line_profile(
    
    vel = None,
    amp = None,
    vel_0 = None,
    width = None,
    abundace_ratio = 67.,
    profile = 'gauss',
    line = 'cii',
    iso_shift = 0.,
    norm = False
    
):

    #print('line profile:')
    #print(profile)
    #print(line)

    if line == 'cii':

        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

    elif line == '12co(3-2)':

        # 12CO (3-2) rest frequency [Hz]

        freq_0 = 345.79598990e9

    elif line == '13co(3-2)':

        # 13CO (3-2) rest frequency [Hz]

        freq_0 = 330.58796500e9

    elif line == '12co(4-3)':

        # 12CO (4-3) rest frequency [Hz]
        freq_0 = 461.0406e9

    # conversion factor from FWHM to sigma
    beta = 4.*np.log(2.)

    if profile == 'gauss':

        if norm:

            line_norm = np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)

            line_shape = line_norm*amp*np.exp(-(vel-vel_0)**2*beta/width**2)

        else:

            line_shape = amp*np.exp(-(vel-vel_0)**2*beta/width**2)

    elif profile == 'double_gauss':

        gauss_main = np.exp(-(vel-vel_0)**2*beta/width**2)

        gauss_iso = (1./abundace_ratio)*np.exp(-(vel-vel_0-iso_shift)**2*beta/width**2)

        if norm:

            line_norm = np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)

            line_shape = line_norm*amp*(gauss_main+gauss_iso)

        else:

            line_shape = amp*(gauss_main+gauss_iso)

    elif profile == 'cii' or profile == 'cii_wing' or profile == '13cii':

        # velocity shift of the [13CII] F(2-1), F(1-0) and F(1-1)
        # transition lines with respect to the [12CII] line [km/s]
        vel_13cii_f21 = 11.2
        vel_13cii_f10 = -65.2
        vel_13cii_f11 = 63.3

        # relative line strength of the 13CII satellites
        line_f21_norm = 0.625
        line_f10_norm = 0.250
        line_f11_norm = 0.125

        # [12CII] line profile
        line_12cii = np.exp(-(vel-vel_0)**2*beta/width**2)

        # [13CII] F(2-1) line profile
        line_13cii_f21 = (line_f21_norm/abundace_ratio)\
                       * np.exp(-(vel-vel_13cii_f21-vel_0)**2*beta/width**2)

        # [13CII] F(1-0) line profile
        line_13cii_f10 = (line_f10_norm/abundace_ratio)\
                       * np.exp(-(vel-vel_13cii_f10-vel_0)**2*beta/width**2)

        # [13CII] F(1-1) line profile
        line_13cii_f11 = (line_f11_norm/abundace_ratio)\
                       * np.exp(-(vel-vel_13cii_f11-vel_0)**2*beta/width**2)

        if norm:

            line_norm = np.sqrt(beta)/(np.sqrt(np.pi)*width*1e3)

            if profile == '13cii':

                line_shape = line_norm*amp\
                    * (line_13cii_f21 + line_13cii_f10 + line_13cii_f11)

            else:

                line_shape = line_norm*amp\
                    * (line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)
        else:

            if profile == '13cii':

                line_shape = amp*(line_13cii_f21 + line_13cii_f10 + line_13cii_f11)

            else:

                line_shape = amp*(line_12cii + line_13cii_f21 + line_13cii_f10 + line_13cii_f11)



    return line_shape


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

def optical_depth(
    vel,
    vel_0,
    width,
    temp_ex,
    colden,
    abundace_ratio = 67.,
    profile = 'cii',
    line = 'cii',
    iso_shift = 0.
):

    molecular_line = False

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
            * colden * (1. - np.exp(-temp_0/temp_ex))\
            / (1. + (g_u/g_l) * np.exp(-temp_0/temp_ex))

    elif line == '13co(3-2)':

        J_up = 3

        dipole_const = 0.11046e-18

        freq_0 = 330.587e9

        rot_const = 55.101012e9

        molecular_line = True


    elif line == '12co(3-2)':

        J_up = 3

        dipole_const = 0.11011e-18

        freq_0 = 345.795e9

        rot_const = 57.635968e9

        molecular_line = True


    elif line == 'c18o(3-2)':

        J_up = 3

        dipole_const = 0.11049e-18

        freq_0 = 329.330e9

        rot_const = 54.891421e9

        molecular_line = True


    if molecular_line:

        const_unit = 1e-2

        T_0 = (const.h.cgs.value * freq_0)/const.k_B.cgs.value

        E_up = const.h.cgs.value*rot_const*J_up*(J_up+1.)

        amp = colden*const_unit*(8.*np.pi**3*dipole_const**2*J_up)\
            /(3.*const.h.cgs.value*((const.k_B.cgs.value * temp_ex)/(const.h.cgs.value*rot_const)+1./3.))\
            * np.exp(-E_up/(const.k_B.cgs.value*temp_ex)) * (np.exp(T_0/temp_ex)-1.)

    tau = line_profile(vel,
                       amp,
                       vel_0,
                       width,
                       abundace_ratio,
                       profile,
                       line,
                       iso_shift,
                       norm = True)


    return tau

def column_density(vel_0,
                   width,
                   tau_0,
                   temp_ex,
                   abundace_ratio = 67.,
                   profile = 'cii',
                   line = 'cii',
                   iso_shift = 0.,
                   channel = False,
                   output_unit = 'SI'):

    molecular_line = False


    if line == 'cii':
        # [12CII] rest frequency [Hz]
        freq_0 = 1900.5369e9

        # statistical weight of the level
        g_u = 4.
        g_l = 2.

        # Einstein's coefficient [1/s]
        A_ul = 2.29e-6

        temp_0 = (const.h.value * freq_0)/const.k_B.value

        amp = (g_u*A_ul*const.c.value**3)/(g_l*8.*np.pi*freq_0**3) \
                    * (1. - np.exp(-temp_0/temp_ex))/(1. + (g_u/g_l) * np.exp(-temp_0/temp_ex))

    elif line == '13co(3-2)':

        J_up = 3

        dipole_const = 0.11046e-18

        freq_0 = 330.587e9

        rot_const = 55.101012e9

        molecular_line = True


    elif line == '12co(3-2)':

        J_up = 3

        dipole_const = 0.11011e-18

        freq_0 = 345.795e9

        rot_const = 57.635968e9

        molecular_line = True


    elif line == 'c18o(3-2)':

        J_up = 3

        dipole_const = 0.11049e-18

        freq_0 = 329.330e9

        rot_const = 54.891421e9

        molecular_line = True


    if molecular_line:

        T_0 = (const.h.cgs.value * freq_0)/const.k_B.cgs.value

        E_up = const.h.cgs.value*rot_const*J_up*(J_up+1.)

        if output_unit == 'SI':

            const_unit = 1.0e-6

        elif output_unit == 'cgs':

            const_unit = 1.0e-2

        amp = const_unit*(8.*np.pi**3*dipole_const**2*J_up)\
            / (3.*const.h.cgs.value*((const.k_B.cgs.value * temp_ex)/(const.h.cgs.value*rot_const)+1./3.))\
            *  np.exp(-E_up/(const.k_B.cgs.value*temp_ex)) * (np.exp(T_0/temp_ex)-1.)


    if channel:

        colden = tau_0/amp

    else:

        colden = tau_0/line_profile(vel_0,
                                    amp,
                                    vel_0,
                                    width,
                                    abundace_ratio,
                                    profile,
                                    line,
                                    iso_shift,
                                    norm = True)

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

    if line=='hi':

        freq_0 = 1.4204057517667e9

    elif line=='cii':

        freq_0 = 1900.5369e9

    elif line == '12co(1-0)':

        freq_0 = 115.271202e9

    elif line == '13co(1-0)':

        freq_0 = 110.201359e9

    elif line == 'c18o(1-0)':

        freq_0 = 109.782176e9

    elif line == '12co(2-1)':

        freq_0 = 230.538000e9

    elif line == '13co(2-1)':

        freq_0 = 220.398686e9

    elif line == 'c18o(2-1)':

        freq_0 = 219.560358e9

    elif line=='12co(3-2)':

        freq_0 = 345.79598990e9

    elif line == '13co(3-2)':

        freq_0 = 330.58796500e9

    elif line == 'c18o(3-2)':

        freq_0 = 329.330e9


    temp_0 = (const.h.value * freq_0)/const.k_B.value

    return (temp_0/(np.exp(temp_0/temp_ex)-1.))


def temp_to_tau(temp_mb,
                temp_ex,
                line = 'cii'):

    if line=='cii':

        freq_0 = 1900.5369e9

    elif line=='12co(3-2)':

        freq_0 = 345.79598990e9

    elif line == '13co(3-2)':

        # 13CO (3-2) rest frequency [Hz]

        freq_0 = 330.58796500e9

    temp_0 = (const.h.value * freq_0)/const.k_B.value

    tau = -np.log(1.-(temp_mb*(np.exp(temp_0/temp_ex)-1.))/temp_0 )

    return tau



#############################################################
#
# determine: excitation temperatur for each componet
#
# the fuction takes as an input:
#
# temp_mb = main beam temperatur of each component [K]
#
#############################################################

#def excitation_temperatur(temp_mb, line = 'cii'):
#
#    if line=='cii':
#
#        freq_0 = 1900.5369e9
#
#    temp_0 = (const.h.value * freq_0)/const.k_B.value
#
#    return (temp_0/(np.log((temp_0/temp_mb) + 1.)))

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

def two_layer(

    bakg_num,
    forg_num,
    temp_ex,
    vel,
    *param,
    profile,
    line,
    iso_shift = 0.,
    line_shift = None,
    line_ratio = None,
    abundace_ratio = None,
    input_colden = False,
    temp_cont = None,

):

    forg_tau_tot = np.zeros_like(vel)
    
    bakg_term = np.zeros_like(vel)
    forg_term = np.zeros_like(vel)

    cont_term = np.zeros_like(vel)
    
    check_list = isinstance(line, list)

    #print(check_list)

    if check_list:

        num_lines = len(line)
    
    else:

        num_lines = 1

    for idx_line in range(num_lines):
        
        bakg_tau = np.zeros_like(vel)
        forg_tau = np.zeros_like(vel) 

        if num_lines == 1:

            profile = [profile]

            line = [line]

      #  else:

      #      profile = profiles[idx_line] 

      #      line = lines[idx_line]

        if (forg_num > 0):

            if num_lines == 1:

                temp_bakg = temp_ex[0]
                temp_forg = temp_ex[1]

                #print('num is = 1')
                #print(temp_bakg)

            else:

                temp_bakg = temp_ex[idx_line, 0]
                temp_forg = temp_ex[idx_line, 1]

                #print('num is > 1')
                #print(temp_bakg)

            for i in range(0, (3*bakg_num), 3):

                if num_lines == 1:

                    vel_0 = param[i+1]
                
                else:
                
                    vel_0 = param[i+1] + line_shift[idx_line]    

                width    = param[i+2]

                if input_colden:

                    if num_lines == 1:

                        colden = param[i]

                    else:

                        colden = param[i]/line_ratio[idx_line, 0]
   
                    tau = optical_depth(

                        vel,
                        vel_0,
                        width,
                        temp_bakg,
                        colden,
                        abundace_ratio = abundace_ratio,
                        profile = profile[idx_line],
                        line = line[idx_line],
                        iso_shift = iso_shift

                    )

                else:

                    if num_lines == 1:
                        
                        tau_0 = param[i]

                    else:

                        tau_0 = param[i]/line_ratio[idx_line, 0]

                    tau = line_profile(

                        vel,
                        tau_0,
                        vel_0,
                        width,
                        abundace_ratio,
                        profile[idx_line],
                        line[idx_line],
                        iso_shift,
                        norm = False

                    )

                bakg_tau = bakg_tau + tau

            for i in range((3*bakg_num), (3*bakg_num+3*forg_num), 3):

                if num_lines == 1:

                    vel_0 = param[i+1]
                
                else:
                
                    vel_0 = param[i+1] + line_shift[idx_line]    

                width    = param[i+2]

                if input_colden:

                    if num_lines == 1:

                        colden = param[i]

                    else:

                        colden = param[i]/line_ratio[idx_line, 1]

                    tau = optical_depth(

                        vel,
                        vel_0,
                        width,
                        temp_forg,
                        colden,
                        abundace_ratio = abundace_ratio,
                        profile = profile[idx_line],
                        line = line[idx_line],
                        iso_shift = iso_shift

                    )

                else:

                    if num_lines == 1:
                        
                        tau_0 = param[i]

                    else:

                        tau_0 = param[i]/line_ratio[idx_line, 1]

                    tau = line_profile(

                        vel,
                        tau_0,
                        vel_0,
                        width,
                        abundace_ratio,
                        profile[idx_line],
                        line[idx_line],
                        iso_shift,
                        norm = False

                    )

                forg_tau = forg_tau + tau


            bakg_term = bakg_term + brightness_temperatur(temp_bakg, line[idx_line])*(1. - np.exp(-bakg_tau))
            forg_term = forg_term + brightness_temperatur(temp_forg, line[idx_line])*(1. - np.exp(-forg_tau))
            
            forg_tau_tot = forg_tau_tot + forg_tau
            
            #print(np.max(bakg_tau))
            #print(np.max(forg_tau))
            
            #print(idx_line)
            #plt.plot(vel, forg_term + bakg_term * np.exp(-forg_tau_tot))
            

            if temp_cont is not None:

                cont_term = cont_term + brightness_temperatur(temp_cont, line[idx_line])*(np.exp(-bakg_tau-forg_tau) - 1.)   

        else:

            if num_lines == 1:

                temp_bakg = temp_ex

            else:

                temp_bakg = temp_ex[idx_line]

            for i in range(0, (3*bakg_num), 3):

                if num_lines == 1:

                    vel_0 = param[i+1]
                
                else:
                
                    vel_0 = param[i+1] + line_shift[idx_line]    

                width    = param[i+2]

                if input_colden:

                    if num_lines == 1:

                        colden = param[i]

                    else:

                        colden = param[i]/line_ratio[idx_line, 0]

                    tau = optical_depth(
                        vel,
                        vel_0,
                        width,
                        temp_bakg,
                        colden,
                        abundace_ratio = abundace_ratio,
                        profile = profile[idx_line],
                        line = line[idx_line],
                        iso_shift = iso_shift
                    )

                else:

                    if num_lines == 1:
                        
                        tau_0 = param[i]

                    else:

                        tau_0 = param[i]/line_ratio[idx_line, 0]

                    #print('two layer function:')
                    #print(profile[idx_line])
                    #print(line[idx_line])

                    tau = line_profile(
                        vel,
                        tau_0,
                        vel_0,
                        width,
                        abundace_ratio,
                        profile[idx_line],
                        line[idx_line],
                        iso_shift,
                        norm = False
                    )

                bakg_tau = bakg_tau + tau


            bakg_term = bakg_term + brightness_temperatur(temp_bakg, line[idx_line])*(1. - np.exp(-bakg_tau))

            if temp_cont is not None:

                cont_term = cont_term + brightness_temperatur(temp_cont, line[idx_line])*(np.exp(-bakg_tau) - 1.)


    #if (forg_num > 0):

    return (cont_term + forg_term + bakg_term * np.exp(-forg_tau_tot))

    #else:

    #    return (cont_term + bakg_term)

def temp_mb_error(

    temp_ex,
    tau,
    temp_ex_err,
    tau_err,
    line = 'cii'

):

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


def plot_2layer_model(

    spect,
    vel,
    rms,
    fit_param,
    bakg_num,
    forg_num,
    temp_ex,
    abundace_ratio,
    profile = 'gauss',
    line = '12co(3-2)',
    input_colden = False,
    vel_range = None,
    iso_shift = 0.,
    res_plot = 0.01,
    title = 'Model Fit',
    fontsize = 24,
    labelsize = 24,
    save_image = False,
    image_path = './',
    image_name = 'plot_2layer_model'

):


    if not vel_range:

        vel_range = [vel[0], vel[-1]]

    vel_plot = np.arange(vel[0], vel[-1], res_plot)

    if forg_num == 0:

        temp_fit = astrokit.two_layer(
            bakg_num,
            forg_num,
            temp_ex[0],
            vel,
            *fit_param,
            profile = profile,
            line = line,
            iso_shift = iso_shift,
            abundace_ratio = abundace_ratio,
            input_colden = input_colden
        )

        temp_plot = astrokit.two_layer(
            bakg_num,
            forg_num,
            temp_ex[0],
            vel_plot,
            *fit_param,
            profile = profile,
            line = line,
            iso_shift = iso_shift,
            abundace_ratio = abundace_ratio,
            input_colden = input_colden
        )

    else:

        temp_fit = astrokit.two_layer(
            bakg_num,
            forg_num,
            temp_ex,
            vel,
            *fit_param,
            profile = profile,
            line = line,
            iso_shift = iso_shift,
            abundace_ratio = abundace_ratio,
            input_colden = input_colden
        )

        temp_plot = astrokit.two_layer(
            bakg_num,
            forg_num,
            temp_ex,
            vel_plot,
            *fit_param,
            profile = profile,
            line = line,
            iso_shift = iso_shift,
            abundace_ratio = abundace_ratio,
            input_colden = input_colden
        )




    # determine residula of the fit
    residual = spect-temp_fit

    # determine chi squared of the fit
    chisq_fit = sum((residual/rms) ** 2)/(len(residual)-(3*bakg_num+3*forg_num))

    print('chi squared=', chisq_fit)

    #############################################################
    #
    # determine single components of both layers
    #
    #############################################################

    # determine the single background layers

    bakg_tau      = np.zeros([bakg_num,len(vel_plot)])
    bakg_tau_sum  = np.zeros([len(vel_plot)])
    bakg_temp     = np.zeros([bakg_num,len(vel_plot)])
    bakg_temp_sum = np.zeros([len(vel_plot)])
    idx_num       = 0

    for idx in range(0,(3*bakg_num), 3):

        # load paramter
        vel_0   = fit_param[idx+1]
        width   = fit_param[idx+2]

        # determine the optical depth of the background component per velocity chanel

        if input_colden:

            colden = fit_param[idx]

            bakg_tau[idx_num,:] = astrokit.optical_depth(
                vel_plot,
                vel_0,
                width,
                temp_ex[0],
                colden,
                abundace_ratio = abundace_ratio,
                profile = profile,
                line = line,
                iso_shift = iso_shift
                )

            bakg_temp[idx_num,:] = astrokit.two_layer(
                1,
                0,
                temp_ex[0],
                vel_plot,
                colden,
                vel_0,
                width,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                abundace_ratio = abundace_ratio,
                input_colden = input_colden
                )

        else:

            tau_0 = fit_param[idx]

            bakg_tau[idx_num,:] = astrokit.line_profile(
                vel_plot,
                tau_0,
                vel_0,
                width,
                abundace_ratio = abundace_ratio,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                norm = False)

            # determine the main beam temperatur of the single background per velocity chanel
            bakg_temp[idx_num,:] = astrokit.two_layer(
                1,
                0,
                temp_ex[0],
                vel_plot,
                tau_0,
                vel_0,
                width,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                abundace_ratio = abundace_ratio,
                input_colden = input_colden
                )

        # determine the total background temperatur per velocity chanel
        #bakg_temp_sum += bakg_temp[idx_num,:]
        bakg_tau_sum  += bakg_tau[idx_num,:]

        # go to the next background component
        idx_num += 1

    bakg_temp_sum = astrokit.two_layer(
        bakg_num,
        0,
        temp_ex[0],
        vel_plot,
        *fit_param[:bakg_num*3],
        profile = profile,
        line = line,
        iso_shift = iso_shift,
        abundace_ratio = abundace_ratio,
        input_colden = input_colden
        )
    # determine the single foreground layers

    forg_tau      = np.zeros([forg_num,len(vel_plot)])
    forg_tau_sum  = np.zeros([len(vel_plot)])
    forg_temp     = np.zeros([forg_num,len(vel_plot)])
    forg_temp_sum = np.zeros([len(vel_plot)])
    idx_num       = 0

    for idx in range((3*bakg_num),(3*bakg_num+3*forg_num), 3):

        vel_0   = fit_param[idx+1]
        width   = fit_param[idx+2]


        if input_colden:

            colden = fit_param[idx]

            forg_tau[idx_num,:] = astrokit.optical_depth(
                vel_plot,
                vel_0,
                width,
                temp_ex[1],
                colden,
                abundace_ratio = abundace_ratio,
                profile = profile,
                line = line,
                iso_shift = iso_shift
                )

            forg_temp[idx_num,:] = astrokit.two_layer(
                1,
                0,
                temp_ex[1],
                vel_plot,
                colden,
                vel_0,
                width,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                abundace_ratio = abundace_ratio,
                input_colden = input_colden
                )

        else:

            tau_0 = fit_param[idx]

            # determine the optical depth of the foreground component per velocity chanel
            forg_tau[idx_num,:] = astrokit.line_profile(
                vel_plot,
                tau_0,
                vel_0,
                width,
                abundace_ratio = abundace_ratio,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                norm = False
                )

            # determine the main beam temperatur of the single foreground per velocity chanel
            forg_temp[idx_num,:] = astrokit.two_layer(
                1,
                0,
                temp_ex[1],
                vel_plot,
                tau_0,
                vel_0,
                width,
                profile = profile,
                line = line,
                iso_shift = iso_shift,
                abundace_ratio = abundace_ratio
                )

        # determine the total foreground temperatur per velocity chanel
        #forg_temp_sum += forg_temp[idx_num,:]
        forg_tau_sum  += forg_tau[idx_num,:]

        # go to the next foreground component
        idx_num += 1

    forg_temp_sum = astrokit.two_layer(
        forg_num,
        0,
        temp_ex[1],
        vel_plot,
        *fit_param[bakg_num*3:],
        profile = profile,
        line = line,
        iso_shift = iso_shift,
        abundace_ratio = abundace_ratio,
        input_colden = input_colden
        )

    vmin = vel_range[0]
    vmax = vel_range[1]

    if max(temp_fit)>max(spect):
        tmax = max(temp_fit)*1.05
        tmin = -max(temp_fit)*0.05
    else:
        tmax = max(spect)*1.05
        tmin = -max(spect)*0.05

    fontsize = fontsize

    fig = plt.figure(figsize=(32, 19))

    plt.subplots_adjust(wspace=0.25,hspace=0.00)
    plt.subplot2grid((3, 3), (0, 0), rowspan=2)
    plt.plot(vel, spect, color='C3', linewidth=2, label="Observed Data")
    plt.plot(vel_plot, temp_plot, color='C2', linewidth=2, label="Model Fit")
    plt.legend(prop={'size': labelsize}, loc='upper right')
    plt.xlim(vmin, vmax)
    plt.ylim(tmin, tmax)
    plt.xticks([])
    plt.yticks(size=fontsize)
    plt.ylabel('$\mathrm{T_{mB}}$ [K]', size=fontsize)
    plt.title(title, size=fontsize, fontweight="bold")

    #tmax = 10.*rms_arr[0]
    #tmin = -10.*rms_arr[0]

    if (np.max(residual)> (3.*rms[0])):
        tmax = np.max(residual) *1.1
        tmin = -np.max(residual) *1.1

    else:

        tmax = (3.*rms[0])*1.1
        tmin = -(3.*rms[0])*1.1


    plt.subplot2grid((3, 3), (2, 0))
    plt.plot(vel, residual, color='C7', linewidth=2, label="Fit Residual")

    rms_plot = np.zeros_like(rms)
    rms_plot[:] = 3.*np.max(rms)
    plt.plot(vel, rms_plot, color='C1', linewidth=2, label="$3\,\sigma$")
    plt.legend(prop={'size': labelsize}, loc='upper right')
    plt.plot(vel, -rms_plot, color='C1', linewidth=2)
    plt.ylabel('Residual', size = fontsize)
    plt.xlabel('Velocity [km/s]', size = fontsize)
    plt.xlim(vmin, vmax)
    plt.yticks(size = fontsize)
    plt.xticks(size = fontsize)
    plt.ylim(tmin, tmax)

    plt.subplot2grid((3, 3), (0, 1), rowspan=2)

    plt.plot(vel, spect, color='C3', linewidth=2)

    idx_peak = 0

    tau_max = max(bakg_tau_sum)*1.2
    tau_min = -tau_max*0.05

    tmax = max(bakg_temp_sum)*1.2
    tmin = tmax*tau_min/tau_max

    if(bakg_num==1):

        plt.plot(vel_plot, bakg_temp_sum, color='C0', linewidth=2, label="Synthetic Background Spectrum")

    elif (bakg_num>1):

        plt.plot(vel_plot, bakg_temp_sum, color='C9', linewidth=2, label="Synthetic Background Spectrum")

        for idx_bakg in range(bakg_num-1):

            plt.plot(vel_plot, bakg_temp[idx_bakg,:], color='C0', linewidth=2)

        plt.plot(vel_plot, bakg_temp[idx_bakg+1,:], color='C0', linewidth=2, label="Background Components")

    plt.xlim(vmin, vmax)
    plt.ylim(tmin, tmax)

    plt.legend(prop={'size': labelsize}, loc='upper right')
    plt.ylabel('$\mathrm{T_{mB}}$ [K]',size=fontsize, color = 'C0')
    plt.tick_params(axis='y', labelcolor='C0')
    plt.xticks([])
    plt.yticks(size=fontsize)

    ax2 = plt.twinx()
    ax2.plot(vel_plot, bakg_tau_sum, color='C1',linestyle = '--', linewidth=2, label="Synthetic Background Optical Depth")
    ax2.set_ylabel('Optical Depth', size=fontsize, color = 'C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.tick_params(labelsize = fontsize)
    ax2.set_ylim([tau_min, tau_max])

    plt.subplot2grid((3, 3), (2, 1))

    if forg_num==0:

        tau_max = 1
        tau_min = -0.1

        tmax = 1
        tmin = tmax*tau_min/tau_max

    else:

        tau_max = max(forg_tau_sum)*1.3
        tau_min = -tau_max*0.05

        tmax = max(forg_temp_sum)*1.3
        tmin = tmax*tau_min/tau_max

    if (forg_num==0):

        plt.plot(vel_plot, forg_temp_sum, color='C6', linewidth=2, label="Synthetic Foreground Spectrum")

    elif (forg_num==1):

        plt.plot(vel_plot, forg_temp_sum, color='C6', linewidth=2, label="Synthetic Foreground Spectrum")

    else:

        plt.plot(vel_plot, forg_temp_sum, color='C4', linewidth=2, label="Synthetic Foreground Spectrum")

        for idx_forg in range(forg_num-1):

            plt.plot(vel_plot, forg_temp[idx_forg,:], color='C6', linewidth=2)

        plt.plot(vel_plot, forg_temp[idx_forg+1,:], color='C6', linewidth=2, label="Foreground Components")

    plt.xlim(vmin,vmax)
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    plt.legend(prop={'size': labelsize}, loc='lower right')
    plt.tick_params(axis='y', labelcolor='C6')
    plt.ylabel('$\mathrm{T_{mB}}$ [K]',size=fontsize, color = 'C6')
    plt.ylim(tmin, tmax)
    plt.gca().invert_yaxis()
    plt.xlabel('Velocity [km/s]',size=fontsize)
    plt.xlim(vmin,vmax)

    ax2 = plt.twinx()
    ax2.plot(vel_plot, forg_tau_sum, color='C1',linestyle = '--', linewidth=2, label="Synthetic Foreground Optical Depth")
    ax2.set_ylabel('Optical Depth', size=fontsize, color = 'C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.tick_params(labelsize = fontsize)
    ax2.set_ylim([tau_min, tau_max])
    ax2.invert_yaxis()


    if save_image:

        matplotlib.pyplot.savefig(image_path + image_name,
                                  transparent = False,
                                  bbox_inches = 'tight',
                                  pad_inches = 0)


def bakg_2layer_cube(

    cube_iso,
    map_rms,
    map_cluster_num_bakg,
    cube_guess,
    cube_bound_min,
    cube_bound_max,
    vel_fit,
    temp_ex_map,
    abundace_ratio,
    rms_threshold = 1.,
    chisq_threshold = 2.,
    chisq_ratio = 0.,
    profile = 'gauss',
    line = '13co(3-2)',
    temp_ex_method = 'single'

):

    axis = 1
    axis_ra = astrokit.get_axis(axis, cube_iso)

    axis = 2
    axis_dec = astrokit.get_axis(axis, cube_iso)

    cube_fit = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = vel_fit*1e3,
        ref_value_ax1 = cube_iso[0].header['CRVAL1'],
        ref_value_ax2 = cube_iso[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_iso[0].header['BMAJ'],
        beam_min = cube_iso[0].header['BMIN']
    )

    cube_bakg = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = vel_fit*1e3,
        ref_value_ax1 = cube_iso[0].header['CRVAL1'],
        ref_value_ax2 = cube_iso[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_iso[0].header['BMAJ'],
        beam_min = cube_iso[0].header['BMIN']
    )

    param_axis = astrokit.get_axis(3, cube_guess)

    fit_param = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = param_axis,
        ref_value_ax1 = cube_iso[0].header['CRVAL1'],
        ref_value_ax2 = cube_iso[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_iso[0].header['BMAJ'],
        beam_min = cube_iso[0].header['BMIN']
    )

    fit_err = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = param_axis,
        ref_value_ax1 = cube_iso[0].header['CRVAL1'],
        ref_value_ax2 = cube_iso[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_iso[0].header['BMAJ'],
        beam_min = cube_iso[0].header['BMIN']
    )

    axis = 3
    vel_iso = astrokit.get_axis(axis, cube_iso)/1e3

    naxis1    = cube_iso[0].header['NAXIS1']
    naxis2    = cube_iso[0].header['NAXIS2']

    rms_spect  = np.zeros_like(vel_iso)

    fit_pram_num = cube_guess[0].header['NAXIS3']

    #fit_param  = np.zeros([fit_pram_num, naxis2, naxis1])

    chisq_fit  = astrokit.zeros_map(cube_iso)
    bakg_num   = astrokit.zeros_map(cube_iso)
    failed_pix = astrokit.zeros_map(cube_iso)

    for dec in range(naxis2):
        clear_output(wait=True)
        print('Progress: ' + '#'* int((dec/naxis2)*30+1) + ' '+ str(round((dec/(naxis2-1))*100, 1))+'%')
        #print('Progress: '+str(round((dec*100/naxis2), 1)) +' %')
        for ra in range(naxis1):

            if temp_ex_method == 'single':

                temp_ex = temp_ex_map

            else:

                temp_ex = temp_ex_map[0].data[dec, ra]

            if (np.max(cube_iso[0].data[:, dec, ra]) > (rms_threshold*map_rms[0].data[dec, ra])):

                comp_num = 1

                continue_loop = True

                first_try = 0

                while (comp_num < (map_cluster_num_bakg[0].data[dec, ra]+1)) and continue_loop:

                    rms_spect[:] = map_rms[0].data[dec, ra]

                    guess     = np.zeros(3*comp_num)
                    bound_min = np.zeros(3*comp_num)
                    bound_max = np.zeros(3*comp_num)

                    for comp in range(0, len(guess), 3):
                        for param in range(3):

                            guess[comp + param]     = cube_guess[0].data[comp + param,   dec, ra]
                            bound_min[comp + param] = cube_bound_min[0].data[comp + param,   dec, ra]
                            bound_max[comp+ param]  = cube_bound_max[0].data[comp + param,   dec, ra]

                    try:

                        popt_new, pcov_new = curve_fit(

                            lambda vel, *param: astrokit.two_layer(

                                comp_num,
                                0,
                                temp_ex,
                                vel_iso,
                                *param,
                                profile = profile,
                                line = line,
                                abundace_ratio = abundace_ratio
                            ),
                            vel_iso,
                            cube_iso[0].data[:, dec, ra],
                            sigma = rms_spect,
                            absolute_sigma = True,
                            p0 = guess,
                            bounds = (bound_min, bound_max),
                            maxfev = 1000
                        )

                        if first_try == 0:

                            comp_old = comp_num
                            popt_old = popt_new
                            perr_old = np.sqrt(np.diag(pcov_new))

                            first_try = 1

                            fit_old = astrokit.two_layer(
                                comp_old,
                                0,
                                temp_ex,
                                vel_iso,
                                *popt_old,
                                profile = profile,
                                line = line,
                                abundace_ratio = abundace_ratio
                                )

                            # determine residual fo the fit
                            residual = cube_iso[0].data[:, dec, ra] - fit_old[:]

                            # determine chi squared of the fit
                            chisq_old = sum((residual/rms_spect) ** 2)/(len(residual)- (3*comp_old))

                            if (chisq_old < chisq_threshold):

                                continue_loop = False

                            else:

                                comp_num += 1

                        else:

                            comp_new = comp_num

                            fit_new = astrokit.two_layer(
                                comp_new,
                                0,
                                temp_ex,
                                vel_iso,
                                *popt_new,
                                profile = profile,
                                line = line,
                                abundace_ratio = abundace_ratio
                            )

                            # determine residual fo the fit
                            residual = cube_iso[0].data[:, dec, ra] - fit_new[:]

                            # determine chi squared of the fit
                            chisq_new = sum((residual/rms_spect) ** 2)/(len(residual) - (3. * comp_new))

                            #print(chisq_new/chisq_old)

                            #print(popt_old)
                            #print(popt_new)

                            if ((1. - (chisq_new/chisq_old)) > chisq_ratio):

                                comp_old = comp_new
                                popt_old = popt_new
                                perr_old = np.sqrt(np.diag(pcov_new))
                                chisq_old = chisq_new

                                if (chisq_old < chisq_threshold):

                                    continue_loop = False

                                else:

                                    comp_num += 1

                            else:

                                comp_num += 1

                                #continue_loop = False

                    except:
                        comp_num += 1

                if first_try > 0:

                    for param in range(len(popt_old)):

                        fit_param[0].data[param, dec, ra]  = popt_old[param]
                        fit_err[0].data[param, dec, ra]  = perr_old[param]

                    cube_fit[0].data[:, dec, ra] = astrokit.two_layer(
                        comp_old,
                        0,
                        temp_ex,
                        vel_fit,
                        *popt_old,
                        profile = profile,
                        line = line,
                        abundace_ratio = abundace_ratio
                    )

                    for comp in range(0, 3*comp_old, 3):

                        popt_old[comp] = popt_old[comp]*abundace_ratio

                    cube_bakg[0].data[:, dec, ra] = astrokit.two_layer(
                        comp_old,
                        0,
                        temp_ex,
                        vel_fit,
                        *popt_old,
                        profile = profile,
                        line = line,
                        abundace_ratio = abundace_ratio
                    )

                    chisq_fit[0].data[dec, ra]  = chisq_old
                    bakg_num[0].data[dec, ra]   = comp_old

                else:

                    print('failed')
                    failed_pix[0].data[dec, ra] = 1

    return fit_param, fit_err, chisq_fit, bakg_num, failed_pix, cube_fit, cube_bakg

def forg_2layer_cube(

    cube_obs,
    cube_rms,
    cube_iso_fit_param,
    cube_iso_fit_err,
    map_iso_bakg_num,
    map_bakg_temp_ex,
    map_forg_temp_ex,
    cube_bakg_guess,
    cube_forg_guess,
    cube_bakg_bound_min,
    cube_forg_bound_min,
    cube_bakg_bound_max,
    cube_forg_bound_max,
    map_num_bakg,
    map_num_forg,
    vel_fit,
    abundace_ratio,
    iso_shift,
    rms_threshold = 1.,
    chisq_threshold = 2.,
    chisq_ratio = 0.,
    profile_fit = 'double_gauss',
    profile_out = 'gauss',
    line = '12co(3-2)',
    maxfev = 1000,
    max_tau = False,
#    fix_forg_temp = True,
    colden_forg_max = None

):

    axis = 1
    axis_ra = astrokit.get_axis(axis, cube_obs)

    axis = 2
    axis_dec = astrokit.get_axis(axis, cube_obs)

    cube_fit = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = vel_fit*1e3,
        ref_value_ax1 = cube_obs[0].header['CRVAL1'],
        ref_value_ax2 = cube_obs[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_obs[0].header['BMAJ'],
        beam_min = cube_obs[0].header['BMIN']

    )

    cube_bakg = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = vel_fit*1e3,
        ref_value_ax1 = cube_obs[0].header['CRVAL1'],
        ref_value_ax2 = cube_obs[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_obs[0].header['BMAJ'],
        beam_min = cube_obs[0].header['BMIN']

    )

    cube_forg = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = vel_fit*1e3,
        ref_value_ax1 = cube_obs[0].header['CRVAL1'],
        ref_value_ax2 = cube_obs[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_obs[0].header['BMAJ'],
        beam_min = cube_obs[0].header['BMIN']

    )

    param_bakg_axis = astrokit.get_axis(3, cube_bakg_guess)
    param_forg_axis = astrokit.get_axis(3, cube_forg_guess)

    param_axis = np.arange(0, len(param_bakg_axis)+len(param_forg_axis), 1)


    fit_param = astrokit.empty_grid(

        grid_ax1 = axis_ra,
        grid_ax2 = axis_dec,
        grid_ax3 = param_axis,
        ref_value_ax1 = cube_obs[0].header['CRVAL1'],
        ref_value_ax2 = cube_obs[0].header['CRVAL2'],
        ref_value_ax3 = 0,
        beam_maj = cube_obs[0].header['BMAJ'],
        beam_min = cube_obs[0].header['BMIN']

    )

    axis = 3
    vel_obs = astrokit.get_axis(axis, cube_obs)/1e3

    naxis1    = cube_obs[0].header['NAXIS1']
    naxis2    = cube_obs[0].header['NAXIS2']

    chisq_fit  = astrokit.zeros_map(cube_obs)
    bakg_num   = astrokit.zeros_map(cube_obs)
    forg_num   = astrokit.zeros_map(cube_obs)
    failed_pix = astrokit.zeros_map(cube_obs)

    temp_ex = np.zeros(2)

    #test_count = 0

    for dec in range(naxis2):
        clear_output(wait=True)
        print('Progress: ' + '#'* int((dec/naxis2)*30+1) + ' '+ str(round((dec/(naxis2-1))*100, 1))+'%')
        #print('Progress: '+str(round((dec*100/naxis2), 1)) +' %')
        for ra in range(naxis1):

            temp_ex[0] = map_bakg_temp_ex[0].data[dec, ra]
            temp_ex[1] = map_forg_temp_ex[0].data[dec, ra]

            #test_count += 1
            #print('Progress: '+str(round((test_count*100/(naxis2*naxis1) ), 1)) +' %')

            first_try = 0

            if ((np.max(cube_obs[0].data[:, dec, ra]) > (rms_threshold*cube_rms[0].data[0, dec, ra]))
            and (map_iso_bakg_num[0].data[dec, ra]>0)):

                comp_forg_num = 1
                comp_bakg_num = 0

                dark_bakg_num = map_num_bakg[0].data[dec, ra] - map_iso_bakg_num[0].data[dec, ra]

                continue_loop = True

                bakg_num_guess   = int(map_iso_bakg_num[0].data[dec, ra])

                while (comp_forg_num < (map_num_forg[0].data[dec, ra]+1 )) and continue_loop:

                    guess     = np.zeros(3 * bakg_num_guess + 3 * comp_forg_num)
                    bound_min = np.zeros(3 * bakg_num_guess + 3 * comp_forg_num)
                    bound_max = np.zeros(3 * bakg_num_guess + 3 * comp_forg_num)

                    for comp in range(0, int(map_iso_bakg_num[0].data[dec, ra])*3, 3):
                        for param in range(3):

                            if param == 0:

                                guess[comp + param]   = cube_iso_fit_param[0].data[comp + param, dec, ra] * abundace_ratio
                                bound_min[comp + param] = (cube_iso_fit_param[0].data[comp + param, dec, ra] - cube_iso_fit_err[0].data[comp + param, dec, ra])* abundace_ratio
                                bound_max[comp + param] = (cube_iso_fit_param[0].data[comp + param, dec, ra] + cube_iso_fit_err[0].data[comp + param, dec, ra])* abundace_ratio

                            else:

                                guess[comp + param]   = cube_iso_fit_param[0].data[comp + param, dec, ra]
                                bound_min[comp + param] = cube_iso_fit_param[0].data[comp + param, dec, ra] - cube_iso_fit_err[0].data[comp + param, dec, ra]
                                bound_max[comp + param] = cube_iso_fit_param[0].data[comp + param, dec, ra] + cube_iso_fit_err[0].data[comp + param, dec, ra]

                            #fit_const = 0.1
                            #print(guess)
                            #bound_min[comp + param]   = guess[comp + param] - abs(guess[comp + param]*fit_const)
                            #bound_max[comp + param]   = guess[comp + param] + abs(guess[comp + param]*fit_const)

                    if ((dark_bakg_num > 0) and (comp_bakg_num > 0)):

                        for comp_bakg in range(
                            int(map_iso_bakg_num[0].data[dec, ra])*3,
                            bakg_num_guess*3,
                            3
                        ):
                            for param in range(3):

                                if max_tau:

                                    tau_rms = astrokit.temp_to_tau(

                                        cube_rms[0].data[-1, dec, ra],
                                        temp_ex[0],
                                        line = '13co(3-2)'
                                    )

                                    tau_max = tau_rms * abundace_ratio *3.

                                else:

                                    tau_max = abundace_ratio

                                if param == 0:

                                    guess[comp_bakg + param] = tau_max/10.
                                    bound_min[comp_bakg + param] = 0.0
                                    bound_max[comp_bakg + param] = tau_max

                                else:

                                    guess[comp_bakg + param]   = cube_bakg_guess[0].data[comp_bakg + param, dec, ra]
                                    bound_min[comp_bakg + param]   = cube_bakg_bound_min[0].data[comp_bakg + param, dec, ra]
                                    bound_max[comp_bakg + param]   = cube_bakg_bound_max[0].data[comp_bakg + param, dec, ra]

                    cube_comp_forg = 0

                    for comp_forg in range(
                        bakg_num_guess*3,
                        len(guess),
                        3
                    ):

                        fit_const = 0.1

                        for param in range(3):

                            guess[comp_forg + param]   = cube_forg_guess[0].data[cube_comp_forg + param,   dec, ra]

                            #if (param == 0) and fix_forg_temp:

                            #        bound_min[comp_forg + param]   = guess[comp_forg + param] - abs(guess[comp_forg + param]*fit_const)
                            #        bound_max[comp_forg + param]   = guess[comp_forg + param] + abs(guess[comp_forg + param]*fit_const)

                            #else:

                            bound_min[comp_forg + param] = cube_forg_bound_min[0].data[cube_comp_forg + param, dec, ra]

                            if (param == 0) and colden_forg_max:

                                bound_max[comp_forg + param] = astrokit.optical_depth(
                                    0.,
                                    0.,
                                    bound_max[comp_forg + 3],
                                    bound_max[comp_forg],
                                    colden_forg_max,
                                    abundace_ratio = abundace_ratio,
                                    profile = profile_out,
                                    line = line,
                                    iso_shift = 0.
                                )
                            else:

                                bound_max[comp_forg + param] = cube_forg_bound_max[0].data[cube_comp_forg + param, dec, ra]

                        cube_comp_forg += 3

                    try:

                        popt_new, pcov_new = curve_fit(

                            lambda vel,
                            *param: astrokit.two_layer(
                                bakg_num_guess,
                                comp_forg_num,
                                temp_ex,
                                vel_obs,
                                *param,
                                profile = profile_fit,
                                line = line,
                                iso_shift = iso_shift,
                                abundace_ratio = abundace_ratio
                            ),
                            vel_obs,
                            cube_obs[0].data[:, dec, ra],
                            sigma = cube_rms[0].data[:, dec, ra],
                            absolute_sigma = True,
                            p0 = guess,
                            bounds = (bound_min, bound_max),
                            maxfev = maxfev
                        )

                        if first_try == 0:

                            comp_bakg_old = bakg_num_guess
                            comp_forg_old = comp_forg_num
                            popt_old = popt_new
                            first_try = 1

                            fit_old = astrokit.two_layer(
                                comp_bakg_old,
                                comp_forg_old,
                                temp_ex,
                                vel_obs,
                                *popt_old,
                                profile = profile_fit,
                                line = line,
                                iso_shift = iso_shift,
                                abundace_ratio = abundace_ratio
                            )

                            # determine residual fo the fit
                            residual = cube_obs[0].data[:, dec, ra] - fit_old[:]

                            # determine chi squared of the fit
                            chisq_old = sum((residual/cube_rms[0].data[:, dec, ra]) ** 2)/(len(residual)- (3.*comp_forg_old + 3.*comp_bakg_old))

                            if (chisq_old < chisq_threshold):

                                continue_loop = False

                            else:

                                if( (dark_bakg_num > 0)
                                and (comp_bakg_num < dark_bakg_num) ):

                                    comp_bakg_num += 1

                                    bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra]) + comp_bakg_num

                                else:

                                    comp_forg_num += 1

                                    comp_bakg_num = 0

                                    bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra])


                        else:

                            comp_bakg_new = bakg_num_guess
                            comp_forg_new = comp_forg_num


                            fit_new = astrokit.two_layer(

                                comp_bakg_new,
                                comp_forg_new,
                                temp_ex,
                                vel_obs,
                                *popt_new,
                                profile = profile_fit,
                                line = line,
                                iso_shift = iso_shift,
                                abundace_ratio = abundace_ratio

                            )

                            # determine residual fo the fit
                            residual = cube_obs[0].data[:, dec, ra] - fit_new[:]

                            # determine chi squared of the fit
                            chisq_new = sum((residual/cube_rms[0].data[:, dec, ra]) ** 2)/(len(residual)- (3.*comp_forg_new + 3.*comp_bakg_new))

                            #print(chisq_new/chisq_old)

                            if ((1. - (chisq_new/chisq_old)) > chisq_ratio):


                                comp_bakg_old = comp_bakg_new
                                comp_forg_old = comp_forg_new
                                popt_old = popt_new
                                chisq_old = chisq_new

                                if (chisq_old < chisq_threshold):

                                    continue_loop = False

                                else:

                                    if( (dark_bakg_num > 0)
                                    and (comp_bakg_num < dark_bakg_num) ):

                                        comp_bakg_num += 1

                                        bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra]) + comp_bakg_num

                                    else:

                                        comp_forg_num += 1

                                        comp_bakg_num = 0

                                        bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra])

                            else:

                                if( (dark_bakg_num > 0)
                                and (comp_bakg_num < dark_bakg_num) ):

                                    comp_bakg_num += 1

                                    bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra]) + comp_bakg_num

                                else:

                                    comp_forg_num += 1

                                    comp_bakg_num = 0

                                    bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra])

                    except:

                        if( (dark_bakg_num > 0)
                        and (comp_bakg_num < dark_bakg_num) ):

                            comp_bakg_num += 1

                            bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra]) + comp_bakg_num

                        else:

                            comp_forg_num += 1

                            comp_bakg_num = 0

                            bakg_num_guess = int(map_iso_bakg_num[0].data[dec, ra])

                if first_try > 0:

                    for param in range(len(popt_old)):

                        fit_param[0].data[param, dec, ra]  = popt_old[param]

                    cube_fit[0].data[:, dec, ra] = astrokit.two_layer(
                        comp_bakg_old,
                        comp_forg_old,
                        temp_ex,
                        vel_fit,
                        *popt_old,
                        profile = profile_out,
                        line = line,
                        abundace_ratio = abundace_ratio

                    )

                    cube_bakg[0].data[:, dec, ra] = astrokit.two_layer(

                        comp_bakg_old,
                        0,
                        temp_ex[0],
                        vel_fit,
                        *popt_old[:comp_bakg_old*3],
                        profile = profile_out,
                        line = line,
                        abundace_ratio = abundace_ratio

                    )


                    cube_forg[0].data[:, dec, ra] = astrokit.two_layer(

                        comp_forg_old,
                        0,
                        temp_ex[1],
                        vel_fit,
                        *popt_old[3*comp_bakg_old:],
                        profile = profile_out,
                        line = line,
                        abundace_ratio = abundace_ratio

                    )

                    chisq_fit[0].data[dec, ra]  = chisq_old
                    forg_num[0].data[dec, ra]   = comp_forg_old
                    bakg_num[0].data[dec, ra]   = comp_bakg_old

                else:

                    if map_iso_bakg_num[0].data[dec, ra] > 0:

                        guess = np.zeros(3*int(map_iso_bakg_num[0].data[dec, ra]))

                        for comp in range(0, int(map_iso_bakg_num[0].data[dec, ra])*3, 3):

                            for param in range(3):

                                if param == 0:

                                    guess[comp + param] = cube_iso_fit_param[0].data[comp + param,   dec, ra]*abundace_ratio

                                else:

                                    guess[comp + param] = cube_iso_fit_param[0].data[comp + param,   dec, ra]

                        bakg_num[0].data[dec, ra] = map_iso_bakg_num[0].data[dec, ra]
                        forg_num[0].data[dec, ra] = 0

                        for param in range(len(guess)):

                            fit_param[0].data[param, dec, ra]  = guess[param]

                        cube_bakg[0].data[:, dec, ra] = astrokit.two_layer(

                            int(map_iso_bakg_num[0].data[dec, ra]),
                            0,
                            temp_ex[0],
                            vel_fit,
                            *guess,
                            profile = profile_out,
                            line = line,
                            abundace_ratio = abundace_ratio

                        )

                    print('failed')
                    failed_pix[0].data[dec, ra] = 1

    return fit_param, chisq_fit, bakg_num, forg_num, failed_pix, cube_fit, cube_bakg, cube_forg
