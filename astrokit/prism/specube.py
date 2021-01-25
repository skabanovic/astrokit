#############################################################
#
# specube.py is part of astrokit and provides
# multiple tools to analyze spectral cube data
#
# Created by Slawa Kabanovic
#
#############################################################


#############################################################
#
# load python libraries
#
#############################################################

import copy
import math
import numpy as np
import numpy.ma as ma
import astropy.units as u
import astrokit

from astropy.io import fits
from astropy.coordinates import Angle

from astropy.wcs import WCS

from astrokit.math import curve

#############################################################
#
# the function "get_axis()" determine the axis values from the
# fits-header of a given spectral cube
#
# function input:
#
# axis: chose which grid axis you like to be determined.
#            1 --> first spatial axis, right ascension (RA) in [deg]
#            2 --> second spatical axis, declination (DEC) in [deg]
#            3 --> velocity/frequency axis in [m/s]/[Hz]
#
# hdul: input spectral cube
#
#############################################################

def get_axis(axis, hdul):

    #length of cube axis
    axis_len    = hdul[0].header["NAXIS" + str(axis)]

    # reference grid position of axis
    ref_pos     = hdul[0].header["CRPIX" + str(axis)]

    # step size of axis
    step_size   = hdul[0].header["CDELT" + str(axis)]

    # value of reference grid position of axis
    ref_value   = hdul[0].header["CRVAL" + str(axis)]

    # determine the grid axis points
    grid_axis = (np.arange(1, axis_len+1) - ref_pos) * step_size + ref_value

    return grid_axis

#############################################################
#
# the function "get_idx()" determines the index
# of a given grid point from a equidistant grid/axis. If the
# given grid position is located between two grid/axis points with
# index index i and i+1, index i is returned
#
# function input:
#
# grid_pos: a position along the grid axis
# grid_axis: a grid axis given as an equidistant 1D array
#
#############################################################

def get_idx(grid_pos, grid_axis, method = 'lower'):

    # determines the reference position,
    # which is the first value of the grid
    grid_ref  = grid_axis[0]

    # determies the step/bin size of the grid axis
    grid_bin  = grid_axis[1]-grid_ref

    # determies the distance of the given grid position to the
    # reference position
    rel_pos   = grid_pos-grid_ref

    # determines the index of the grid point on the given grid axis
    if method == 'lower':

        grid_idx = int(rel_pos/grid_bin)

    elif method == 'closer':

        grid_idx = int(round(rel_pos/grid_bin))

    return grid_idx


def get_grid(hdul):

    dim = hdul[0].header['NAXIS']

    if dim == 2:

        hdul_inp = hdul

    elif dim == 3:

        hdul_inp = astrokit.zeros_map(hdul)

    w = WCS(hdul_inp[0].header)

    pix_ra  = np.arange(1, hdul_inp[0].header['NAXIS1']+1, 1)
    pix_dec = np.arange(1, hdul_inp[0].header['NAXIS2']+1, 1)

    idx_grid_ra, idx_grid_dec = np.meshgrid(pix_ra, pix_dec, sparse=False, indexing='xy')

    grid_ra, grid_dec = w.all_pix2world(idx_grid_ra, idx_grid_dec, 1)

    return grid_ra, grid_dec



def get_solid_angle(hdu, output = 'rad'):

    wcs = WCS(hdu[0].header)
    nx = hdu[0].header['NAXIS1']
    ny = hdu[0].header['NAXIS2']
    xx = np.arange(-0.5, nx, 1)
    xy = np.arange(0, ny, 1)
    yx = np.arange(0, nx, 1)
    yy = np.arange(-0.5, ny, 1)
    xX, xY = np.meshgrid(xx, xy)
    yX, yY = np.meshgrid(yx, yy)
    xXw, xYw = wcs.all_pix2world(xX, xY, 0)
    yXw, yYw = wcs.all_pix2world(yX, yY, 0)
    dXw = xXw[:,1:] - xXw[:,:-1]
    dYw = yYw[1:] - yYw[:-1]
    if output == 'rad':

        A = abs(Angle(dXw*u.deg).rad * Angle(dYw*u.deg).rad)

    else:

        A = abs(dXw*dYw)

    return A

#############################################################
#
# the function "get_value()" determines the spectral intensity
# value for a given velocity/frequency. If the given
# velocity/frequency point is located between two grid points
# i and i+1, the corresponding spectral intensity is determined by
# interpolation between the two grid points.
#
# function input:
#
# ax_pos: velocity/frequency value
# ax_grid: velocity/frequency axis
# ay_grid: intensity axis
# method: chose interpolation method
#   --> "linear"
#   --> to be included
#
#############################################################

def get_value(ax_pos, ax_grid, ay_values, method="linear"):


    # determines the first value of the axis
    axis_ref  = ax_grid[0]

    # determies the step/bin size of the axis/grid
    ax_res  = ax_grid[1]-axis_ref

    # determies the distance of the given value to the
    # first value on the axis
    rel_pos   = ax_pos-axis_ref

    # determines the index of the value on the given grid/axis
    idx_rel = rel_pos/ax_res

    idx_pos = int(idx_rel)

    if method == "linear":

        if (idx_rel - idx_pos) <1e-6:

            value_pos = ay_values[idx_pos]

        else:

            weight_2 = idx_rel-idx_pos

            weight_1 = 1. - weight_2

            value_pos = (ay_values[idx_pos]*weight_1\
             + ay_values[idx_pos+1]*weight_2)

        return value_pos

#############################################################
#
# the function "channel_map()" integrate a the
# spectral cube between a given velocity/frequency range.
#
# function input:
#
# vel_min: integration staring point
# vel_max: integration ending point
# hdul: input spectral cube
#
#############################################################

def cube_integral(vel_min, vel_max, hdul_inp):

    # make an empty 2d grid with the same spatial
    # size as the input cube
    map_size = np.zeros_like(hdul_inp[0].data[0,:,:])
    hdu = fits.PrimaryHDU(map_size)
    map_intg = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    map_intg[0].header = copy.deepcopy(hdul_inp[0].header)

    # remove 3D attributes from header
    for attribute in list(map_intg[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '3'):
                del map_intg[0].header[attribute]

    map_intg[0].header['NAXIS'] = 2
    map_intg[0].header['BUNIT'] = 'K km/s (Tmb)'

    num_pos = int((vel_max-vel_min)/(hdul_inp[0].header["CDELT3"]/1.0e3))

    axis = 3
    vel_inp = get_axis(axis, hdul_inp)/1e3

    if num_pos == 0:

        vel = [[vel_min, vel_max]]

    else:

        vel_res = (vel_max - vel_min)/num_pos

        vel = np.arange(vel_min, vel_max, vel_res)

        vel = np.append(vel, vel_max)

        temp = np.zeros_like(vel)

    for idx_dec in range(len(map_intg[0].data[:,0])):
        for idx_ra in range(len(map_intg[0].data[0,:])):


            temp = np.interp(vel, vel_inp, hdul_inp[0].data[:, idx_dec, idx_ra])

            map_intg[0].data[idx_dec, idx_ra] = np.trapz(temp, vel)


    return map_intg


def channel_map(hdul, ch_min, ch_max, ch_res = 1, unit = 'channel'):

    # make an empty 2d grid with the same spatial
    # size as the input cube
    map_size = np.zeros_like(hdul[0].data[0,:,:])
    hdu = fits.PrimaryHDU(map_size)
    map_intg = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    map_intg[0].header = copy.deepcopy(hdul[0].header)

    # remove 3D attributes from header
    for attribute in list(map_intg[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '3'):
                del map_intg[0].header[attribute]

    map_intg[0].header['NAXIS']=2
    map_intg[0].header['BUNIT'] = 'K km/s (Tmb)'

    axis = 3
    vel = get_axis(3, hdul)/1.0e3

    ch_maps = []

    map_num = int((ch_max-ch_min)/ch_res)

    for idx_map in range(map_num):

        if unit == 'channel':

            idx_min = ch_min+(ch_res*idx_map)
            idx_max = ch_min+(ch_res*(idx_map+1))

            for idx_dec in range(len(map_intg[0].data[:,0])):
                for idx_ra in range(len(map_intg[0].data[0,:])):

                    map_intg[0].data[idx_dec, idx_ra]=np.trapz(hdul[0].data[idx_min:idx_max, idx_dec, idx_ra],\
                                                               vel[idx_min:idx_max])


        elif unit == 'velocity':

            vel_min = ch_min+(ch_res*idx_map)
            vel_max = ch_min+(ch_res*(idx_map+1))

            ch_num = int((vel_max-vel_min)/(hdul[0].header["CDELT3"]/1.0e3))

            if ch_num == 0:

                ch_vel = [[vel_min, vel_max]]

            else:

                vel_res = (vel_max - vel_min)/ch_num

                ch_vel = np.arange(vel_min, vel_max, vel_res)

                ch_vel = np.append(ch_vel, vel_max)

            for idx_dec in range(len(map_intg[0].data[:,0])):
                for idx_ra in range(len(map_intg[0].data[0,:])):

                    temp = np.interp(ch_vel, vel, hdul[0].data[:, idx_dec, idx_ra])

                    map_intg[0].data[idx_dec, idx_ra] = np.trapz(temp, ch_vel)

        ch_maps.append(copy.deepcopy(map_intg))

    return ch_maps


#############################################################
#
# the function "moment_N()" determines the moment of order N
#
# function input:
#
# order: the order of the moment
# hdul: the spectral cube
# the integrated noise level of the spectral cube
# (only important for N>0, the default value is set to 1e-6)
#
#############################################################

def moment_N(order, hdul, noise_level = 0):

    moment_0 = astrokit.zeros_map(hdul)

    # determine velocity axis
    grid_axis=3
    vel = get_axis(grid_axis, hdul)/1e3

    if order > 0:

        moment_1 = astrokit.zeros_map(hdul)
        moment_N = astrokit.zeros_map(hdul)

    # determine moment 0 and 1
    for idx_dec in range(len(moment_0[0].data[:,0])):
        for idx_ra in range(len(moment_0[0].data[0,:])):

            moment_0[0].data[idx_dec, idx_ra]=\
            np.trapz(hdul[0].data[:, idx_dec, idx_ra], vel[:])

            moment_min = noise_level

            if (order>0):

                #if not np.isnan(moment_0[0].data[idx_dec,idx_ra])\
                #and (abs(moment_0[0].data[idx_dec,idx_ra])>moment_min):

                # determine weighted velocity
                moment_wei = hdul[0].data[:,idx_dec, idx_ra]*vel[:]

                # determine moment 1
                moment_1[0].data[idx_dec,idx_ra] = np.trapz(moment_wei, vel)\
                                                 / moment_0[0].data[idx_dec, idx_ra]

                if order > 1:
                    # determine moment N weight
                    moment_wei =  hdul[0].data[:, idx_dec, idx_ra] \
                               * (vel[:] - moment_1[0].data[idx_dec, idx_ra])**order


                    # determine moment N
                    moment_N[0].data[idx_dec,idx_ra] = np.trapz(moment_wei, vel)\
                                                     / moment_0[0].data[idx_dec, idx_ra]

                #else:
                #
                #        moment_1[0].data[idx_dec, idx_ra] = np.nan
                #        moment_N[0].data[idx_dec, idx_ra] = np.nan

    # determine moment N
    if order == 0:

        return moment_0

    elif order == 1:

        return moment_1

    else:

        return moment_N

#############################################################
#
# the function "pdf()" determines the number of pixel
# per intensity bin of a given map
#
# function input:
#
# bin_size: bin size of the intensity
# bin_min, bin_max: the intensity range [min, max]
# hdul: the spectral cube
#
#############################################################

def histogram(grid, hdul):

    pixel_count = np.zeros([len(grid)])

    idx_max = len(grid)-1
    idx_min = 0

    for dec in range(len(hdul[0].data[:,0])):
        for ra in range(len(hdul[0].data[0,:])):
            if not (np.isnan(hdul[0].data[dec,ra])
                    or np.isinf(hdul[0].data[dec,ra])
                    or hdul[0].data[dec,ra]==0):

                idx = get_idx(hdul[0].data[dec,ra], grid)

                if (( idx < idx_max) & (idx >= idx_min)):
                    pixel_count[idx] += 1

    return pixel_count

#############################################################
#
# the function "chop_cube()" cuts out a given interval
# in the velocity axis of a spectral cube
#
# function input:
#
# vel_min, vel_max: define the velocity range [min, max]
# hdul_inp: the spectral cube
#
#############################################################

def chop_cube(vel_min, vel_max, hdul_inp):

    # determine velocity axis
    grid_axis=3
    vel = get_axis(grid_axis, hdul_inp)/1e3

    idx_min = get_idx(vel_min, vel, 'closer')
    idx_max = get_idx(vel_max, vel, 'closer')

    empty_cube=np.zeros_like(hdul_inp[0].data[idx_min:idx_max,:,:])
    hdu_chop = fits.PrimaryHDU(empty_cube)
    hdul_chop = fits.HDUList([hdu_chop])
    hdul_chop[0].header=copy.deepcopy(hdul_inp[0].header)

    axis_len  = len(vel[idx_min : idx_max+1])

    ref_pos   = hdul_inp[0].header["CRPIX3"]
    step_size = hdul_inp[0].header["CDELT3"]
    ref_value = vel[idx_min] * 1e3 - (1. - ref_pos) * step_size

    hdul_chop[0].header["NAXIS3"] = axis_len
    hdul_chop[0].header["CRVAL3"] = ref_value

    hdul_chop[0].data=hdul_inp[0].data[idx_min:idx_max+1,:,:]

    return hdul_chop

#############################################################
#
# the function "rms_spectrum()" determines the
# root mean square of a given spectrum
#
# function input:
#
# window: the velocitx range of the emission [min, max]
# vel: the velocity axis of the rms_spectrum
# spec: the intensity of the spectrum
#
#############################################################

def rms_spectrum(vel, spect, window=None, rms_range=None):

    # set nan and inf to number in spectrum
    spect = np.nan_to_num(spect)

    if window:
        # genarate an empty array
        # with the same size as the velocity axis
        rms_mask=np.zeros_like(vel)

        # copy the spectral data
        spect_noise=copy.deepcopy(spect)

        # generate a mask at the location of the windwos/emission
        for win_idx in range(0, len(window), 2):

            start_idx = get_idx(window[win_idx], vel, 'closer')
            end_idx   = get_idx(window[win_idx+1], vel, 'closer')

            rms_mask[start_idx:end_idx]=1

        spect_mask = ma.masked_array(spect, rms_mask)

        # set the emission in the spectrum to 0
        spect_noise[spect_mask.mask]=0.0

        # determine the rms of the noise
        rms_noise = np.sqrt(sum(spect_noise**2)/(len(spect_noise)-sum(rms_mask)))

    elif range:

        idx_min = get_idx(rms_range[0], vel, 'closer')
        idx_max = get_idx(rms_range[1], vel, 'closer')

        rms_noise = np.sqrt(sum(spect[idx_min:idx_max]**2)/(len(spect[idx_min:idx_max])))

    return rms_noise

def rms_map(hdul, window=None, rms_range=None):

    # make an empty map
    map_size = np.zeros_like(hdul[0].data[0,:,:])
    hdu = fits.PrimaryHDU(map_size)
    hdul_rms = fits.HDUList([hdu])
    hdul_rms[0].header = copy.deepcopy(hdul[0].header)

    # remove 3d attributes from header
    for attribute in list(hdul_rms[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '3'):
                del hdul_rms[0].header[attribute]
            elif (attribute == 'WCSAXES'):
                hdul_rms[0].header['WCSAXES'] = 2

    hdul_rms[0].header['NAXIS']=2

    axis=3
    vel = get_axis(axis, hdul)/1e3

    len_ax1 = len(hdul_rms[0].data[:,0])
    len_ax2 = len(hdul_rms[0].data[0,:])

    for idx1 in range(len_ax1):
        for idx2 in range(len_ax2):

            hdul_rms[0].data[idx1, idx2] = rms_spectrum(vel,
                                                        hdul[0].data[:, idx1, idx2],
                                                        window,
                                                        rms_range)

    return hdul_rms

def rms2weight(rms_map, min_lim = 1e-9):

    weight_map = astrokit.zeros_map(rms_map)

    weight_map[0].data = 1./rms_map[0].data**2

    weight_map[0].data[rms_map[0].data<min_lim]=0.0

    return weight_map

def gauss_weight(pos_ra, pos_dec, width, sky_map):

    weight = astrokit.zeros_map(sky_map)

    grid2d_ra, grid2d_dec = get_grid(sky_map)

    len_ra = sky_map[0].header['NAXIS1']
    len_dec = sky_map[0].header['NAXIS2']

    for dec in range(len_dec):
        for ra in range(len_ra):

            weight[0].data[dec, ra] =  curve.gauss_2d(width,
                                                      pos_ra,
                                                      pos_dec,
                                                      grid2d_ra[dec, ra],
                                                      grid2d_dec[dec, ra])

    return weight

def rm_borders(hdul, wei_map, wei_min):

    hdul_new = copy.deepcopy(hdul)
    dim=hdul[0].header['NAXIS']

    if dim == 3:
        hdul_new[0].data[:, wei_map[0].data<wei_min] = np.nan
    elif dim ==2:
        hdul_new[0].data[idx_dec, wei_map[0].data<wei_min] = np.nan
    return hdul_new

def noise_intensity(rms = None,
                    vel_res = None,
                    vel_min = 0,
                    vel_max = 0,
                    vel = None,
                    spect = None,
                    variable_rms = False):


    if not vel_res:
        vel_res = abs(vel[1]-vel[0])

    ch_num = (vel_max - vel_min)/vel_res + 1.

    if variable_rms:

        intensity = vel_res*np.sqrt(np.sum(rms**2))

    else:

        if rms == None:

            rms = rms_spectrum(vel,
                               spect,
                               window = None,
                               rms_range = [vel_min, vel_max])


        intensity = rms * np.sqrt(ch_num) * vel_res



    return intensity

def noise_intensity_map(vel_min, vel_max, hdul):

    noise_map = astrokit.zeros_map(hdul)

    axis=3
    vel = get_axis(axis, hdul)/1e3

    vel_res = abs(vel[1]-vel[0])

    len_ax1 = len(noise_map[0].data[:,0])
    len_ax2 = len(noise_map[0].data[0,:])

    for idx1 in range(len_ax1):
        for idx2 in range(len_ax2):


            rms = rms_spectrum(vel,
                               hdul[0].data[:, idx1, idx2],
                               window = None,
                               rms_range = [vel_min, vel_max])

            noise_map[0].data[idx1, idx2] = noise_intensity(rms,
                                                            vel_res,
                                                            vel_min = vel_min,
                                                            vel_max = vel_max)

    return noise_map

#############################################################
#
# the function "average_cube()" determines the averaged
# spectrum of a spectral cube
#
# function input:
#
# hdul: the spectral cube
#
#############################################################

def average_cube(hdul_inp, weight = None):

    spect_size = np.zeros_like(hdul_inp[0].data[:,0,0])
    hdu = fits.PrimaryHDU(spect_size)
    spect_aver = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    spect_aver[0].header = copy.deepcopy(hdul_inp[0].header)

    # remove 3D and 2D attributes from header
    for attribute in list(hdul_inp[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '3'):
                spect_aver[0].header[attribute[:-1]+'1'] = hdul_inp[0].header[attribute[:-1]+'3']
            elif (attribute == 'WCSAXES'):
                spect_aver[0].header['WCSAXES'] = 1

    #spect_aver[0].header['CTYPE1'] = hdul_inp[0].header['CTYPE3']
    #spect_aver[0].header['CRVAL1'] = hdul_inp[0].header['CRVAL3']
    #spect_aver[0].header['CDELT1'] = hdul_inp[0].header['CDELT3']
    #spect_aver[0].header['CRPIX1'] = hdul_inp[0].header['CRPIX3']
    #spect_aver[0].header['CROTA1'] = hdul_inp[0].header['CROTA3']

    # remove 3D and 2D attributes from header
    for attribute in list(spect_aver[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '2') or (attribute[-1] == '3'):
                del spect_aver[0].header[attribute]

    spect_aver[0].header['NAXIS']=1

    # determine velocity axis
    grid_axis=3
    vel = get_axis(grid_axis, hdul_inp)/1e3

    hdul_inp[0].data=np.nan_to_num(hdul_inp[0].data)

    len_ax1 = len(hdul_inp[0].data[0,:,0])
    len_ax2 = len(hdul_inp[0].data[0,0,:])

    if weight:

        weight_sum=0.

        for idx1 in range(len_ax1):
            for idx2 in range(len_ax2):
                if not (weight[0].data[idx1, idx2] == 0):

                    weight_sum += weight[0].data[idx1, idx2]
                    spect_aver[0].data[:] += hdul_inp[0].data[:, idx1, idx2] * weight[0].data[idx1, idx2]

        spect_aver[0].data = spect_aver[0].data/weight_sum

    else:

        spect_count=0.

        for idx1 in range(len_ax1):
            for idx2 in range(len_ax2):
                if not (np.sum(hdul_inp[0].data[:, idx1, idx2])==0):
                    spect_count += 1.
                    spect_aver[0].data[:] += hdul_inp[0].data[:, idx1, idx2]

        spect_aver[0].data = spect_aver[0].data/spect_count


    return spect_aver

def average_volume(hdul_inp,
                   pos,
                   shape,
                   radius = None,
                   radius_2 = None,
                   width = None,
                   height = None,
                   weight = None):

    spect_size = np.zeros_like(hdul_inp[0].data[:,0,0])
    hdu = fits.PrimaryHDU(spect_size)
    spect_aver = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    spect_aver[0].header = copy.deepcopy(hdul_inp[0].header)

    # remove 3D and 2D attributes from header
    for attribute in list(spect_aver[0].header.keys()):
        if not (attribute == ''):
            if (attribute[-1] == '3') or (attribute[-1] == '2'):
                del spect_aver[0].header[attribute]

    spect_aver[0].header['NAXIS']=1

    hdul_inp[0].data=np.nan_to_num(hdul_inp[0].data)

    grid2d_ra, grid2d_dec = get_grid(hdul_inp)

    if shape == "circle" or shape == "sphere":
        # determie the distance from the line points to every grid point
        if shape == "circle":
            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2+(grid2d_ra-pos[0])**2)

        elif shape == "sphere":
            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2+((grid2d_ra-pos[0])*np.cos(Angle(pos[1]*u.deg)) )**2)


        if radius_2:

            grid2d_sig3=grid2d_r[np.where( np.logical_and(grid2d_r>=radius, grid2d_r<=radius_2))]

        else:
            # find relavant coordinat values
            grid2d_sig3=grid2d_r[np.where( grid2d_r <= radius )]

        bool_sig3 = np.isin(grid2d_r, grid2d_sig3)
        idx_sig3=np.asarray(np.where(bool_sig3))

        if weight:

            spect_aver[0].data = np.average(hdul_inp[0].data[:,idx_sig3[0, :], idx_sig3[1, :]],
                                            weights = weight[0].data[idx_sig3[0, :], idx_sig3[1, :]],
                                            axis = 1)



        else:

            spect_aver[0].data = np.average(hdul_inp[0].data[:, idx_sig3[0, :], idx_sig3[1, :]],
                                            axis = 1)

    else:
        print("error: no valid shape entered")

    return spect_aver


def interp_spect(vel, line, vel_interp):

    line_interp = np.zeros_like(vel_interp)

    for idx in range(len(vel_interp)):
        line_interp[idx] = get_value(vel_interp[idx], vel, line)

    return line_interp

def average_spectra(spectra, wei):

    aver_spect = np.zeros_like(spectra[0][:])

    for line in range(len(spectra)):

        aver_spect = aver_spect + spectra[line][:]*wei[line]

    aver_spect = aver_spect/np.sum(wei)

    return aver_spect

def area_size(hdul_inp, pos, dist, shape, radius = None, radius_2 = None,
              width = None, height = None, dist_err = 0):

    res_ra = abs(hdul_inp[0].header["CDELT1"])
    res_dec = abs(hdul_inp[0].header["CDELT2"])

    pix_size = Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad

    grid2d_ra, grid2d_dec = get_grid(hdul_inp)

    if shape == "circle" or shape == "sphere":

        # determie the distance from the line points to every grid point
        if shape == "circle":

            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2 + (grid2d_ra-pos[0])**2)

        elif shape == "sphere":

            grid2d_r = np.sqrt((grid2d_dec-pos[1])**2 + ((grid2d_ra-pos[0])*np.cos(Angle(pos[1]*u.deg)))**2)


        if radius_2:

            grid2d_sig3=grid2d_r[np.where(np.logical_and(grid2d_r>=radius, grid2d_r<=radius_2))]

        else:
            # find relavant coordinat values
            grid2d_sig3=grid2d_r[np.where( grid2d_r <= radius )]

        bool_sig3 = np.isin(grid2d_r, grid2d_sig3)
        idx_sig3=np.asarray(np.where(bool_sig3))

        shape_size = dist**2 * pix_size * len(idx_sig3[0][:])

        shape_size_err = 2. * dist * dist_err * pix_size * len(idx_sig3[0][:])

################################################################################
#        shape_size = dist**2\
#                   * np.sum(Angle( abs((grid2d_dec[idx_sig3[0][:]+1, idx_sig3[1][:]+1]-
#                                        grid2d_dec[idx_sig3[0][:]-1, idx_sig3[1][:]-1])/2.)*u.deg).rad*
#                            Angle( abs((grid2d_ra[idx_sig3[0][:]+1, idx_sig3[1][:]+1]-
#                                        grid2d_ra[idx_sig3[0][:]-1, idx_sig3[1][:]-1])/2.)*u.deg).rad*
#                            np.cos( Angle(grid2d_dec[idx_sig3[0][:], idx_sig3[1][:]]*u.deg).rad ))
#
#
#
#
#        shape_size_err = 2.*dist*dist_err\
#                       * np.sum(Angle( abs((grid2d_dec[idx_sig3[0][:]+1, idx_sig3[1][:]+1]-
#                                            grid2d_dec[idx_sig3[0][:]-1, idx_sig3[1][:]-1])/2.)*u.deg).rad*
#                                Angle( abs((grid2d_ra[idx_sig3[0][:]+1, idx_sig3[1][:]+1]-
#                                            grid2d_ra[idx_sig3[0][:]-1, idx_sig3[1][:]-1])/2.)*u.deg).rad*
#                                np.cos( Angle(grid2d_dec[idx_sig3[0][:], idx_sig3[1][:]]*u.deg).rad ))
#
################################################################################

    return shape_size, shape_size_err


def find_peak(vel, spect, peak_window):

    idx_st = get_idx(peak_window[0], vel)
    idx_end = get_idx(peak_window[1], vel)

    idx_peak = np.where(spect == np.max(spect[idx_st:idx_end]))

    return idx_peak

def find_wing(vel, spect, idx_peak, rms = 0, wing_hight_rel = 5, wing_hight_rms = 5, wing='red'):

    sigma = wing_hight_rms*rms

    if wing=="red":

        idx_red = idx_peak[0][len(idx_peak)-1]

        if (wing_hight_rel/100.*spect[idx_red]) > sigma:

            idx_wing = np.where(spect < (wing_hight_rel/100.*spect[idx_red]) )
        else:
            idx_wing = np.where(spect < sigma)

        idx=0
        while idx_red > idx_wing[0][idx]:
            idx+=1

        return idx_wing[0][idx]


    elif wing =="blue":

        idx_blue = idx_peak[0][0]

        if (wing_hight_rel/100.*spect[idx_blue])> sigma:

            idx_wing = np.where(spect < (wing_hight_rel/100.*spect[idx_blue]) )

        else:

            idx_wing = np.where(spect < sigma)

        idx=0
        while idx_red > idx_wing[0][idx]:
            idx+=1

        return idx_wing[0][idx-1]

    else:

        print("error: chose red or blue wing")



def pi_diagram(path, hdul, radius = None):

    intens = np.zeros_like(path[0,:])

    if radius:

        radius = Angle(radius*u.arcsec)

    else:

        radius = Angle(((hdul[0].header["BMAJ"]+hdul[0].header["BMIN"])/4.)*u.deg)


    grid2d_ax1, grid2d_ax2 = get_grid(hdul)

    for pos in range(len(intens)):

        # determie the distance from the line points to every grid point
        grid2d_r=np.sqrt((grid2d_ax2-path[1, pos])**2+(grid2d_ax1-path[0, pos])**2)

        # define relativ radius (3*sigma)
        sig3_r= radius.deg

        # find relavant coordinat values
        grid2d_sig3 = grid2d_r[np.where( grid2d_r < sig3_r )]
        bool_sig3 = np.isin(grid2d_r, grid2d_sig3)
        idx_sig3 = np.asarray(np.where(bool_sig3))

        weight_sum = 0.
        for idx_beam in range(len(idx_sig3[0,:])):

            weight = curve.gauss_2d(radius.deg, path[0, pos], path[1, pos], \
                           grid2d_ax1[idx_sig3[0,idx_beam],idx_sig3[1,idx_beam]], \
                           grid2d_ax2[idx_sig3[0,idx_beam],idx_sig3[1,idx_beam]])
            intens[pos] = intens[pos]+\
            hdul[0].data[idx_sig3[0,idx_beam],idx_sig3[1,idx_beam]]*weight
            weight_sum=weight_sum+weight

        intens[pos]=intens[pos]/weight_sum

    return intens


def pv_diagram(path,
               hdul,
               path_dist,
               radius = None):

    #spectra=np.zeros([len(path[0,:]),len(hdul[0].data[:,0,0])])

    axis=3
    vel = get_axis(axis, hdul)/1e3

    diagram_size = np.zeros([len(vel), len(path_dist)])
    hdu = fits.PrimaryHDU(diagram_size)
    empty_diagram = fits.HDUList([hdu])

    empty_diagram[0].header['NAXIS1'] = len(path)

    ref_pos = 0.
    step_size = path_dist[1]-path_dist[0]

    empty_diagram[0].header['CTYPE1'] = 'DIST--pc'
    empty_diagram[0].header['CRVAL1'] = path_dist[0] - (1. - ref_pos) * step_size
    empty_diagram[0].header['CDELT1'] = step_size
    empty_diagram[0].header['CRPIX1'] = ref_pos

    empty_diagram[0].header['NAXIS1'] = hdul[0].header['NAXIS1']
    empty_diagram[0].header['CTYPE2'] = hdul[0].header['CTYPE3']
    empty_diagram[0].header['CRVAL2'] = hdul[0].header['CRVAL3']
    empty_diagram[0].header['CDELT2'] = hdul[0].header['CDELT3']
    empty_diagram[0].header['CRPIX2'] = hdul[0].header['CRPIX3']

    empty_diagram[0].header['OBJECT'] = hdul[0].header['OBJECT']
    empty_diagram[0].header['LINE'] = hdul[0].header['LINE']

    if radius:

        radius = Angle(radius*u.arcsec)

    else:

        radius = Angle(((hdul[0].header["BMAJ"]+hdul[0].header["BMIN"])/4.)*u.deg)


    grid2d_ax1, grid2d_ax2 = get_grid(hdul)

    for pos in range(len(path[0,:])):

        # determie the distance from the line points to every grid point
        grid2d_r=np.sqrt((grid2d_ax2-path[1, pos])**2+(grid2d_ax1-path[0, pos])**2)

        # define relativ radius (3*sigma)
        sig3_r = radius.deg

        # find relavant coordinat values
        grid2d_sig3=grid2d_r[np.where( grid2d_r < sig3_r )]
        bool_sig3 = np.isin(grid2d_r, grid2d_sig3)
        idx_sig3=np.asarray(np.where(bool_sig3))

        weight_sum=0.

        for idx_beam in range(len(idx_sig3[0,:])):

            weight=curve.gauss_2d(radius.deg, path[0,pos], path[1,pos], \
                                 grid2d_ax1[idx_sig3[0, idx_beam], idx_sig3[1, idx_beam]], \
                                 grid2d_ax2[idx_sig3[0, idx_beam], idx_sig3[1, idx_beam]])

            empty_diagram[0].data[:, pos] = empty_diagram[0].data[:, pos]+\
            hdul[0].data[:,idx_sig3[0,idx_beam],idx_sig3[1,idx_beam]]*weight

            weight_sum=weight_sum+weight


        empty_diagram[0].data[:, pos] = empty_diagram[0].data[:, pos]/weight_sum

    return empty_diagram

def correlation_velocity(hdul_i, hdul_ii):

    order = 1

    moment_1_i  = moment_N(order, hdul_i)

    moment_1_ii = moment_N(order, hdul_ii)

    len_map_dec = len(moment_1_i[0].data[:,0])

    len_map_ra = len(moment_1_i[0].data[0,:])

    axis=3

    vel_i = get_axis(axis, hdul_i)/1e3

    vel_ii = get_axis(axis, hdul_ii)/1e3

    len_vel_i = len(vel_i)

    len_vel_ii = len(vel_ii)

    corlation_map = np.zeros([len_vel_i, len_vel_ii])

    for idx_dec in range(len_map_dec):
        for idx_ra in range(len_map_ra):

            if  moment_1_i[0].data[idx_dec,idx_ra] >vel_i[0]\
            and moment_1_i[0].data[idx_dec,idx_ra] <vel_i[len_vel_i-1]\
            and moment_1_ii[0].data[idx_dec,idx_ra]>vel_i[0]\
            and moment_1_ii[0].data[idx_dec,idx_ra]<vel_ii[len_vel_ii-1]:

                idx_vel_i  = get_idx(moment_1_i[0].data[idx_dec,idx_ra], vel_i)
                idx_vel_ii = get_idx(moment_1_ii[0].data[idx_dec,idx_ra], vel_ii)

                corlation_map[idx_vel_i, idx_vel_ii] +=1

    return corlation_map


def velocity_difference(hdul_1, hdul_2):

    map_size = np.zeros_like(hdul_1[0].data)
    hdu = fits.PrimaryHDU(map_size)
    vel_diff = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    vel_diff[0].header = copy.deepcopy(hdul_1[0].header)

    vel_diff[0].data = hdul_1[0].data - hdul_2[0].data

    return vel_diff
