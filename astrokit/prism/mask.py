#############################################################
#
# mask.py includes function to equalize spectral cubes,
# e.g. interpolation to a reference grid,
# convolution to the same beam size, etc.
#
# Created by Slawa Kabanovic
#
#############################################################

import time
import copy
import scipy
import astropy
import astrokit

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from scipy import signal
from astropy.io import fits
from astropy.coordinates import Angle

import cygrid

def regrid_map(map_inp,
               map_ref,
               beam_new = None,
               beam_old = None,
               coor_inp = 'icrs',
               coor_ref = 'icrs'):

    if beam_new:

        beam_new = Angle(beam_new*u.arcsec)

        if beam_old:

            beam_inp = Angle(beam_old*u.arcsec)

        else:

            beam_inp = Angle(((map_inp[0].header['BMAJ'] + map_inp[0].header['BMIN'])/2.)*u.deg)

        kernel = Angle(np.sqrt(beam_new.deg**2-beam_inp.deg**2)*u.deg)

        if kernel < beam_inp/2.:

            print('Warning: The kernal size is small! Larger beam size is recommended')

    else:

        beam_new = Angle(((map_ref[0].header['BMAJ'] + map_ref[0].header['BMIN'])/2.)*u.deg)

        if beam_old:

            beam_inp = Angle(beam_old*u.arcsec)

        else:

            beam_inp = Angle(((map_inp[0].header['BMAJ'] + map_inp[0].header['BMIN'])/2.)*u.deg )

        if abs(beam_new.deg - beam_inp.deg) <1e-6:

            print('Warning: Both maps have the same beam size and a new beam size is not set! Half of the intial beam size is used for the Gaussian kernal')

            beam_new = Angle(np.sqrt(beam_ref.deg**2 + (beam_ref.deg/2.)**2)*u.deg)

            kernel = Angle((beam_ref.deg/2.)*u.deg)

        else:

            kernel = Angle(np.sqrt(beam_new.deg**2-beam_inp.deg**2)*u.deg)

        if kernel < beam_inp/2.:

            print('Warning: The kernal size is small! Larger beam size is recommended')



    # nan and inf are set to numbers (e.g. nan to 0)
    map_inp[0].data=np.nan_to_num(map_inp[0].data)

    # check reference grid dimension
    dim_ref = map_ref[0].header['NAXIS']

    # create an empty 2d hdulist
    if (dim_ref == 2):
        empty_map = np.zeros_like(map_ref[0].data)
    elif (dim_ref == 3):
        empty_map = np.zeros_like(map_ref[0].data[0,:,:])

    hdu_mask = fits.PrimaryHDU(empty_map)
    map_mask = fits.HDUList([hdu_mask])

    # the new (masked) hdulist gets the header of the input fits file
    map_mask[0].header = copy.deepcopy(map_inp[0].header)

    for attribute in list(map_ref[0].header.keys()):
        if not (attribute == ''):
            if ((attribute[-1] == '1') or  (attribute[-1] == '2') ):
                map_mask[0].header[attribute] = copy.deepcopy(map_ref[0].header[attribute])

    # the grid information in the header of the masked hdulist list are set
    # to the reference grid
    #map_mask[0].header['NAXIS1'] = copy.deepcopy(map_ref[0].header['NAXIS1'])
    #map_mask[0].header['CRPIX1'] = copy.deepcopy(map_ref[0].header['CRPIX1'])
    #map_mask[0].header['CDELT1'] = copy.deepcopy(map_ref[0].header['CDELT1'])
    #map_mask[0].header['CRVAL1'] = copy.deepcopy(map_ref[0].header['CRVAL1'])

    #map_mask[0].header['NAXIS2'] = copy.deepcopy(map_ref[0].header['NAXIS2'])
    #map_mask[0].header['CRPIX2'] = copy.deepcopy(map_ref[0].header['CRPIX2'])
    #map_mask[0].header['CDELT2'] = copy.deepcopy(map_ref[0].header['CDELT2'])
    #map_mask[0].header['CRVAL2'] = copy.deepcopy(map_ref[0].header['CRVAL2'])

    map_mask[0].header['BMAJ'] = beam_new.deg
    map_mask[0].header['BMIN'] = beam_new.deg

    if coor_ref != coor_inp:

        grid_1_trans, grid_2_trans = astrokit.get_grid(map_inp)

        coor_trans = SkyCoord(grid_1_trans, grid_2_trans, frame = coor_inp, unit='deg')

        if coor_ref == 'icrs':

            grid_1_inp = coor_trans.icrs.ra.deg
            grid_2_inp = coor_trans.icrs.dec.deg

        elif coor_ref == 'galactic':

            grid_1_inp = coor_trans.galactic.l.deg
            grid_2_inp = coor_trans.galactic.b.deg

    else:

        # determine 2d grid of the input map
        grid_1_inp, grid_2_inp = astrokit.get_grid(map_inp)

    # determine 2d grid of the reference map
    grid_1_ref, grid_2_ref = astrokit.get_grid(map_ref)

    len_ax1 = map_mask[0].header['NAXIS1']
    len_ax2 = map_mask[0].header['NAXIS2']

    #solid_angle = astrokit.get_solid_angle(map_inp)

    for ax1 in range(len_ax1):

        time_start = time.time()

        for ax2 in range(len_ax2):

            #grid2d_r = np.zeros_like(grid_1_inp)

            grid2d_r = np.sqrt((grid_1_inp - grid_1_ref[ax2, ax1])**2+((grid_2_inp - grid_2_ref[ax2, ax1])*np.cos(grid_2_ref[ax2, ax1]*np.pi/180.))**2)
            #grid2d_r = np.sqrt((grid_1_inp - grid_1_ref[ax2, ax1])**2+(grid_2_inp - grid_2_ref[ax2, ax1])**2)
            
            # define relativ radius (5*sigma)
            sig5_r = 5.*kernel.deg

            # find relavant coordinat values
            grid2d_sig5 = grid2d_r[np.where( grid2d_r < sig5_r )]

            bool_sig5 = np.isin(grid2d_r, grid2d_sig5)
            idx_sig5 = np.asarray(np.where(bool_sig5))

            #print(idx_sig5.shape)

            weight = astrokit.gauss_2d(kernel.deg,
                                       grid_1_ref[ax2, ax1],
                                       grid_2_ref[ax2, ax1],
                                       grid_1_inp[idx_sig5[0, :], idx_sig5[1, :]],
                                       grid_2_inp[idx_sig5[0, :], idx_sig5[1, :]],
                                       mode = 'FWHM',
                                       do_norm = False,
                                       amp = 1.)

            weight = weight/np.sum(weight)

            intensity = map_inp[0].data[idx_sig5[0, :], idx_sig5[1, :]]

            map_mask[0].data[ax2, ax1] = np.sum(intensity * weight)

        time_end = time.time()

        astrokit.loop_time(ax1, len_ax1, time_start, time_end)

    return map_mask

def regrid_spectral_cube(input_cube,
                         target_cube,
                         input_beam = None,
                         target_beam = None,
                         input_frame = 'icrs',
                         target_frame = 'icrs',
                         target_header = False):

    if input_beam:

        input_beam = input_beam/3600.

    else:

        input_beam = (input_cube[0].header['BMAJ'] + input_cube[0].header['BMIN'])/2.

    if target_beam:

        target_beam = target_beam/3600.

    else:

        target_beam = (target_cube[0].header['BMAJ'] + target_cube[0].header['BMIN'])/2.

    gridder = cygrid.WcsGrid(target_cube[0].header)

    if input_frame != target_frame:

        grid_ax1, grid_ax2 = astrokit.get_grid(input_cube)

        coords = SkyCoord(grid_ax1, grid_ax2, frame = input_frame, unit='deg')

        if target_frame == 'icrs':

            input_lon_grid = coords.icrs.ra.deg
            input_lat_grid = coords.icrs.dec.deg

        elif target_frame == 'galactic':

            input_lon_grid = coords.galactic.l.deg
            input_lat_grid = coords.galactic.b.deg

    else:

        input_lon_grid, input_lat_grid = astrokit.get_grid(input_cube)

    kernelsize_fwhm = np.sqrt(target_beam**2-input_beam**2)

    if (kernelsize_fwhm < 0.5*input_beam):

        print('Warning: the target beam is to small, thus the gaussian kernal is undersampled')

    # see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    kernelsize_sigma = kernelsize_fwhm / np.sqrt(8 * np.log(2))
    sphere_radius = 3. * kernelsize_sigma

    gridder.set_kernel(
        'gauss1d',
        (kernelsize_sigma,),
        sphere_radius,
        kernelsize_sigma / 2.
        )

    input_dimension = input_cube[0].header['NAXIS']

    input_ax1_len = input_cube[0].header['NAXIS1']
    input_ax2_len = input_cube[0].header['NAXIS2']

    target_ax1_len = target_cube[0].header['NAXIS1']
    target_ax2_len = target_cube[0].header['NAXIS2']

    # create an empty 2d hdulist
    if (input_dimension == 2):

        empty_cube=np.zeros([target_ax2_len,
                             target_ax1_len])

        gridder.grid(
            input_lon_grid.flatten(),
            input_lat_grid.flatten(),
            input_cube[0].data.flatten(),
            )

    elif (input_dimension == 3):

        input_ax3_len = input_cube[0].header['NAXIS3']

        empty_cube=np.zeros([input_ax3_len,
                             target_ax2_len,
                             target_ax1_len])

        gridder.grid(
        input_lon_grid.flatten(),
        input_lat_grid.flatten(),
        input_cube[0].data.reshape(input_ax3_len, input_ax2_len*input_ax1_len).transpose(),
        )


    hdu_mask = fits.PrimaryHDU(empty_cube)
    mask_cube = fits.HDUList([hdu_mask])

    if target_header:

        if input_dimension == 2:

            empty_map = astrokit.zeros_map(target_cube)

            mask_cube[0].header = copy.deepcopy(empty_map[0].header)

            for attribute in list(input_cube[0].header.keys()):
                if not (attribute == ''):
                    if (attribute == 'BUNIT'):
                        mask_cube[0].header['BUNIT'] = copy.deepcopy(input_cube[0].header['BUNIT'])

        elif input_dimension == 3:

            mask_cube[0].header = copy.deepcopy(target_cube[0].header)

            for attribute in list(input_cube[0].header.keys()):
                if not (attribute == ''):
                    if (attribute[-1] == '3'):
                        mask_cube[0].header[attribute] = copy.deepcopy(input_cube[0].header[attribute])
                    elif (attribute == 'BUNIT'):
                        mask_cube[0].header['BUNIT'] = copy.deepcopy(input_cube[0].header['BUNIT'])


    else :

        mask_cube[0].header = copy.deepcopy(input_cube[0].header)

        use_equinox = False

        for attribute in list(target_cube[0].header.keys()):
            if not (attribute == ''):
                if ( (attribute[-1] == '1')
                or   (attribute[-1] == '2')
                or   (attribute == 'RA')
                or   (attribute == 'DEC') ):

                    mask_cube[0].header[attribute] = copy.deepcopy(target_cube[0].header[attribute])

                elif  attribute == 'EQUINOX':

                    use_equinox = True

                    mask_cube[0].header[attribute] = copy.deepcopy(target_cube[0].header[attribute])


        if use_equinox:

            for attribute in list(input_cube[0].header.keys()):
                if not (attribute == ''):
                    if  attribute == 'EPOCH':
                        del mask_cube[0].header[attribute]


    mask_cube[0].header['BMAJ'] = target_beam
    mask_cube[0].header['BMIN'] = target_beam

    output_data = gridder.get_datacube()

    mask_cube[0].data = output_data

    return mask_cube

#############################################################
#
# the function "interp_map()" interpolates an integrated map
# to a reference grid:
#
# function input:
#
# hdul_inp: integrated (e.g. moment 0) map
# hdul_ref: reference grid/map
#
#############################################################

def interp_map(hdul_inp, hdul_ref):

    # nan and inf are set to numbers (e.g. nan to 0)
    hdul_inp[0].data=np.nan_to_num(hdul_inp[0].data)

    # check reference grid dimension
    dim_ref=hdul_ref[0].header['NAXIS']

    # create an empty 2d hdulist
    if (dim_ref==2):
        empty_map = np.zeros_like(hdul_ref[0].data)
    elif (dim_ref==3):
        empty_map = np.zeros_like(hdul_ref[0].data[0,:,:])

    hdu_mask = fits.PrimaryHDU(empty_map)
    hdul_mask = fits.HDUList([hdu_mask])

    # the new (masked) hdulist gets the header of the input fits file
    hdul_mask[0].header = copy.deepcopy(hdul_inp[0].header)

    # the grid information in the header of the masked hdulist list are set
    # to the reference grid
    hdul_mask[0].header['NAXIS1']=copy.deepcopy(hdul_ref[0].header['NAXIS1'])
    hdul_mask[0].header['CRPIX1']=copy.deepcopy(hdul_ref[0].header['CRPIX1'])
    hdul_mask[0].header['CDELT1']=copy.deepcopy(hdul_ref[0].header['CDELT1'])
    hdul_mask[0].header['CRVAL1']=copy.deepcopy(hdul_ref[0].header['CRVAL1'])

    hdul_mask[0].header['NAXIS2']=copy.deepcopy(hdul_ref[0].header['NAXIS2'])
    hdul_mask[0].header['CRPIX2']=copy.deepcopy(hdul_ref[0].header['CRPIX2'])
    hdul_mask[0].header['CDELT2']=copy.deepcopy(hdul_ref[0].header['CDELT2'])
    hdul_mask[0].header['CRVAL2']=copy.deepcopy(hdul_ref[0].header['CRVAL2'])

    # determine 2d grid of the input map
    grid_1_inp, grid_2_inp = astrokit.get_grid(hdul_inp)

    # determine 2d grid of the reference map
    grid_1_ref, grid_2_ref = astrokit.get_grid(hdul_ref)

    # interpolate the input grid data on the reference grid
    hdul_mask[0].data=scipy.interpolate.griddata((grid_1_inp.flatten(), grid_2_inp.flatten()),\
    hdul_inp[0].data.flatten(), (grid_1_ref, grid_2_ref), method='cubic')

    hdul_mask[0].data=np.nan_to_num(hdul_mask[0].data)

    return hdul_mask

#############################################################
#
# the function interp_cube interpolate an spectral cube
# to an reference grid:
#
# hdul_inp: inpud spectral cube (3D)
# hdul_ref: reference spectral map (2D grid)
# or cube 3D
#
#############################################################

def interp_cube(hdul_inp, hdul_ref):

    # nan and inf are set to numbers (e.g. nan to 0)
    hdul_inp[0].data = np.nan_to_num(hdul_inp[0].data)

    order = 0
    map_inp = astrokit.moment_N(order, hdul_inp)
    binary_map = astrokit.zeros_map(map_inp)
    binary_map[0].data[map_inp[0].data != 0] = 1

    # check dimension of the reference grid
    dim_ref=hdul_ref[0].header['NAXIS']

    # create an empty 3D hdulist
    if (dim_ref==2):
        empty_cube=np.zeros([len(hdul_inp[0].data[:,0,0]),len(hdul_ref[0].data[:,0]),len(hdul_ref[0].data[0,:])])
    else:
        empty_cube=np.zeros([len(hdul_inp[0].data[:,0,0]),len(hdul_ref[0].data[0,:,0]),len(hdul_ref[0].data[0,0,:])])


    hdu_mask = fits.PrimaryHDU(empty_cube)
    hdul_mask = fits.HDUList([hdu_mask])

    # the new (masked) hdulist gets the header of the input fits file
    hdul_mask[0].header=copy.deepcopy(hdul_inp[0].header)

    # the grid information in the header of the masked hdulist list are set
    # to the reference grid
    hdul_mask[0].header['NAXIS1'] = copy.deepcopy(hdul_ref[0].header['NAXIS1'])
    hdul_mask[0].header['CRPIX1'] = copy.deepcopy(hdul_ref[0].header['CRPIX1'])
    hdul_mask[0].header['CDELT1'] = copy.deepcopy(hdul_ref[0].header['CDELT1'])
    hdul_mask[0].header['CRVAL1'] = copy.deepcopy(hdul_ref[0].header['CRVAL1'])

    hdul_mask[0].header['NAXIS2'] = copy.deepcopy(hdul_ref[0].header['NAXIS2'])
    hdul_mask[0].header['CRPIX2'] = copy.deepcopy(hdul_ref[0].header['CRPIX2'])
    hdul_mask[0].header['CDELT2'] = copy.deepcopy(hdul_ref[0].header['CDELT2'])
    hdul_mask[0].header['CRVAL2'] = copy.deepcopy(hdul_ref[0].header['CRVAL2'])


    # determine 2d grid of the input map
    grid_1_inp, grid_2_inp = astrokit.get_grid(hdul_inp)

    # determine 2d grid of the reference map
    grid_1_ref, grid_2_ref = astrokit.get_grid(hdul_ref)

    # determine the number of velocity chanels of the map
    vel_len = len(hdul_mask[0].data[:,0,0])

    # the input grid is interpolated on the reference grid
    # for every velocity channel.

    binary_mask = interp_map(binary_map, hdul_ref)

    for idx_vel in range(vel_len):

        time_start = time.time()

        hdul_mask[0].data[idx_vel,:,:]=\
        scipy.interpolate.griddata((grid_1_inp.flatten(), \
                                    grid_2_inp.flatten()), \
                                    hdul_inp[0].data[idx_vel,:,:].flatten(), \
                                   (grid_1_ref, grid_2_ref), method='cubic')

        time_end = time.time()

        # checks the time remaining untill the 3D interpolation routine
        # is done.
        astrokit.loop_time(idx_vel, vel_len, time_start, time_end)

        hdul_mask[0].data[:, binary_mask[0].data < 0.9 ] = 0

    return hdul_mask

#############################################################
#
# the function convolve_spectra convolvs a cube or a map
# to a new beam size
#
# new_beam: the beam of the new map or cube
# hdul_inp: inpud spectral cube (3D) or map (2D)
#
#############################################################

def convolve_map(new_beam, hdul_inp):

    new_beam=Angle(new_beam*u.arcsec)
    old_beam=Angle(((hdul_inp[0].header['BMAJ']+hdul_inp[0].header['BMIN'])/2.)*u.deg)
    dif_beam=Angle(np.sqrt(new_beam**2-old_beam**2))

    # make an empty 2d grid with the same spatial
    # size as the input cube
    map_size = np.zeros_like(hdul_inp[0].data)
    hdu = fits.PrimaryHDU(map_size)
    hdul_outp = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    hdul_outp[0].header = copy.deepcopy(hdul_inp[0].header)

    hdul_outp[0].header['BMAJ']=new_beam.deg
    hdul_outp[0].header['BMIN']=new_beam.deg

    # determine coordinats
    #grid_2d_ax1, grid_2d_ax2 = astrokit.get_grid(hdul_inp)

    res_ax1 = abs(hdul_inp[0].header["CDELT1"])
    res_ax2 = abs(hdul_inp[0].header["CDELT2"])

    len_func_ax1=int(round(dif_beam.deg/res_ax1+0.49)*10+1)
    len_func_ax2=int(round(dif_beam.deg/res_ax2+0.49)*10+1)

    conv_func = np.zeros([len_func_ax1+1, len_func_ax2+1])

    for i in range(len_func_ax1+1):
        for j in range(len_func_ax2+1):
            conv_func[i, j]=astrokit.gauss_2d(dif_beam.deg, 0.0, 0.0, res_ax1*(-int(len_func_ax1/2)+j),\
                                          res_ax2*(-int(len_func_ax2/2)+i))

    conv_func = conv_func/np.sum(conv_func)
    #conv_func = np.flip(conv_func)

    print(conv_func.shape)

    input_dim = hdul_inp[0].header['NAXIS']

    if input_dim == 2:

        #hdul_outp[0].data = signal.convolve2d(hdul_inp[0].data, conv_func, boundary='symm', mode='same')
        hdul_outp[0].data = signal.convolve2d(hdul_inp[0].data, conv_func, mode='same', boundary='fill', fillvalue=0)

        return hdul_outp

    elif input_dim == 3:

        len_ax3 = len(hdul_inp[0].data[:,0,0])

        for idx_ax3 in range(len_ax3):
            hdul_outp[0].data[idx_ax3,:,:] = \
            signal.convolve2d(hdul_inp[0].data[idx_ax3,:,:], conv_func, mode='same', boundary='fill', fillvalue=0)
            #signal.convolve2d(hdul_inp[0].data[idx_ax3,:,:], conv_func, boundary='symm', mode='same')

        return hdul_outp

    else:

        print('error: input dimension must be 2 or 3')

def fix_dim(dim, hdul_inp):

    if (dim==3):

        cube=np.zeros([len(hdul_inp[0].data[0,:,0,0]),len(hdul_inp[0].data[0,0,:,0]),len(hdul_inp[0].data[0,0,0,:])])
        hdu_out = fits.PrimaryHDU(cube)
        hdul_out = fits.HDUList([hdu_out])
        hdul_out[0].header=copy.deepcopy(hdul_inp[0].header)

        hdul_out[0].data=copy.deepcopy(hdul_inp[0].data[0,:,:,:])

        # remove 4d attributes from header
        for attribute in list(hdul_out[0].header.keys()):
            if not (attribute == ''):
                if (attribute[-1] == '4'):
                    del hdul_out[0].header[attribute]

        hdul_out[0].header['NAXIS']=3

    elif (dim==2):
        map=np.zeros([len(hdul_inp[0].data[0,0,:,0]),len(hdul_inp[0].data[0,0,0,:])])
        hdu_out = fits.PrimaryHDU(map)
        hdul_out = fits.HDUList([hdu_out])
        hdul_out[0].header=copy.deepcopy(hdul_inp[0].header)

        hdul_out[0].data=copy.deepcopy(hdul_inp[0].data[0,0,:,:])

        # remove 4d and 3d attributes from header
        for attribute in list(hdul_out[0].header.keys()):
            if not (attribute == ''):
                if (attribute[-1] == '4') or (attribute[-1] == '3'):
                    del hdul_out[0].header[attribute]

        hdul_out[0].header['NAXIS']=2

    return hdul_out

def extract_subcube(hdul_inp,
                    pos_cent,
                    width,
                    height):

    # header information for axis = 1,2
    axis_len = np.zeros(2, dtype=int)
    ref_pos = np.zeros(2)
    step_size = np.zeros(2)
    ref_value = np.zeros(2)

    # the 4 position determining the location of the subcube
    pos_ax1 = np.zeros(2)
    pos_ax2 = np.zeros(2)

    # the index of the 4 position determining the location of the subcube
    idx_ax1 = np.zeros(2, dtype=int)
    idx_ax2 = np.zeros(2, dtype=int)

    # determine step size of the grid
    step_size[0] = hdul_inp[0].header["CDELT1"]
    step_size[1] = hdul_inp[0].header["CDELT2"]

    # determine the reference value of the grid
    ref_value[0] = hdul_inp[0].header["CRVAL1"]
    ref_value[1] = hdul_inp[0].header["CRVAL2"]

    # load axis=1,2 of the original large cube
    axis_1 = astrokit.get_axis(1, hdul_inp)
    axis_2 = astrokit.get_axis(2, hdul_inp)

    # determine the start and end position/index along axis=1
    pos_ax1[0] = pos_cent[0]+width/2.
    pos_ax1[1] = pos_cent[0]-width/2.

    idx_ax1[0] = astrokit.get_idx(pos_ax1[0], axis_1)
    idx_ax1[1] = astrokit.get_idx(pos_ax1[1], axis_1)

    pos_ax1[0] = axis_1[idx_ax1[0]]
    pos_ax1[1] = axis_1[idx_ax1[1]]

    # determine the start and end position/index along axis=2
    pos_ax2[0] = pos_cent[1]-height/2.
    pos_ax2[1] = pos_cent[1]+height/2.

    idx_ax2[0] = astrokit.get_idx(pos_ax2[0], axis_2)
    idx_ax2[1] = astrokit.get_idx(pos_ax2[1], axis_2)

    pos_ax2[0] = axis_2[idx_ax2[0]]
    pos_ax2[1] = axis_2[idx_ax2[1]]

    # determine the length of axis=1,2 of the new subcube
    axis_len[0] = idx_ax1[1]-idx_ax1[0]
    axis_len[1] = idx_ax2[1]-idx_ax2[0]

    # determine the reference position of axis=1,2
    ref_pos[0] = 1. - (pos_ax1[0] - ref_value[0])/step_size[0]
    ref_pos[1] = 1. - (pos_ax2[0] - ref_value[1])/step_size[1]

    dim_inp = hdul_inp[0].header['NAXIS']

    if dim_inp==3:

        empty_cube=hdul_inp[0].data[:, idx_ax2[0]:idx_ax2[1], idx_ax1[0]:idx_ax1[1]]

    elif dim_inp==2:

        empty_cube=hdul_inp[0].data[idx_ax2[0]:idx_ax2[1], idx_ax1[0]:idx_ax1[1]]

    else:
        print('error: input dimension must be 2 or 3')

    hdu_extract = fits.PrimaryHDU(empty_cube)
    hdul_extract = fits.HDUList([hdu_extract])
    hdul_extract[0].header=copy.deepcopy(hdul_inp[0].header)

    #length of data axis
    hdul_extract[0].header["NAXIS1"] = axis_len[0]

    hdul_extract[0].header["NAXIS2"] = axis_len[1]

    # reference grid position of axis
    hdul_extract[0].header["CRPIX1"] = ref_pos[0]

    hdul_extract[0].header["CRPIX2"] = ref_pos[1]

    # step size of axis
    hdul_extract[0].header["CDELT1"] = step_size[0]

    hdul_extract[0].header["CDELT2"] = step_size[1]

    # value of reference grid position of axis
    hdul_extract[0].header["CRVAL1"] = ref_value[0]

    hdul_extract[0].header["CRVAL2"] = ref_value[1]

    return hdul_extract

def chop_edges(input_hdul,
               pix_ax1,
               pix_ax2):

    input_dim = input_hdul[0].header['NAXIS']

    if input_dim == 3:

        cube_size = np.zeros_like(input_hdul[0].data[:, pix_ax2[0]:-pix_ax2[1], pix_ax1[0]:-pix_ax1[1]])

    elif input_dim == 2:

        cube_size = np.zeros_like(input_hdul[0].data[pix_ax2[0]:-pix_ax2[1], pix_ax1[0]:-pix_ax1[1]])

    hdu = fits.PrimaryHDU(cube_size)
    output_hdul = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    output_hdul[0].header = copy.deepcopy(input_hdul[0].header)


    if input_dim == 3:

        output_hdul[0].data = copy.deepcopy(input_hdul[0].data[:, pix_ax2[0]:-pix_ax2[1], pix_ax1[0]:-pix_ax1[1]])
        output_hdul[0].header['NAXIS2'] = len(output_hdul[0].data[0,:,0])
        output_hdul[0].header['NAXIS1'] = len(output_hdul[0].data[0,0,:])

    if input_dim == 2:

        output_hdul[0].data = copy.deepcopy(input_hdul[0].data[pix_ax2[0]:-pix_ax2[1], pix_ax1[0]:-pix_ax1[1]])
        output_hdul[0].header['NAXIS2'] = len(output_hdul[0].data[:,0])
        output_hdul[0].header['NAXIS1'] = len(output_hdul[0].data[0,:])

    # step size of axis
    step_size_ax1   = output_hdul[0].header['CDELT1']

    # value of reference grid position of axis
    ref_value_ax1   = output_hdul[0].header['CRVAL1']

    # step size of axis
    step_size_ax2   = output_hdul[0].header['CDELT2']

    # value of reference grid position of axis
    ref_value_ax2   = output_hdul[0].header['CRVAL2']

    axis = 1
    grid_ax1 = astrokit.get_axis(axis, input_hdul)

    axis = 2
    grid_ax2 = astrokit.get_axis(axis, input_hdul)

    output_hdul[0].header['CRPIX1'] = 1. - (grid_ax1[pix_ax1[0]] - ref_value_ax1)/step_size_ax1

    output_hdul[0].header['CRPIX2'] = 1. - (grid_ax2[pix_ax2[0]] - ref_value_ax2)/step_size_ax2

    return output_hdul

def resample(hdul_inp, vel_res):

    hdul_inp[0].data=np.nan_to_num(hdul_inp[0].data)

    axis_len_inp    = hdul_inp[0].header["NAXIS3"]
    ref_pos_inp     = hdul_inp[0].header["CRPIX3"]
    step_size_inp   = hdul_inp[0].header["CDELT3"]
    ref_value_inp   = hdul_inp[0].header["CRVAL3"]

    axis = 3
    vel_inp = astrokit.get_axis(axis, hdul_inp)

    step_size_out = vel_res*1e3

    vel_out = np.arange(vel_inp[0], vel_inp[axis_len_inp-1]+step_size_out, step_size_out)
    axis_len_out = len(vel_out)
    weight = np.zeros_like(vel_out)

    map_size = np.zeros([axis_len_out, len(hdul_inp[0].data[0,:,0]), len(hdul_inp[0].data[0,0,:])])
    hdu = fits.PrimaryHDU(map_size)
    hdul_out = fits.HDUList([hdu])
    hdul_out[0].header = copy.deepcopy(hdul_inp[0].header)


    hdul_out[0].header["NAXIS3"] = axis_len_out
    hdul_out[0].header["CDELT3"] = step_size_out
    hdul_out[0].header["CRVAL3"] = vel_out[0] - (1. - ref_pos_inp) * step_size_out

    if (step_size_inp >= step_size_out):

        print("error: new velocity resoltuin should be larger then the old velocity resolution")

    else:

        for idx_inp in range(axis_len_inp):


            idx_out = int(round((vel_inp[idx_inp]-vel_out[0])/step_size_out))

            #print("idx_out: " + str(idx_out))

            hdul_out[0].data[idx_out,:,:] += hdul_inp[0].data[idx_inp,:,:]

            weight[idx_out] += 1


        for idx_out in range(axis_len_out):

            hdul_out[0].data[idx_out,:,:] = hdul_out[0].data[idx_out,:,:]/ weight[idx_out]


    return hdul_out

def empty_grid(grid_ax1 = None,
               grid_ax2 = None,
               grid_ax3 = None,
               ref_value_ax1 = None,
               ref_value_ax2 = None,
               ref_value_ax3 = None,
               beam_maj = None,
               beam_min = None,
               CTYPE1 = 'RA---GLS',
               CTYPE2 = 'DEC--GLS',
               CTYPE3 = 'VRAD',
               EQUINOX = 2000):

    len_ax1 = len(grid_ax1)
    len_ax2 = len(grid_ax2)

    if ref_value_ax3 or ref_value_ax3 == 0:

        len_ax3 = len(grid_ax3)

        cube_size = np.zeros([len_ax3, len_ax2, len_ax1])

    else:

        cube_size = np.zeros([len_ax2, len_ax1])

    hdu = fits.PrimaryHDU(cube_size)
    output_hdul = fits.HDUList([hdu])

    # step size of axis
    step_size_ax1   = grid_ax1[1] - grid_ax1[0]

    # value of reference grid position of axis
    ref_value_ax1   = ref_value_ax1

    # step size of axis
    step_size_ax2   = grid_ax2[1] - grid_ax2[0]

    # value of reference grid position of axis
    ref_value_ax2   = ref_value_ax2

    output_hdul[0].header["CTYPE1"] = CTYPE1
    output_hdul[0].header["NAXIS1"] = len_ax1
    output_hdul[0].header["CDELT1"] = step_size_ax1
    output_hdul[0].header["CRVAL1"] = ref_value_ax1
    output_hdul[0].header["CRPIX1"] = 1. - (grid_ax1[0] - ref_value_ax1)/step_size_ax1

    output_hdul[0].header["CTYPE2"] = CTYPE2
    output_hdul[0].header["NAXIS2"] = len_ax2
    output_hdul[0].header["CDELT2"] = step_size_ax2
    output_hdul[0].header["CRVAL2"] = ref_value_ax2
    output_hdul[0].header["CRPIX2"] = 1. - (grid_ax2[0] - ref_value_ax2)/step_size_ax2

    output_hdul[0].header["BMAJ"] = beam_maj
    output_hdul[0].header["BMIN"] = beam_min

    output_hdul[0].header["EQUINOX"] = EQUINOX

    if ref_value_ax3 or ref_value_ax3 == 0:

        # step size of axis
        step_size_ax3   = grid_ax3[1] - grid_ax3[0]

        # value of reference grid position of axis
        ref_value_ax3   = ref_value_ax3

        output_hdul[0].header["CTYPE3"] = CTYPE3
        output_hdul[0].header["NAXIS3"] = len_ax3
        output_hdul[0].header["CDELT3"] = step_size_ax3
        output_hdul[0].header["CRVAL3"] = ref_value_ax3
        output_hdul[0].header["CRPIX3"] = 1. - (grid_ax3[0] - ref_value_ax3)/step_size_ax3

    return output_hdul

def empty_spectrum(grid_ax1,
                   unit = 'VRAD'):

    len_ax1 = len(grid_ax1)

    spectrum_size = np.zeros_like(grid_ax1)
    hdu = fits.PrimaryHDU(spectrum_size)
    output_hdul = fits.HDUList([hdu])

    # step size of axis
    step_size_ax1   = grid_ax1[1] - grid_ax1[0]

    # value of reference grid position of axis
    ref_value_ax1   = grid_ax1[0]

    output_hdul[0].header["CTYPE1"] = unit
    output_hdul[0].header["NAXIS1"] = len_ax1
    output_hdul[0].header["CDELT1"] = step_size_ax1
    output_hdul[0].header["CRVAL1"] = ref_value_ax1
    output_hdul[0].header["CRPIX1"] = 1. - (grid_ax1[0] - ref_value_ax1)/step_size_ax1

    return output_hdul

def zeros_map(hdul):

    dim = hdul[0].header['NAXIS']

    if dim == 3:
        # make an empty 2d grid with the same spatial
        # size as the input cube
        map_size = np.zeros_like(hdul[0].data[0,:,:])
        hdu = fits.PrimaryHDU(map_size)
        empty_map = fits.HDUList([hdu])

        # the output hdulist (map_intg) is geting the header information
        # of the input spectral cube ()hdul
        empty_map[0].header = copy.deepcopy(hdul[0].header)

        # remove 3D attributes from header
        for attribute in list(empty_map[0].header.keys()):
            if not (attribute == ''):
                if (attribute[-1] == '3'):
                    del empty_map[0].header[attribute]
                elif (attribute == 'WCSAXES'):
                    empty_map[0].header['WCSAXES'] = 2

        empty_map[0].header['NAXIS']=2

    elif dim == 2:

        map_size = np.zeros_like(hdul[0].data)
        hdu = fits.PrimaryHDU(map_size)
        empty_map = fits.HDUList([hdu])

        # the output hdulist (map_intg) is geting the header information
        # of the input spectral cube ()hdul
        empty_map[0].header = copy.deepcopy(hdul[0].header)

    return empty_map

def zeros_cube(hdul):

    cube_size = np.zeros_like(hdul[0].data)
    hdu = fits.PrimaryHDU(cube_size)
    empty_cube = fits.HDUList([hdu])

    # the output hdulist (map_intg) is geting the header information
    # of the input spectral cube ()hdul
    empty_cube[0].header = copy.deepcopy(hdul[0].header)

    return empty_cube

def merge_cubes(cube_1,
                cube_2,
                vel_range=None):


    if vel_range:

        cube_merge_1 = astrokit.chop_cube(vel_range[0], vel_range[1], cube_1)
        cube_merge_2 = astrokit.chop_cube(vel_range[0], vel_range[1], cube_2)

    else:

        cube_merge_1 = cube_1
        cube_merge_2 = cube_2



    axis = 3

    vel = astrokit.get_axis(axis, cube_merge_1)/1e3

    vel_res = cube_merge_1[0].header['CDELT3']/1e3

    vel_merge = np.arange(vel[0], vel[-1] + vel[-1]-vel[0] + 2.*vel_res, vel_res)

    axis = 1
    axis_ra = astrokit.get_axis(axis, cube_merge_1)

    axis = 2
    axis_dec = astrokit.get_axis(axis, cube_merge_1)

    cube_merge = astrokit.empty_grid(grid_ax1 = axis_ra,
                                     grid_ax2 = axis_dec,
                                     grid_ax3 = vel_merge*1.0e3,
                                     ref_value_ax1 = cube_merge_1[0].header['CRVAL1'],
                                     ref_value_ax2 = cube_merge_1[0].header['CRVAL2'],
                                     ref_value_ax3 = cube_merge_1[0].header['CRVAL3'],
                                     beam_maj = cube_merge_1[0].header['BMAJ'],
                                     beam_min = cube_merge_1[0].header['BMIN'])

    cube_merge[0].data[: cube_merge_1[0].header['NAXIS3'], :, :] = cube_merge_1[0].data
    cube_merge[0].data[cube_merge_1[0].header['NAXIS3'] :, :, :] = cube_merge_2[0].data

    cube_merge[0].header['VSHIFT'] = (vel[-1]-vel[0]+vel_res)*1.0e3

    return cube_merge
