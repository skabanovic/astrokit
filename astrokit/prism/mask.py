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

from scipy import signal
from astropy.io import fits
from astropy.coordinates import Angle



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
    grid_2d_ax1, grid_2d_ax2 = astrokit.get_grid(hdul_inp)

    res_ax1 = abs(hdul_inp[0].header["CDELT1"])
    res_ax2 = abs(hdul_inp[0].header["CDELT2"])

    len_func_ax1=int(round(dif_beam.deg/res_ax1+0.49)*4+1)
    len_func_ax2=int(round(dif_beam.deg/res_ax2+0.49)*4+1)

    conv_func = np.zeros([len_func_ax1, len_func_ax2])

    for i in range(len_func_ax1):
        for j in range(len_func_ax2):
            conv_func[i,j]=astrokit.gauss_2d(dif_beam.deg, 0.0, 0.0, res_ax1*(-int(len_func_ax1/2)+j),\
                                          res_ax2*(-int(len_func_ax2/2)+i))

    conv_func=conv_func/sum(sum(conv_func))

    input_dim = hdul_inp[0].header['NAXIS']

    if input_dim == 2:

        hdul_outp[0].data = signal.convolve2d(hdul_inp[0].data, conv_func, boundary='symm',mode='same')

        return hdul_outp

    elif input_dim == 3:

        len_ax3 = len(hdul_inp[0].data[:,0,0])

        for idx_ax3 in range(len_ax3):
            hdul_outp[0].data[idx_ax3,:,:] = \
            signal.convolve2d(hdul_inp[0].data[idx_ax3,:,:], conv_func, boundary='symm',mode='same')

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
    ref_pos[0] = (ref_value[0]-pos_ax1[0])/step_size[0]
    ref_pos[1] = (ref_value[1]-pos_ax2[0])/step_size[1]

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

    output_hdul[0].header['CRPIX1'] = (ref_value_ax1 - grid_ax1[pix_ax1[0]])/step_size_ax1

    output_hdul[0].header['CRPIX2'] = (ref_value_ax2 - grid_ax2[pix_ax2[0]])/step_size_ax2

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
    hdul_out[0].header["CRVAL3"] = vel_out[0] + ref_pos_inp*step_size_out

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
               beam_min = None):

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

    output_hdul[0].header["NAXIS1"] = len_ax1
    output_hdul[0].header["CDELT1"] = step_size_ax1
    output_hdul[0].header["CRVAL1"] = ref_value_ax1
    output_hdul[0].header["CRPIX1"] = (ref_value_ax1 - grid_ax1[0])/step_size_ax1

    output_hdul[0].header["NAXIS2"] = len_ax2
    output_hdul[0].header["CDELT2"] = step_size_ax2
    output_hdul[0].header["CRVAL2"] = ref_value_ax2
    output_hdul[0].header["CRPIX2"] = (ref_value_ax2 - grid_ax2[0])/step_size_ax2

    output_hdul[0].header["BMAJ"] = beam_maj
    output_hdul[0].header["BMIN"] = beam_min

    if ref_value_ax3 or ref_value_ax3 == 0:

        # step size of axis
        step_size_ax3   = grid_ax3[1] - grid_ax3[0]

        # value of reference grid position of axis
        ref_value_ax3   = ref_value_ax3

        output_hdul[0].header["NAXIS3"] = len_ax3
        output_hdul[0].header["CDELT3"] = step_size_ax3
        output_hdul[0].header["CRVAL3"] = ref_value_ax3
        output_hdul[0].header["CRPIX3"] = (ref_value_ax3 - grid_ax3[0])/step_size_ax3

    return output_hdul
