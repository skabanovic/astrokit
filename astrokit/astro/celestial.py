from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

#from astropy.modeling.models import BlackBody1D
#from astropy.modeling.blackbody import FLAM
#from astropy.modeling.blackbody import FNU
import astropy.units as u

from astropy.coordinates import Angle
import astropy.coordinates as coord

from PyAstronomy import pyasl
from astropy.io import fits

import numpy as np
import copy

from astropy.wcs import WCS

from astrokit.prism import specube

def galactic_plane_distance(obj_pos1, obj_pos2, obj_dist, obj_dist_err, coor_frame = 'icrs'):

    obj_coor = SkyCoord(ra=obj_pos1, dec=obj_pos2, distance = obj_dist, frame = coor_frame)

    gc_dist = 8178
    #gc_dist = 7900

    gc_dist_err = 22

    gc2obj_const = gc_dist**2 + (obj_dist.value*np.cos(obj_coor.galactic.b))**2\
                 - 2. * gc_dist*obj_dist.value * np.cos(obj_coor.galactic.b) * np.cos(obj_coor.galactic.l)

    gc2obj_dist = np.sqrt(gc2obj_const)

    gc2obj_err_gc = 2. * gc_dist - 2. * obj_dist.value * np.cos(obj_coor.galactic.b) * np.cos(obj_coor.galactic.l)

    gc2obj_err_obj = 2. * obj_dist.value * np.cos(obj_coor.galactic.b)**2 \
                     - 2. * gc_dist * np.cos(obj_coor.galactic.b) * np.cos(obj_coor.galactic.l)

    gc2obj_err = np.sqrt(1./(4. * gc2obj_const) *(gc2obj_err_gc**2 * gc_dist_err**2 \
                                                   + gc2obj_err_obj**2*obj_dist_err.value**2))

    return gc2obj_dist, gc2obj_err

def galactic_distance(obj_pos1, obj_pos2, obj_dist, obj_dist_err, coor_frame = 'icrs'):

    obj_coor = SkyCoord(ra=obj_pos1, dec=obj_pos2, distance = obj_dist, frame = coor_frame)

    gc_dist = 8178
    #gc_dist = 7900

    gc_dist_err = 22

    gc2obj_const = gc_dist**2 + obj_dist.value**2 - 2. * gc_dist*obj_dist.value \
                 * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

    gc2obj_dist = np.sqrt(gc2obj_const)

    gc2obj_err_gc = 2. * gc_dist - 2. * obj_dist.value * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

    gc2obj_err_obj = 2. * obj_dist.value \
                     - 2. * gc_dist * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

    gc2obj_err = np.sqrt(1./(4. * gc2obj_const) * ( gc2obj_err_gc**2 * gc_dist_err**2 \
                                                   + gc2obj_err_obj**2 * obj_dist_err.value**2))

    return gc2obj_dist, gc2obj_err

def plx2dist(parallax, parallax_err):

    if parallax > 1e-9:

        distance = 1./(parallax*1.0e-3)

        distance_err =  ((parallax_err*1.0e-3)/(parallax*1.0e-3)**2)

    else:

        distance = 0.0

        distance_err = 0.0

    return distance, distance_err

def star_properties(pos_ra, pos_dec, area_size, coord_sys, sp_type = 'all'):

    customSimbad = Simbad()
    customSimbad.add_votable_fields('sptype', 'parallax', 'plx')
    customSimbad.get_votable_fields()
    query_table = customSimbad.query_region(coord.SkyCoord(pos_ra, pos_dec, frame=coord_sys),\
                                            radius=area_size)

    star_id      = []
    star_ra      = []
    star_dec     = []
    star_type    = []
    star_plx     = []
    star_plx_err = []

    for obj in range(len(query_table)):

        star = query_table[obj]["SP_TYPE"]

        if star:

            print(star[0])

            if sp_type == 'all' or sp_type == chr(star[0]):
            #if sp_type == 'all' or sp_type == star[0]:

                if not star_id[:] == query_table[obj]["MAIN_ID"]:

                    star_id.append(query_table[obj]["MAIN_ID"])
                    star_type.append(query_table[obj]["SP_TYPE"])
                    star_plx.append(query_table[obj]["PLX_VALUE"].item())
                    star_plx_err.append(query_table[obj]["PLX_ERROR"].item())

                    star_coord=SkyCoord(query_table[obj]["RA"].item(), query_table[obj]["DEC"].item(),\
                                        frame="icrs", unit=(u.hourangle, u.deg))


                    star_ra.append(star_coord.ra.deg)
                    star_dec.append(star_coord.dec.deg)

    return star_id, star_ra, star_dec, star_type, star_plx, star_plx_err


def fix_spectral_type(stars):

    star_type = [None]*len(stars[0][:])
    star_evo  = [None]*len(stars[0][:])

    for star_idx in range(len(stars[0][:])):

        star_type[star_idx]=stars[3][star_idx][0:2].decode("utf-8")

        len_str = len(stars[3][star_idx][:])
        check_str = ['I', 'V']

        if len_str>2:

            count_str = 2

            check_break = True

            if chr(stars[3][star_idx][count_str]) ==  check_str[0] \
            or chr(stars[3][star_idx][count_str]) ==  check_str[1] \
            or count_str == len_str-1:

                check_break = False

            while check_break:

                count_str += 1

                if chr(stars[3][star_idx][count_str]) ==  check_str[0] \
                or chr(stars[3][star_idx][count_str]) ==  check_str[1] \
                or count_str == len_str-1:

                    check_break = False


            num_str = len_str - count_str


            if num_str>0:

                star_evo[star_idx] = chr(stars[3][star_idx][-num_str])

                if num_str>1:

                    for idx in range(num_str-1):

                        if chr(stars[3][star_idx][-num_str+1+idx]) == '/':
                            break

                        star_evo[star_idx] += chr(stars[3][star_idx][-num_str+1+idx])

            else:
                star_evo[star_idx]='V'


        else:
            star_evo[star_idx]='V'



    return star_type, star_evo

def uv_luminosity(temp, lum):

    #Lsun = 3.839e33 # in erg/s

    uv_lum=np.zeros_like(lum)

    num_star = len(temp)

    for star in range(num_star):

        bb = BlackBody1D(temperature=temp[star]*u.K)

        wav_max  = 2897.8e4/temp[star]
        wav_st   = wav_max*1.0e-1
        wav_end  = wav_max*1.0e1
        wav_step = (wav_end-wav_st)/1e6

        wav = np.arange(wav_st, wav_end, wav_step) * u.AA
        flux = bb(wav).to(FLAM, u.spectral_density(wav))

        integ_flux = np.trapz(flux, wav)

        wav_uv = np.arange(910, 2066, wav_step) * u.AA
        flux_uv = bb(wav_uv).to(FLAM, u.spectral_density(wav_uv))

        integ_flux_uv = np.trapz(flux_uv, wav_uv)

        uv_lum[star] = (integ_flux_uv/integ_flux)*lum[star]

    return uv_lum

def sky_grid(cent_ra, cent_dec, size_ra, size_dec, res = 'nan'):

    cent_ax1 = Angle(cent_ra).deg
    cent_ax2 = Angle(cent_dec).deg

    size_ax1 = Angle(size_ra).deg
    size_ax2 = Angle(size_dec).deg

    if res == 'nan':

        if size_ax1>size_ax2:

            res = size_ax1/1.0e2

        else:

            res = size_ax2/1.0e2

    grid_ax1 = np.arange(cent_ax1-size_ax1/2., cent_ax1+size_ax1/2., res)

    grid_ax2 = np.arange(cent_ax2-size_ax2/2., cent_ax2+size_ax2/2., res)

    empty_grid = np.zeros([len(grid_ax2), len(grid_ax1)])

    return empty_grid, grid_ax1, grid_ax2


def uv_sky(pos_ra, pos_dec, pos_radius, coord_sys, sp_type, grid_dist,\
           hdul_inp = 0.0, size_ra = 0.0, size_dec =0.0, input_type = 'fits' ):

    stars   =  star_properties(pos_ra, pos_dec, pos_radius, coord_sys, sp_type)

    dist   = np.zeros([2, len(stars[5][:])])

    for star in range(len(stars[5][:])):

        dist[0, star], dist[1, star] = plx2dist(stars[4][star], stars[5][star])


    # Instantiate class object
    sdj = pyasl.SpecTypeDeJager()

    star_type, star_evo = fix_spectral_type(stars)

    num_stars = len(star_type)

    llum  = np.zeros(num_stars)
    lteff = np.zeros(num_stars)

    for star in range(num_stars):

        llum[star], lteff[star] = sdj.lumAndTeff(star_type[star], star_evo[star])

    lum  = 10**llum* 3.839e33
    temp = 10**lteff

    uv_lum = uv_luminosity(temp, lum)


    if input_type == 'fits':

        map_size = np.zeros_like(hdul_inp[0].data)
        hdu = fits.PrimaryHDU(map_size)
        hdul_uv = fits.HDUList([hdu])

        # the output hdulist (map_intg) is geting the header information
        # of the input spectral cube ()hdul
        hdul_uv[0].header = copy.deepcopy(hdul_inp[0].header)

        grid_ra, grid_dec = specube.get_grid(hdul_uv)

    elif input_type == 'area':

        uv_grid, grid_ra, grid_dec = sky_grid(pos_ra, pos_dec, size_ra, size_dec)

    else:

        print('error: the input type is incorrect or not supported.')

    star_pos = \
    SkyCoord(ra = stars[1][:]*u.deg, dec=stars[2][:]*u.deg,\
             distance = dist[0, :]*u.pc, frame = coord_sys)


    grid_pos = \
    SkyCoord(ra = grid_ra*u.deg, dec = grid_dec*u.deg,\
             distance = grid_dist*u.pc, frame = coord_sys)


    num_stars = len(stars[0][:])

    for star in range(num_stars):

        if dist[0, star] > 0.:

            grid2star = grid_pos.separation_3d(star_pos[star])

            uv_grid = uv_lum[star]/(4.*np.pi*grid2star.cm**2)

            hdul_uv[0].data = hdul_uv[0].data + uv_grid

    return hdul_uv
