from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import time

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

from astropy import constants as const
#import gdr3bcg.bcg as bcg

from joblib import Parallel, delayed

import astrokit

#def galactic_distance(
#    obj_pos,
#    obj_dist, 
#    obj_dist_err, 
#    gc_dist = 8178, # Gravity Colaboration et al. 2019. gc_dist = 7900 (Milam et al. 2015) 
#    gc_dist_err = 22,
#    coor_frame = 'icrs'
#    ):

#    obj_coor = SkyCoord(
#        ra=obj_pos[0], 
#        dec=obj_pos[1], 
#        distance = obj_dist, 
#        frame = coor_frame
#        )

def galactic_plane_distance(obj_pos1, obj_pos2, obj_dist, obj_dist_err, gc_dist = 8178, coor_frame = 'icrs'):

    obj_coor = SkyCoord(ra=obj_pos1, dec=obj_pos2, distance = obj_dist, frame = coor_frame)

    #gc_dist = 8178
    #gc_dist = 7900 # used in Milam + 2015

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

#def galactic_distance(obj_pos1, obj_pos2, obj_dist, obj_dist_err, coor_frame = 'icrs'):

#    obj_coor = SkyCoord(ra=obj_pos1, dec=obj_pos2, distance = obj_dist, frame = coor_frame)

#    gc_dist = 8178
    #gc_dist = 7900

#    gc_dist_err = 22

#    gc2obj_const = gc_dist**2 + obj_dist.value**2 - 2. * gc_dist*obj_dist.value \
#                 * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

#    gc2obj_dist = np.sqrt(gc2obj_const)

#    gc2obj_err_gc = 2. * gc_dist - 2. * obj_dist.value * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

#    gc2obj_err_obj = 2. * obj_dist.value \
#                     - 2. * gc_dist * (np.sin(np.pi/2.-obj_coor.galactic.b.rad) * np.cos(obj_coor.galactic.l) + np.cos(np.pi/2.-obj_coor.galactic.b.rad))

#    gc2obj_err = np.sqrt(1./(4. * gc2obj_const) * ( gc2obj_err_gc**2 * gc_dist_err**2 \
#                                                   + gc2obj_err_obj**2 * obj_dist_err.value**2))

#    return gc2obj_dist, gc2obj_err

def plx2dist(parallax, parallax_err):

    if parallax > 1e-9:

        distance = 1./(parallax*1.0e-3)

        distance_err =  ((parallax_err*1.0e-3)/(parallax*1.0e-3)**2)

    else:

        distance = 0.0

        distance_err = 0.0

    return distance, distance_err

def gaia_search(

    cent_ra,
    cent_dec,
    radius = None,
    width = None,
    hight = None,
    frame = 'icrs',
    cone_search = False,
    
):

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    coord = SkyCoord(ra=cent_ra, dec=cent_dec, unit=(u.degree, u.degree), frame=frame)


    if cone_search:

        cone_search = Gaia.cone_search_async(coord, radius=u.Quantity(radius, u.deg))
        stars = cone_search.get_results()
    
    else:

        if radius is not None:
            
            radius = u.Quantity(radius, u.deg)
            stars = Gaia.query_object_async(coordinate=coord, radius=radius)
            
        else:
            
            width = u.Quantity(width, u.deg)
            height = u.Quantity(hight, u.deg)
            stars = Gaia.query_object_async(coordinate=coord, width=width, height=height)
            
    return stars
    

###########################################################################################
#
#def gaia_luminosity(
#    magnitud_g_band,
#    extinction_g_band,
#    distance,
#    temp_eff,
#    log_surface_gravity,
#    iron_abundace, 
#    alpha_enhancment = 0.,
    
#):
    
#    absolute_magnitud_g_band = magnitud_g_band+5.-5.*np.log10(distance)-extinction_g_band
#    stellar_properties=[temp_eff, log_surface_gravity, iron_abundace, alpha_enhancment]
#    bc=bcg.BolometryTable()
#    band_correction_g = bc.computeBc(stellar_properties)
    
#    luminosity = 10**(-0.4*(absolute_magnitud_g_band+band_correction_g-4.74)) 
    
#    return luminosity

##################################################################################################

def extract_stellar_info(stars): 

    stellar_info = []

    # stars.columns gives additional options
        
    for star in range(len(stars.columns['teff_val'])):
        
        if stars.columns['teff_val'][star]>0:
        
            stellar_info.append([
                
                stars.columns['source_id'][star], 
                stars.columns['ra'][star],
                stars.columns['ra_error'][star],
                stars.columns['dec'][star],
                stars.columns['dec_error'][star],
                stars.columns['parallax'][star],
                stars.columns['parallax_error'][star],
                stars.columns['teff_val'][star],
                stars.columns['lum_val'][star],
            ])

    return stellar_info


def star_properties(
        
        pos_ra, 
        pos_dec, 
        area_size, 
        coord_sys, 
        sp_type = 'all', 
        timeout=300
        
        ):

    customSimbad = Simbad()
    customSimbad.TIMEOUT = timeout
    customSimbad.add_votable_fields('sptype', 'parallax', 'plx')
    customSimbad.get_votable_fields()
    query_table = customSimbad.query_region(
        coord.SkyCoord(pos_ra, pos_dec, frame=coord_sys),
        radius=area_size)

    star_id      = []
    star_ra      = []
    star_dec     = []
    star_type    = []
    star_plx     = []
    star_plx_err = []

    check_list = isinstance(sp_type, list)

    if check_list:

        num_sp_types = len(sp_type)
    
    else:

        num_sp_types = 1

        sp_type = [sp_type]


    for obj in range(len(query_table)):

        star = query_table[obj]["SP_TYPE"]

        if star:

            #print(star[0])

            for sp_idx in range(num_sp_types):

            #if sp_type == 'all' or sp_type == chr(star[0]):
                if sp_type[sp_idx] == 'all' or sp_type[sp_idx] == star[0]:

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

#def uv_luminosity(temp, lum):
#
#    #Lsun = 3.839e33 # in erg/s

#    uv_lum=np.zeros_like(lum)

#    num_stars = len(temp)

#    show_progress = 0
#    for star in range(num_stars):
#        time_start = time.time()

        #wav_max  = np.log10(2897.8e-4/temp[star])
        #wav_st = wav_max-2
        #wav_end = wav_max+2
        #wav_step = (wav_end-wav_st)/1e6

        #wav = np.arange(wav_st, wav_end, wav_step)
        #flux = astrokit.black_body(
        #    wav,
        #    temp[star],
        #    input_unit='wavelength',
        #    system_of_units = 'cgs',
        #    input_scale = 'log10'
        #    )

        #integ_flux = np.trapz(flux, 10**wav)
        #print(integ_flux*1e-15)

#        integ_flux_const = 2*np.pi**4*const.k_B.cgs.value**4/15./const.h.cgs.value**3/const.c.cgs.value**2

#        integ_flux = integ_flux_const * temp[star]**4

        #print(integ_flux*1e-15)

#        wav_st = np.log10(910e-8)
#        wav_end = np.log10(2066e-8)
#        wav_step = (wav_end - wav_st)/1e6

#        wav_uv = np.arange(wav_st, wav_end, wav_step)

#        flux_uv = astrokit.black_body(
#            wav_uv,
#            temp[star],
#            input_unit='wavelength',
#            system_of_units = 'cgs',
#            input_scale = 'log10'
#            )

#        integ_flux_uv = np.trapz(flux_uv, 10**wav_uv)

        #print(integ_flux_uv*1e-15)

#        uv_lum[star] = (integ_flux_uv/integ_flux)*lum[star]

#        time_end = time.time()
#        if (star/(num_stars-1)*100. > show_progress):
#            astrokit.loop_time(star, num_stars, time_start, time_end)
#            show_progress = show_progress + 5

#    return uv_lum

def fuv_luminosity(temp, lum):

    #integ_flux_const = 2*np.pi**4*const.k_B.cgs.value**4/15./const.h.cgs.value**3/const.c.cgs.value**2
    integ_flux_const = const.sigma_sb.cgs.value/np.pi

    integ_flux = integ_flux_const * temp**4

    wav_st = 910e-8 #np.log10(910e-8) # in cm
    wav_end = 2066e-8 #np.log10(2066e-8) # in cm 
    #wav_step = (wav_end - wav_st)/1e6

    wav_uv = np.linspace(wav_st, wav_end, num = 100000)#np.arange(wav_st, wav_end, wav_step)

    flux_uv = astrokit.black_body(

        wav_uv,
        temp,
        input_unit='wavelength',
        system_of_units = 'cgs',
        input_scale = 'linear' #'log10'
    )

    #integ_flux_uv = np.trapz(flux_uv, 10**wav_uv)
    integ_flux_uv = np.trapz(flux_uv, wav_uv)

    uv_lum = (integ_flux_uv/integ_flux)*lum

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


def uv_sky(

    pos_ra,
    pos_dec,
    pos_radius,
    coord_sys,
    sp_type,
    grid_dist,
    hdul_inp = 0.0,
    size_ra = 0.0,
    size_dec =0.0,
    input_type = 'fits'
    ):

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

def stellar_flux(

    source_ra, # deg
    source_dec, # deg
    source_dist, # pc
    star_ra, # deg
    star_dec, # deg
    star_dist, # pc
    star_lum, # 
    optical_depth = 0.0, # extinction_fuv
    frame = 'icrs',

):

    star_coord = SkyCoord(
        ra = star_ra * u.deg,
        dec = star_dec * u.deg,
        distance = star_dist * u.pc,
        frame = frame
    )

    source_coord = SkyCoord(
        ra = source_ra*u.deg,
        dec = source_dec*u.deg,
        distance = source_dist*u.pc,
        frame = frame
    )

    star2source_dist = source_coord.separation_3d(star_coord)

    star_flux =  star_lum/(4.*np.pi*star2source_dist.cm**2)*np.exp(-optical_depth)

    return star_flux



def radiation_field(

    stars_pos,
    stars_dist,
    grid_dist,
    list_pos = None,
    grid_pos = None, # input is fits file
    coord_sys = 'icrs',
    stars_lum = None,
    stars_temp = None,
    extinction = False,
    num_cores = -1,
    lum_inp = 'bol', # define input luminosity 'bol' 
    flux_out = 'fuv' # define output flux

):

    num_stars = len(stars_pos)

    if (flux_out == 'fuv') and (lum_inp == 'bol'):

        print('determine the FUV part of stellar bol luminosity')
        stars_lum_fuv  = Parallel(n_jobs=num_cores)(delayed(astrokit.fuv_luminosity)(stars_temp[star], stars_lum[star]) for star in range(num_stars))

    else:

        print('Warrning: the assumped input luminosity is:' + lum_inp)
        stars_lum_fuv = stars_lum

    if list_pos is not None:

        if len(list_pos.shape) > 1:

            len_of_list = len(list_pos)
            
            #uv_field = np.zeros(len_of_list)
            
            grid_ra = np.zeros(len_of_list)
            grid_dec = np.zeros(len_of_list)

            print('get list of positions')

            grid_ra[:] = list_pos[:, 0]
            grid_dec[:] = list_pos[:, 1]

            stars_flux = np.zeros(len_of_list)

        else:

            len_of_list = 1

            stars_flux = 0

            print('get list of positions')

            grid_ra = [list_pos[0]]
            grid_dec = [list_pos[1]]

        flux_at_position = np.zeros(num_stars)
        for pos in range(len_of_list):

            if extinction:
                optical_depth = np.zeros(num_stars)
                print('determine extinction of the FUV-field')
                optical_depth = Parallel(n_jobs=num_cores)(delayed(astrokit.galactic_extinction)(grid_ra[pos], grid_dec[pos], grid_dist, stars_pos[star, 0], stars_pos[star, 1], stars_dist[star], frame = 'icrs', step_size = 0.1, cross_section_scattering = 7.5e-22, scattering_events = 0.9, cross_section_absorption = 8.0e-22) for star in range(num_stars))
                
                #print('min optical depth:', np.min(optical_depth))
                print('determine stellar flux at position: ', pos)
                flux_at_position  = Parallel(n_jobs=num_cores)(delayed(astrokit.stellar_flux)(grid_ra[pos], grid_dec[pos], grid_dist, stars_pos[star, 0], stars_pos[star, 1], stars_dist[star], stars_lum_fuv[star], optical_depth = optical_depth[star], frame = 'icrs') for star in range(num_stars))

            else:
                print('determine stellar flux at position: ', pos)
                flux_at_position  = Parallel(n_jobs=num_cores)(delayed(astrokit.stellar_flux)(grid_ra[pos], grid_dec[pos], grid_dist, stars_pos[star, 0], stars_pos[star, 1], stars_dist[star], stars_lum_fuv[star], optical_depth = 0.0, frame = 'icrs') for star in range(num_stars))

            if len_of_list == 1:

                stars_flux = np.sum(flux_at_position)
            else:

                stars_flux[pos] = np.sum(flux_at_position)

    else:
        
        stars_flux = np.zeros_like(grid_pos[0].data)

        print('get grid')
        grid_ra, grid_dec = astrokit.get_grid(grid_pos)

        center_dec_idx = grid_ra.shape[0] // 2
        center_ra_idx = grid_dec.shape[1] // 2


        grid_ra_center = grid_ra[center_dec_idx, center_ra_idx]
        grid_dec_center = grid_dec[center_dec_idx, center_ra_idx]
        #map_uv = astrokit.zeros_map(grid_inp)

        if extinction:
            optical_depth = np.zeros(num_stars)
            print('determine extinction of the FUV-field')
            optical_depth = Parallel(n_jobs=num_cores)(delayed(astrokit.galactic_extinction)(grid_ra_center, grid_dec_center, grid_dist, stars_pos[star, 0], stars_pos[star, 1], stars_dist[star], frame = 'icrs', step_size = 0.1, cross_section_scattering = 7.5e-22, scattering_events = 0.9, cross_section_absorption = 8.0e-22) for star in range(num_stars))
    
        print('set star positions')
        stars_coord = SkyCoord(
            ra = stars_pos[:, 0] * u.deg,
            dec = stars_pos[:, 1] * u.deg,
            distance = stars_dist * u.pc,
            frame = coord_sys
        )

        print('set grid position')
        grid_coord = SkyCoord(
            ra = grid_ra*u.deg,
            dec = grid_dec*u.deg,
            distance = grid_dist*u.pc,
            frame = coord_sys
        )

        show_progress = 0

        print('start to determine the UV-Field')
        for star in range(num_stars):

            time_start = time.time()

            grid2star = grid_coord.separation_3d(stars_coord[star])

            if extinction:
                stars_flux = stars_flux + stars_lum_fuv[star]/(4.*np.pi*grid2star.cm**2)*np.exp(-optical_depth[star])
            else:
                stars_flux = stars_flux + stars_lum_fuv[star]/(4.*np.pi*grid2star.cm**2)

            time_end = time.time()
            if (star/(num_stars-1)*100. > show_progress):
                astrokit.loop_time(star, num_stars, time_start, time_end)
                show_progress = show_progress + 5

    if list_pos is not None:

        return stars_flux           

    else:

        stars_flux_map = astrokit.zeros_map(grid_pos)

        stars_flux_map[0].data = stars_flux

        return stars_flux_map    
    


def get_uv_map_old(

    stars_pos,
    stars_dist,
    grid_dist,
    grid_inp,
    coord_sys,
    stars_lum = None,
    stars_temp = None,
    #stars_type = None,
    #stars_evo = None,
    num_cores = -1

):

    num_stars = len(stars_pos)

    ###############################################################
    #print('number of stars: '+str(num_stars))

    #if stars_lum is None:
    #    print('warning: no luminosity input, start to determine luminosity:')

    #    # Instantiate class object
    #    sdj = pyasl.SpecTypeDeJager()

    #    llum  = np.zeros(num_stars)
    #    lteff = np.zeros(num_stars)

    #    for star in range(num_stars):

    #        llum[star], lteff[star] = sdj.lumAndTeff(stars_type[star], stars_evo[star])

    #    stars_lum  = 10**llum * const.L_sun.cgs.value
    #    stars_temp = 10**lteff
    ###############################################################

    print('start dertermin FUV luminosity')
    #uv_lum = astrokit.uv_luminosity(stars_temp, stars_lum)
   
    stars_lum_uv  = Parallel(n_jobs=num_cores)(delayed(astrokit.uv_luminosity)(stars_temp[star], stars_lum[star]) for star in range(num_stars))

    map_uv = astrokit.zeros_map(grid_inp)

    print('get grid')
    grid_ra, grid_dec = astrokit.get_grid(map_uv)

    print('set star positions')
    star_pos = SkyCoord(
        ra = stars_pos[:, 0] * u.deg,
        dec = stars_pos[:, 1] * u.deg,
        distance = stars_dist * u.pc,
        frame = coord_sys
    )

    print('set grid position')
    grid_pos = SkyCoord(
        ra = grid_ra*u.deg,
        dec = grid_dec*u.deg,
        distance = grid_dist*u.pc,
        frame = coord_sys
    )

    show_progress = 0

    print('start to determine the UV-Field')
    for star in range(num_stars):

        time_start = time.time()

        grid2star = grid_pos.separation_3d(star_pos[star])

        uv_grid = stars_lum_uv[star]/(4.*np.pi*grid2star.cm**2)

        map_uv[0].data = map_uv[0].data + uv_grid

        time_end = time.time()
        if (star/(num_stars-1)*100. > show_progress):
            astrokit.loop_time(star, num_stars, time_start, time_end)
            show_progress = show_progress + 5

    return map_uv

def galactic_hydrogen_distribution( # Dickey and Lockman (1990)  galactic height hydrogen distribution

    z_height # in parsec
):

    density_term_i = 0.69*np.exp(-(z_height/127.)**2)
    density_term_ii = 0.189*np.exp(-(z_height/318.)**2)
    density_term_iii = 0.113*np.exp(-abs(z_height/403.))

    density_hydrogen = 0.566*(density_term_i + density_term_ii + density_term_iii) # in 1/cm^3

    return density_hydrogen

#def galactic_hydrogen_column(
#        
#        z_height_0,
#        z_height_1,
#        distance,
#        num_steps = 1000
#): # Dickey and Lockman (1990) 
#    
#    int_steps = np.linspace(0, 1, num_steps)
#    density_term_i = np.zeros_like(int_steps)
#    density_term_ii = np.zeros_like(int_steps)
#    density_term_iii = np.zeros_like(int_steps)
#    
#    density_term_i = 0.69*np.exp(-((z_height_0+int_steps*(z_height_1-z_height_0))/127.)**2)
#    density_term_ii = 0.189*np.exp(-((z_height_0+int_steps*(z_height_1-z_height_0))/318.)**2)
#    density_term_iii = 0.113*np.exp(-((z_height_0+int_steps*(z_height_1-z_height_0))/403.)**2)
#
#    density_hydrogen = 0.566*distance*(density_term_i + density_term_ii + density_term_iii) # in 1/cm^3
#
#    hydrogen_column = np.trapz(density_hydrogen, int_steps)
#
#    return hydrogen_column

def galactic_extinction(
    
    source_ra, # deg
    source_dec, # deg
    source_dist, # pc
    star_ra, # deg
    star_dec, # deg
    star_dist, # pc
    frame = 'icrs',
    step_size = 0.1, #pc
    cross_section_scattering = 7.5e-22,
    scattering_events = 0.9,
    cross_section_absorption = 8.0e-22,
    
):
    
    cross_section_total = cross_section_scattering *(1.-scattering_events) + cross_section_absorption

    # Create a SkyCoord objects in Equatorial (J2000) coordinates
    source_coord = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg, distance=source_dist*u.pc, frame=frame)
    star_coord = SkyCoord(ra=star_ra*u.deg, dec=star_dec*u.deg, distance=star_dist*u.pc, frame=frame)

    star2source_dist = source_coord.separation_3d(star_coord)

    num_steps = int(star2source_dist.value/step_size)+1

    # Convert to Galactic coordinates
    source_galactic_coord = source_coord.galactic
    star_galactic_coord = star_coord.galactic
    
    # Get the z-coordinate (height above the galactic plane)
    source_x, source_y, source_z = source_galactic_coord.cartesian.xyz
    star_x, star_y, star_z = star_galactic_coord.cartesian.xyz

    star2source_dist_z = source_z-star_z

    num_steps = int(np.abs(star2source_dist_z).value/step_size)+1
    #num_steps = int(star2source_dist.value/step_size)+1


    if  num_steps<2:

        volume_density = galactic_hydrogen_distribution(star_z.value)
        column_density = volume_density*star2source_dist.to(u.cm).value
        optical_depth = column_density*cross_section_total
        
    else:

        z_hight_steps = np.linspace(star_z.value, source_z.value, num_steps)
        #star2source_step = star2source_dist.to(u.cm).value/num_steps
        volume_density = galactic_hydrogen_distribution(z_hight_steps)
        column_density = np.trapz(volume_density, z_hight_steps)*star2source_dist.to(u.cm).value/star2source_dist_z.value
        optical_depth = np.abs(column_density*cross_section_total)

    return optical_depth
