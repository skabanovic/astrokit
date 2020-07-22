#############################################################
#
#  dendro_tools.py
#
#  Created by Slawa Kabanovic
#
#############################################################

import astrokit
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
import copy

#############################################################
#
# the function "find_leaf()"
#
# function input:
#
#
#############################################################

def find_struct(dendro, struct_type="leaf"):

    struct_num=0
    struct_idx = []
    struct_num = len(dendro)
    for strcut in range(struct_num):

        if (struct_type=="leaf" and dendro[strcut].is_leaf) or \
        (struct_type=="branch" and dendro[strcut].is_branch):

            struct_num+=1
            struct_idx.append(strcut)

    struct_idx = np.array(struct_idx)

    return struct_idx



def struct_average(hdul, dendro, weight_map = None,  struct_type="leaf", weight = False):

    if struct_type == "leaf" or struct_type == "branch":

        struct_idx = find_struct(dendro, struct_type)

    elif struct_type == "all":

        struct_idx = np.arange(len(dendro))

    ax_len = len(hdul[0].data[:,0,0])

    struct_num = len(struct_idx)

    struct_spect = np.zeros([struct_num, ax_len])

    for struct in range(struct_num):

        hdul_idx = dendro[struct_idx[struct]].indices(subtree=True)

        for ax in range(ax_len):

            if weight :

                struct_spect[struct, ax] = \
                np.sum(hdul[0].data[ax,hdul_idx[0][:],hdul_idx[1][:]]\
                *weight_map[0].data[hdul_idx[0][:],hdul_idx[1][:]])/np.sum(weight_map[0].data[hdul_idx[0][:],hdul_idx[1][:]])

            else:

                struct_spect[struct, ax] = \
                np.sum(hdul[0].data[ax,hdul_idx[0][:],hdul_idx[1][:]])/(len(hdul_idx[0][:]))

    return struct_spect


def struct_size(sky_map, dendro, dist, dist_err = 0, struct_type = "leaf"):

    res_ra = abs(sky_map[0].header["CDELT1"])
    res_dec = abs(sky_map[0].header["CDELT2"])

    pix_size = Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad

    grid2d_ra, grid2d_dec = astrokit.get_grid(sky_map)

    if struct_type == "leaf" or struct_type == "branch":

        struct_idx = find_struct(dendro, struct_type)

    elif struct_type == "all":

        struct_idx = np.arange(len(dendro))

    struct_num = len(struct_idx)

    area_size = np.zeros([struct_num])
    area_size_err = np.zeros([struct_num])

    for struct in range(struct_num):

        map_idx = dendro[struct_idx[struct]].indices(subtree=True)

        area_size[struct] = dist**2 * pix_size * len(map_idx[0][:])

        area_size_err[struct] = 2. * dist * dist_err * len(map_idx[0][:])

################################################################################
#
#        area_size[struct] = dist**2\
#                          * np.sum(Angle( abs((grid2d_dec[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                               grid2d_dec[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                   Angle( abs((grid2d_ra[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                               grid2d_ra[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                   np.cos( Angle(grid2d_dec[map_idx[0][:], map_idx[1][:]]*u.deg).rad ))
#
#
#
#        area_size_err[struct] = 2.*dist*dist_err\
#                              * np.sum(Angle( abs((grid2d_dec[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                   grid2d_dec[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                       Angle( abs((grid2d_ra[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                   grid2d_ra[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                       np.cos( Angle(grid2d_dec[map_idx[0][:], map_idx[1][:]]*u.deg).rad ))
#
################################################################################

    return area_size, area_size_err


def struct_mass(colden,
                colden_err,
                dendro,
                dist,
                dist_err = 0,
                abundance_ratio = 1.2e-4,
                struct_type = "leaf",
                line = 'cii'):

    hydro_mass = 1.6735575e-27

    res_ra = abs(colden[0].header["CDELT1"])
    res_dec = abs(colden[0].header["CDELT2"])

    grid2d_ra, grid2d_dec = astrokit.get_grid(colden)

    if struct_type == "leaf" or struct_type == "branch":

        struct_idx = find_struct(dendro, struct_type)

    elif struct_type == "all":

        struct_idx = np.arange(len(dendro))

    struct_num = len(struct_idx)

    mass_map = []
    mass_map_err = []

    for struct in range(struct_num):

        mass           = astrokit.zeros_map(colden)
        mass_err       = astrokit.zeros_map(colden)
        pixel_size     = astrokit.zeros_map(colden)
        pixel_size_err = astrokit.zeros_map(colden)

        map_idx = dendro[struct_idx[struct]].indices(subtree=True)

        pix_size =  Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad * dist**2

        pix_size_err =  Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad * 2.* dist * dist_err

        mass[0].data[map_idx[0][:], map_idx[1][:]] = (  pix_size
                                                      * colden[0].data[map_idx[0][:], map_idx[1][:]]*hydro_mass)/abundance_ratio

        mass_err[0].data[map_idx[0][:], map_idx[1][:]] = hydro_mass/abundance_ratio\
                                                       * np.sqrt(  pix_size_err**2
                                                                 * colden[0].data[map_idx[0][:], map_idx[1][:]]**2
                                                                 + pix_size**2
                                                                 * colden_err[0].data[map_idx[0][:], map_idx[1][:]]**2)

################################################################################
#
#        pixel_size[0].data[map_idx[0][:], map_idx[1][:]] = dist**2\
#                                                         * (Angle( abs((grid2d_dec[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                                        grid2d_dec[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                                            Angle( abs((grid2d_ra[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                                        grid2d_ra[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                                            np.cos( Angle(grid2d_dec[map_idx[0][:], map_idx[1][:]]*u.deg).rad ))
#
#
#        pixel_size_err[0].data[map_idx[0][:], map_idx[1][:]] = 2.*dist*dist_err\
#                                                             * (Angle( abs((grid2d_dec[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                                            grid2d_dec[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                                                Angle( abs((grid2d_ra[map_idx[0][:]+1, map_idx[1][:]+1]-
#                                                                            grid2d_ra[map_idx[0][:]-1, map_idx[1][:]-1])/2.)*u.deg).rad*
#                                                                np.cos( Angle(grid2d_dec[map_idx[0][:], map_idx[1][:]]*u.deg).rad ))
#
#
#        mass[0].data[map_idx[0][:], map_idx[1][:]] = (pixel_size[0].data[map_idx[0][:], map_idx[1][:]]
#                                                      *colden[0].data[map_idx[0][:], map_idx[1][:]]*hydro_mass)/abundance_ratio
#
#        mass_err[0].data[map_idx[0][:], map_idx[1][:]] = hydro_mass/abundance_ratio\
#                                                       * np.sqrt(  pixel_size_err[0].data[map_idx[0][:], map_idx[1][:]]**2
#                                                                 * colden[0].data[map_idx[0][:], map_idx[1][:]]**2
#                                                                 + pixel_size[0].data[map_idx[0][:], map_idx[1][:]]**2
#                                                                 * colden_err[0].data[map_idx[0][:], map_idx[1][:]]**2)
################################################################################

        mass_map.append(copy.deepcopy(mass))
        mass_map_err.append(copy.deepcopy(mass_err))

    return mass_map, mass_map_err
