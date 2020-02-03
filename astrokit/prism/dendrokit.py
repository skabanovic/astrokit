#############################################################
#
#  dendro_tools.py
#
#  Created by Slawa Kabanovic
#
#############################################################

import numpy as np

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



def struct_average(hdul, dendro, weight_map = 'nan',  struct_type="leaf", weight = False):

    struct_idx = find_struct(dendro, struct_type)

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
