#############################################################
#
#  mask.py includes function to equalize spectral cubes
#
#  CHAOS (Cologne Hacked Astrophysical Software)
#
#  Created by Slawa Kabanovic
#
#############################################################

import time
import numpy as np

#############################################################
#
# loop_time determies the remining time of the loop
# to finish
#
#############################################################

def loop_time(idx, loop_len, time_start, time_end):

    progress=round(((idx+1)/float(loop_len))*1e4)/1e2

    time_remain=(time_end - time_start)*(float(loop_len)-(idx+1))


    if time_remain < 60.:

        print('progress: ' + str(progress) +\
              ' %, remaining time: ' + str(round(time_remain)) + ' sec')

    elif time_remain < 3600:

        time_remain=time_remain/60.

        print('progress: ' + str(progress) +\
              ' %, remaining time: ' + str(round(time_remain)) + ' min')

    else:

        time_remain=time_remain/3600.

        print('progress: ' + str(progress) +\
              ' %, remaining time: ' + str(round(time_remain*10.)/10.) + ' h')
