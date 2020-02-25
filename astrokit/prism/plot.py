from astropy.wcs import WCS
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_histogram(axis, counts,
                   fontsize=18,
                   title = 'Distribution',
                   xlabel = 'Intensity [K km/s]',
                   ylabel = 'Counts [#]'):


    fig = plt.figure(figsize=(15,15))

    width = abs(axis[1]-axis[0])

    plt.bar(axis, counts, width=width)

    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.title(title, size=fontsize)
    plt.xlabel(xlabel, size=fontsize)
    plt.ylabel(ylabel, size=fontsize)

def plot_spectrum(vel, temp ,
                  fontsize = 18,
                  title = 'Spectrum',
                  ylable = 'T$_{\mathrm{mb}}$ [K]',
                  xlabel = 'Velocity [km/s]',
                  xmin = None, xmax = None,
                  ymin = None, ymax = None,
                  do_save= False, img_path = './', img_name = 'spectrum', img_format = 'pdf'):

    fig = plt.figure(figsize=(15,15))

    plt.step(vel, temp, linewidth=3)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.title(title, size=fontsize)
    plt.ylabel(ylable, size=fontsize)
    plt.xlabel(xlabel, size=fontsize)

    if (xmin and xmax) is not None:

        plt.xlim([xmin, xmax])

    if (ymin and ymax) is not None:

        plt.ymin([ymin, ymax])

    if do_save:
        matplotlib.pyplot.savefig(img_path + img_name + '.' + img_format,\
                                  transparent = True, bbox_inches = 'tight', pad_inches = 0)

def plot_map(sky_map, fontsize=18,
             vmin = None, vmax = None,
             ra_lable = 'RA (J2000)',
             dec_lable = 'Dec (J2000)',
             cbar_lable = 'Intensity [K$\,$km/s]',
             title = 'Moment 0',
             save_img = False, img_path='./', img_name = 'image', img_format = 'pdf'):

    wcs = WCS(sky_map[0].header)

    fig = plt.figure(figsize=(25, 15))
    plt.subplot(projection=wcs)
    if (vmin and vmax) is not None:
        cax=plt.imshow(sky_map[0].data, origin='lower', vmin = vmin, vmax =vmax,
                       cmap='Spectral_r', interpolation='gaussian')
    else:
        cax=plt.imshow(sky_map[0].data, origin='lower',
                       cmap='Spectral_r', interpolation='gaussian')
    cbar=plt.colorbar(cax)
    plt.xlabel(ra_lable ,fontsize=fontsize)
    plt.ylabel(dec_lable ,fontsize=fontsize)
    plt.title(title ,fontsize=fontsize)
    cbar.set_label(cbar_lable ,size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)

    if save_img:

        matplotlib.pyplot.savefig(img_path+img_name+'.'+img_format,\
                                  transparent = True, bbox_inches = 'tight', pad_inches = 0)


# added by Cristian Guevara
def plot_channel_maps(channel_maps, vel_start, vel_end, vel_res,
                      columns, rows,
                      set_limit = True, vmin = None, vmax = None,
                      do_contour = False, contour_level = None,
                      fontsize = 18, font_scale = 6, axis_scale = 6,
                      save_img = False, img_path='./', img_name = 'image', img_format = 'pdf'):


    w = WCS(channel_maps[0][0].header)

    #aspect ratio withn respecto to vertical direction
    yratio = channel_maps[0][0].header["NAXIS2"]/channel_maps[0][0].header["NAXIS1"]

    maxv=[]

    channel_number = columns*rows

    for channel in range(channel_number):

        channel_maps[channel][0].data = np.nan_to_num(channel_maps[channel][0].data)
        maxv.append(np.nanmax(channel_maps[channel][0].data))

    if set_limit:

        vmax= max(maxv)
        vmin = 0

    if do_contour :

        step = (vmax-vmin)/contour_level #step of the contours
        levels = np.arange(v_min+step,v_max+step,step) # levels array

    if columns==1 and rows==1:

        fig = plt.figure(figsize=(25, 15))
        plt.subplot(projection=w)
        cax=plt.imshow(channel_maps[0][0].data, origin='lower',cmap='Spectral_r', vmax=vmax, vmin=vmin)
        cbar=plt.colorbar(cax)
        if do_contour:
            plt.contour(channel_maps[0][0].data, contour_level)
        plt.text(channel_maps[0][0].header["NAXIS1"]*(0.1), channel_maps[0][0].header["NAXIS2"]*0.85, \
                 str(vel_start)+" to "+str(vel_start+(1)*vel_res)+" [km/s]", \
                 bbox={'facecolor': 'lightgrey', 'pad': 10}, fontsize = fontsize)
        plt.xlabel('RA (J2000)', fontsize = fontsize)
        plt.ylabel('Dec (J2000)', fontsize = fontsize)
        plt.title('Channel Map '+' ['+str(vel_start)+'_'+str(vel_start+vel_res)+'] km/s' , fontsize = fontsize)
        cbar.set_label('Intensity [K$\,$km/s]', size = fontsize)
        cbar.ax.tick_params(labelsize = fontsize)
        cbar.set_label('[K km/s]', size=8*max(columns,rows))
        plt.rc('xtick', labelsize = fontsize)
        plt.rc('ytick', labelsize = fontsize)

    elif rows==1 :

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows, columns, hspace=0, wspace=0)
        for column in range(columns):
                a =  fig.add_subplot(gs[0, column], projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*(0.1),\
                       channel_maps[0][0].header["NAXIS2"]*0.85,\
                       str(vel_start+column*vel_res)+" to "+str(vel_start+(column+1)*vel_res)+" [km/s]",\
                       bbox={'facecolor': 'lightgrey', 'pad': 10},fontsize=18)
                lon = a.coords[0]
                lon.set_ticklabel(size=axis_scale*max(columns,rows))
                lat = a.coords[1]
                lat.set_ticklabel_visible(False)
                if column == 0:
                    lat.set_ticklabel_visible(True)
                    lat.set_ticklabel(size=axis_scale*max(columns,rows))
                im = a.imshow(channel_maps[column][0].data, cmap='Spectral_r',\
                              vmax=vmax, vmin=vmin, origin='lower')
                if do_contour:
                    a.contour(channel_maps[column][0].data, levels )
                if column == 0:
                    lon.set_axislabel('RA (J2000)', size=8*max(columns,rows))
                    lat.set_axislabel('Dec (J2000)', size=8*max(columns,rows))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05/columns, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label('[K km/s]',size=8*max(columns, rows))


    elif columns==1:

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows, columns, hspace=0, wspace=0)
        for row in range(0,rows):
                a =  fig.add_subplot(gs[row, 0], projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*(0.1), channel_maps[0][0].header["NAXIS2"]*0.85,\
                       str(vel_start+row*vel_res)+" to "+str(vel_start+(row+1)*vel_res)+" [km/s]",\
                       bbox={'facecolor': 'lightgrey', 'pad': 10}, fontsize=font_scale*max(columns,rows))
                lon = a.coords[0]
                lat = a.coords[1]
                lat.set_ticklabel(size=axis_scale*max(columns,rows))
                lon.set_ticklabel_visible(False)
                if row == rows - 1:
                    lon.set_ticklabel_visible(True)
                    lon.set_ticklabel(size=axis_scale*max(columns,rows))
                im = a.imshow(channel_maps[row][0].data, cmap='Spectral_r',\
                              vmax=vmax, vmin=vmin, origin='lower')
                if do_contour:
                    a.contour(channel_maps[row][0].data, levels)
                if row == rows-1:
                    lon.set_axislabel('RA (J2000)',size=8*max(columns,rows))
                    lat.set_axislabel('Dec (J2000)',size=8*max(columns,rows))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label('[K km/s]',size=8*max(columns,rows))

    else:

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows,columns, hspace=0,wspace=0)
        la=0
        for row in range(0,rows):
            for column in range(0,columns):
                a =  fig.add_subplot(gs[row, column],projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*(0.1),
                       channel_maps[0][0].header["NAXIS2"]*0.85,\
                       str(vel_start+la*vel_res)+" to "+str(vel_start+(la+1)*vel_res)+" [km/s]",\
                       bbox={'facecolor': 'lightgrey', 'pad': 10},fontsize=font_scale*max(columns, rows))
                lon = a.coords[0]
                lat = a.coords[1]
                lon.set_ticklabel_visible(False)
                lat.set_ticklabel_visible(False)
                if row == rows - 1:
                    lon.set_ticklabel_visible(True)
                    lon.set_ticklabel(size=axis_scale*max(columns,rows))
                if column == 0:
                    lat.set_ticklabel_visible(True)
                    lat.set_ticklabel(size=axis_scale*max(columns,rows))
                if row == rows-1 and column == 0:
                    lon.set_axislabel('RA (J2000)',size=8*max(columns,rows))
                    lat.set_axislabel('DEC (J2000)',size=8*max(columns,rows))
                im = a.imshow(channel_maps[la][0].data, cmap='Spectral_r',\
                              vmax=vmax, vmin=vmin, origin='lower')
                if do_contour:
                    a.contour(channel_maps[la][0].data,levels )
                la = la+1


        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05/rows, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label('[K km/s]',size=8*max(columns,rows))

        if save_img:

            matplotlib.pyplot.savefig(img_path+img_name+'.'+img_format,\
                                      transparent = True, bbox_inches = 'tight', pad_inches = 0)