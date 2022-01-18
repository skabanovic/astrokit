from astropy.wcs import WCS
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astrokit

def plot_histogram(
    axis,
    counts,
    fontsize=18,
    xmin = None,
    xmax = None,
    ymin = None,
    ymax = None,
    title = 'Distribution',
    xlabel = 'Intensity [K km/s]',
    ylabel = 'Counts [$\#$]',
    save_img= False,
    img_path = './',
    img_name = 'spectrum',
    img_format = 'pdf'
):


    fig = plt.figure(figsize=(15,15))

    width = abs(axis[1]-axis[0])

    plt.bar(axis, counts, width=width)

    if (xmin and xmax) is not None:

        plt.xlim([xmin, xmax])

    if (ymin and ymax) is not None:

        plt.ylim([ymin, ymax])

    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.title(title, size=fontsize)
    plt.xlabel(xlabel, size=fontsize)
    plt.ylabel(ylabel, size=fontsize)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    if save_img:
        matplotlib.pyplot.savefig(img_path + img_name + '.' + img_format,\
                                  transparent = True, bbox_inches = 'tight', pad_inches = 0)

def plot_spectrum(
    vel,
    temp,
    fontsize = 18,
    title = 'Spectrum',
    ylable = 'T$_{\mathrm{mb}}$ [K]',
    xlabel = 'Velocity [km/s]',
    xmin = None,
    xmax = None,
    ymin = None,
    ymax = None,
    do_save= False,
    img_path = './',
    img_name = 'spectrum',
    img_format = 'pdf'
):

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

        plt.ylim([ymin, ymax])

    if do_save:
        matplotlib.pyplot.savefig(img_path + img_name + '.' + img_format,\
                                  transparent = True, bbox_inches = 'tight', pad_inches = 0)

def plot_map(
    sky_map,
    fontsize=18,
    figsize = (18, 15),
    vmin = None,
    vmax = None,
    ra_lable = 'RA (J2000)',
    dec_lable = 'Dec (J2000)',
    cbar_lable = 'Intensity [K$\,$km/s]',
    title = 'Moment 0',
    interpolation = 'none',
    font_style = 'none',
    save_img = False,
    img_path='./',
    img_name = 'image',
    img_format = 'pdf'
):

    wcs = WCS(sky_map[0].header)

    fig = plt.figure(figsize = figsize)
    plt.subplot(projection=wcs)
    if (vmin and vmax) is not None:
        cax=plt.imshow(sky_map[0].data, origin='lower', vmin = vmin, vmax =vmax,
                       cmap='Spectral_r', interpolation = interpolation, aspect='auto')
    else:
        cax=plt.imshow(sky_map[0].data, origin='lower',
                       cmap='Spectral_r', interpolation = interpolation, aspect='auto')
    cbar=plt.colorbar(cax)
    plt.xlabel(ra_lable ,fontsize=fontsize)
    plt.ylabel(dec_lable ,fontsize=fontsize)
    plt.title(title ,fontsize=fontsize)
    cbar.set_label(cbar_lable ,size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)

    if font_style == 'latex':
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    if save_img:

        matplotlib.pyplot.savefig(img_path+img_name+'.'+img_format,\
                                  transparent = False, bbox_inches = 'tight', pad_inches = 0)


# added by Cristian Guevara
def plot_channel_maps(
    channel_maps,
    vel_start,
    vel_end,
    vel_res,
    columns,
    rows,
    legend_location = 'upper left',
    xloc = None,
    yloc = None,
    vmin = None,
    vmax = None,
    ra_lable = 'RA (J2000)',
    dec_lable = 'Dec (J2000)',
    color_bar = 'Intensity [K$\,$km/s]',
    color_map = 'Spectral_r',
    interpolation = 'none',
    do_contour = False,
    contour_level = None,
    fontsize = 18,
    font_scale = 6,
    axis_scale = 6,
    save_img = False,
    img_path='./',
    img_name = 'image',
    img_format = 'pdf'
):

    if (xloc or yloc) is None:

        if legend_location == 'upper left':

            yloc = 0.9
            xloc = 0.1

        elif legend_location == 'lower left':

            yloc = 0.1
            xloc = 0.1



    w = WCS(channel_maps[0][0].header)

    #aspect ratio withn respecto to vertical direction
    yratio = channel_maps[0][0].header["NAXIS2"]/channel_maps[0][0].header["NAXIS1"]

    maxv=[]
    minv=[]

    channel_number = columns*rows

    for channel in range(channel_number):

        channel_maps[channel][0].data = np.nan_to_num(channel_maps[channel][0].data)
        maxv.append(np.nanmax(channel_maps[channel][0].data))
        minv.append(np.nanmin(channel_maps[channel][0].data))

    if vmin is None:

        vmin = min(maxv)

    if vmax is None:

        vmax= max(maxv)

    if do_contour :

        step = (vmax-vmin)/contour_level #step of the contours
        levels = np.arange(v_min+step,v_max+step,step) # levels array

    if columns==1 and rows==1:

        fig = plt.figure(figsize=(25, 15))
        plt.subplot(projection=w)
        cax=plt.imshow(channel_maps[0][0].data, origin='lower',cmap=color_map, vmax=vmax, vmin=vmin, interpolation = interpolation)
        cbar=plt.colorbar(cax)
        if do_contour:
            plt.contour(channel_maps[0][0].data, contour_level)
        plt.text(channel_maps[0][0].header["NAXIS1"]*xloc, channel_maps[0][0].header["NAXIS2"]*yloc, \
                 str(vel_start)+" to "+str(vel_start+(1)*vel_res)+" [km/s]", \
                 bbox={'facecolor': 'lightgrey', 'pad': 10}, fontsize = fontsize)
        plt.xlabel(ra_lable, fontsize = fontsize)
        plt.ylabel(dec_lable, fontsize = fontsize)
        plt.title('Channel Map '+' ['+str(vel_start)+'_'+str(vel_start+vel_res)+']$\,$km/s' , fontsize = fontsize)
        cbar.set_label(color_bar, size = fontsize)
        cbar.ax.tick_params(labelsize = fontsize)
        cbar.set_label('Intensity [K km/s]', size=font_scale*max(columns,rows))
        plt.rc('xtick', labelsize = fontsize)
        plt.rc('ytick', labelsize = fontsize)
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')

    elif rows==1 :

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows, columns, hspace=0, wspace=-0.01)
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        for column in range(columns):
                a =  fig.add_subplot(gs[0, column], projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*xloc,\
                       channel_maps[0][0].header["NAXIS2"]*yloc,\
                       str(vel_start+column*vel_res)+" to "+str(vel_start+(column+1)*vel_res)+"$\,$[km/s]",\
                       bbox={'facecolor': 'lightgrey', 'pad': 10},fontsize=18)
                lon = a.coords[0]
                lon.set_ticklabel(size=axis_scale*max(columns,rows))
                lat = a.coords[1]
                lat.set_ticklabel_visible(False)
                if column == 0:
                    lat.set_ticklabel_visible(True)
                    lat.set_ticklabel(size=axis_scale*max(columns,rows))
                im = a.imshow(channel_maps[column][0].data, cmap=color_map,\
                              vmax=vmax, vmin=vmin, origin='lower',
                              interpolation = interpolation)
                if do_contour:
                    a.contour(channel_maps[column][0].data, levels )
                if column == 0:
                    lon.set_axislabel(ra_lable, size=font_scale*max(columns,rows))
                    lat.set_axislabel(dec_lable, size=font_scale*max(columns,rows))
                else:
                    lon.set_axislabel(' ')
                    lat.set_axislabel(' ')



        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05/columns, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label(color_bar,size=font_scale*max(columns, rows))


    elif columns==1:

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows, columns, hspace=0, wspace=-0.01)
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        for row in range(0,rows):
                a =  fig.add_subplot(gs[row, 0], projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*xloc, channel_maps[0][0].header["NAXIS2"]*yloc,\
                       str(vel_start+row*vel_res)+" to "+str(vel_start+(row+1)*vel_res)+"$\,$[km/s]",\
                       bbox={'facecolor': 'lightgrey', 'pad': 10}, fontsize=font_scale*max(columns,rows))
                lon = a.coords[0]
                lat = a.coords[1]
                lat.set_ticklabel(size=axis_scale*max(columns,rows))
                lon.set_ticklabel_visible(False)
                if row == rows - 1:
                    lon.set_ticklabel_visible(True)
                    lon.set_ticklabel(size=axis_scale*max(columns,rows))
                im = a.imshow(channel_maps[row][0].data, cmap=color_map,\
                              vmax=vmax, vmin=vmin, origin='lower',
                              interpolation = interpolation)
                if do_contour:
                    a.contour(channel_maps[row][0].data, levels)
                if row == rows-1:
                    lon.set_axislabel(ra_lable, size=font_scale*max(columns,rows))
                    lat.set_axislabel(dec_lable, size=font_scale*max(columns,rows))
                else:
                    lon.set_axislabel(' ')
                    lat.set_axislabel(' ')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label(color_bar, size=font_scale*max(columns,rows))

    else:

        fig = plt.figure(dpi=70,figsize=(10*columns, 9*rows*yratio))
        gs = gridspec.GridSpec(rows,columns, hspace=0,wspace=-0.01)
        la=0
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        for row in range(0,rows):
            for column in range(0,columns):
                a =  fig.add_subplot(gs[row, column],projection=w)
                a.text(channel_maps[0][0].header["NAXIS1"]*xloc,
                       channel_maps[0][0].header["NAXIS2"]*yloc,\
                       str(vel_start+la*vel_res)+" to "+str(vel_start+(la+1)*vel_res)+"$\,$[km/s]",\
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
                    lon.set_axislabel(ra_lable, size=font_scale*max(columns,rows))
                    lat.set_axislabel(dec_lable, size=font_scale*max(columns,rows))
                else:
                    lon.set_axislabel(' ')
                    lat.set_axislabel(' ')

                im = a.imshow(channel_maps[la][0].data, cmap=color_map,\
                              vmax=vmax, vmin=vmin, origin='lower', interpolation = interpolation)



                if do_contour:
                    a.contour(channel_maps[la][0].data,levels )
                la = la+1


        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.80, 0.125, 0.05/rows, 0.76/rows])
        cbar=fig.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8*max(columns, rows))
        cbar.set_label(color_bar, size=font_scale*max(columns,rows))

        if save_img:

            matplotlib.pyplot.savefig(img_path+img_name+'.'+img_format,\
                                      transparent = True, bbox_inches = 'tight', pad_inches = 0)


def plot_cluster_spectra(cube,
                         mode_signal_mean,
                         mode_signal_std,
                         save_image = False,
                         image_path = None,
                         image_name = None):

    axis = 3

    vel = astrokit.get_axis(axis, cube)/1e3

    for i in range(len(mode_signal_mean)):

        fig= plt.figure(figsize=(14,7))

        fontsize = 18

        plt.plot(vel,
                 mode_signal_mean[i],
                 linewidth = 5,
                 label = 'Averaged Spectrum')

        plt.plot(vel,
                 mode_signal_mean[i]+mode_signal_std[i],
                 color = 'C1',
                 linestyle='dashed',
                 linewidth = 3,
                 label = 'Standard Deviation')

        plt.plot(vel,
                 mode_signal_mean[i]-mode_signal_std[i],
                 color = 'C1',
                 linestyle='dashed',
                 linewidth = 3)

        plt.legend(prop={'size': fontsize}, frameon=False, loc='upper left')
        plt.title('Cluster - ' + str(i), size=fontsize)
        plt.xlabel('Velocity [km/s]', size=fontsize)
        plt.ylabel('Temperature [K]', size=fontsize)

        if save_image:

            matplotlib.pyplot.savefig(image_path + image_name,
                                      transparent = False,
                                      bbox_inches = 'tight',
                                      pad_inches = 0)
