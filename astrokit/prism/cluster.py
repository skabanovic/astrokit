from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import numpy as np
import astrokit
import copy
#import pomegranate
from scipy.optimize import curve_fit

from astropy.coordinates import Angle
import astropy.units as u

def cluster_information(cube,
                        cluster_range,
                        #weight=None,
                        methode = 'BIC',
                        reduce_dim = 'pca',
                        norm = 'mean',
                        threshold = 1e-3,
                        gmm_iter = 1000,
                        sample = None,
                        pca_threshold = 0.99,
                        rms_threshold = 0):

    # Dimension of the input cube
    dim_ax1 = cube[0].header['NAXIS1'] # Points in X/RA axis
    dim_ax2 = cube[0].header['NAXIS2'] # Points in Y/Dec axis
    dim_ax3 = cube[0].header['NAXIS3'] # Points in velocity

    # Check: if a grid point is masked with nan at velocity v it should be masked at all v
    for idx_ax1 in range(dim_ax1):
        for idx_ax2 in range(dim_ax2):
            if np.isnan(np.sum(cube[0].data[:,idx_ax2, idx_ax1])):
                cube[0].data[:,idx_ax2,idx_ax1] = np.nan

    # From a [dimV,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_data_masked = cube[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()

    #if weight:
    #    flat_weight_masked = weight[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()
    #    flat_weight = flat_weight_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    # Consider only the points that are not masked
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked,axis=1))]

    if rms_threshold>0:

        rms_of_dim = np.sqrt(np.mean(flat_data**2, axis=0))

        #noise_threshold = rms_threshold*np.sqrt(len(flat_data))
        #noise_of_dim = np.sqrt(np.mean(flat_data**2, axis=0)*len(flat_data))
        #itensity_of_dim = np.trapz(flat_data, axis = 0)

        flat_data = flat_data[:, rms_of_dim>rms_threshold]

    #    if weight:
    #        flat_weight = flat_weight[:, rms_of_dim>rms_threshold]

        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(flat_data)[1])+' dimensions')


    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        normed_flat_data = np.zeros_like(flat_data)

        if norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'z-score':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        elif norm == 'max_scale':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.max(flat_data[spect, :])

        elif norm == 'min-max':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] =\
                    (flat_data[spect, :]-np.min(flat_data[spect, :]))/(np.max(flat_data[spect, :])-np.min(flat_data[spect, :]))


        else:
            normed_flat_data = flat_data

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(pca_threshold, whiten=True)

        reduced_data = pca.fit_transform(normed_flat_data)

        print('PCA done')
        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(reduced_data)[1])+' dimensions')

    elif reduce_dim == 'original':

        normed_flat_data = np.zeros_like(flat_data)

        if norm == 'mean':

            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'max_scale':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.max(flat_data[spect, :])

        elif norm == 'min-max':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] =\
                    (flat_data[spect, :]-np.min(flat_data[spect, :]))/(np.max(flat_data[spect, :])-np.min(flat_data[spect, :]))

        elif norm == 'length':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.sqrt(np.sum(flat_data[spect, :]**2))

        elif norm == 'intensity':

            axis = 3
            vel = astrokit.get_axis(axis, cube)

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.trapz(flat_data[spect, :], vel)


        elif norm == 'std':

            for i in range(len(normed_flat_data)):

                normed_flat_data[i,:]= flat_data[i,:]/np.std(flat_data[i,:])

        elif norm == 'z-score':

            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        elif norm == 'max_scale':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.max(flat_data[spect, :])

            #for i in range(len(normed_flat_data)):
            #    normed_flat_data[i,:]= (flat_data[i,:] - np.mean(flat_data[i,:]))/np.std(flat_data[i,:])

        else:

            print('warning: continue without normalization of data')

            normed_flat_data = flat_data

        reduced_data = normed_flat_data

    # REMEMBER: things will slightly change if you ask for, say 10000 spectra.

    print('Starting the computation of Bayesian Information Criterion')

    # We compute the BIC on a random sample of the data
    # Consider a random sample

    if sample:

        indices = np.random.choice(reduced_data.shape[0], sample, replace=False)

        sample_data = reduced_data[indices]

    #    if weight:

    #        sample_weight = flat_weight[indices]

    else:

        sample_data = reduced_data

    #    if weight:

    #        sample_weight = flat_weight



    info_criterion = []

    cluster_list = np.arange(cluster_range[0], cluster_range[1], 1)

    for cluster in cluster_list:

    #    if weight:
    #
    #        #Make the GMM model using pomegranate
    #        model = pomegranate.gmm.GeneralMixtureModel.from_samples(
    #            pomegranate.MultivariateGaussianDistribution,   #Either single function, or list of functions
    #            n_components = cluster,     #Required if single function passed as first arg
    #            X = sample_data,     #data format: each row is a point-coordinate, each column is a dimension
    #            stop_threshold = 0.01,  #Lower this value to get better fit but take longer.
    #            max_iterations = 100
    #            )
    #
    #        #Force the model to train again, using additional fitting parameters
    #        model.fit(
    #            X = sample_data,         #data format: each row is a coordinate, each column is a dimension
    #            weights = sample_weight,  #List of weights. One for each point-coordinate
    #            stop_threshold = 0.01,  #Lower this value to get better fit but take longer.
    #            max_iterations = 100
    #                        #   (sklearn likes better/slower fits than pomegrante by default)
    #            )

    #    else:

        model = GaussianMixture(

            n_components = cluster,
            covariance_type ='full',
            tol = threshold,
            max_iter = gmm_iter

        ).fit(sample_data)


        if methode == 'BIC':

        #    if weight:

        #        N_k = cluster-1.+cluster*dim_ax3+(cluster*dim_ax3*(dim_ax3-1.))/2.
        #        cluster_info_criterion = N_k*np.log(len(sample_data))-2.*np.sum(model.log_probability(sample_data))
        #        print('cluster = '+str(cluster)+' ---> BIC = '+str(cluster_info_criterion))

        #    else:

            cluster_info_criterion = model.bic(sample_data)
            print('cluster = '+str(cluster)+' ---> BIC = '+str(cluster_info_criterion))

        elif methode == 'AIC':

            cluster_info_criterion = model.aic(sample_data)
            print('cluster = '+str(cluster)+' ---> AIC = '+str(cluster_info_criterion))


        info_criterion.append(cluster_info_criterion)

    return info_criterion

# Length of each time series is simply dimV

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def spectra_clustering(cube,
                       n_com,
                     #  weight = None,
                       gmm_iter = 1000,
                       threshold = 1e-3,
                       reduce_dim = 'original',
                       norm = 'mean',
                       pca_threshold = 0.99,
                       rms_threshold = 0):

    # Dimension of the input cube
    dim_ax1 = cube[0].header['NAXIS1'] # Points in X/RA axis
    dim_ax2 = cube[0].header['NAXIS2'] # Points in Y/Dec axis
    dim_ax3 = cube[0].header['NAXIS3'] # Points in velocity

    # Check: if a grid point is masked with nan at velocity v it should be masked at all v
    index_map = np.zeros([2, dim_ax2, dim_ax1])

    for idx_ax1 in range(dim_ax1):
        for idx_ax2 in range(dim_ax2):

            index_map[0, idx_ax2, idx_ax1] = idx_ax1
            index_map[1, idx_ax2, idx_ax1] = idx_ax2

            if np.isnan(np.sum(cube[0].data[:,idx_ax2, idx_ax1])):

                cube[0].data[:, idx_ax2, idx_ax1] = np.nan

    # From a [dimV,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_index_masked = index_map.reshape(2, dim_ax2*dim_ax1).transpose()
    flat_data_masked = cube[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()

#    if weight:
#        flat_weight_masked = weight[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()
#        flat_weight = flat_weight_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    # Consider only the points that are not masked
    flat_index = flat_index_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    if rms_threshold>0:

        rms_of_dim = np.sqrt(np.mean(flat_data**2, axis=0))

        #noise_threshold = rms_threshold*np.sqrt(len(flat_data))
        #noise_of_dim = np.sqrt(np.mean(flat_data**2, axis=0)*len(flat_data))
        #itensity_of_dim = np.trapz(flat_data, axis = 0)

        flat_data = flat_data[:, rms_of_dim>rms_threshold]

    #    if weight:
    #        flat_weight = flat_weight[:, rms_of_dim>rms_threshold]

        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(flat_data)[1])+' dimensions')

    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        normed_flat_data = np.zeros_like(flat_data)

        if norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'z-score':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        elif norm == 'max_scale':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.max(flat_data[spect, :])

        elif norm == 'min-max':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] =\
                    (flat_data[spect, :]-np.min(flat_data[spect, :]))/(np.max(flat_data[spect, :])-np.min(flat_data[spect, :]))

        else:
            normed_flat_data = flat_data

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(pca_threshold, whiten=True)

        reduced_data = pca.fit_transform(normed_flat_data)

        print('PCA done')
        #print(normed_flat_data.shape)
        #print(reduced_data.shape)
        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(reduced_data)[1])+' dimensions')

    elif reduce_dim == 'original':

        normed_flat_data = np.zeros_like(flat_data)

        if norm == 'mean':

            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'max_scale':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.max(flat_data[spect, :])

        elif norm == 'min-max':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] =\
                    (flat_data[spect, :]-np.min(flat_data[spect, :]))/(np.max(flat_data[spect, :])-np.min(flat_data[spect, :]))

        elif norm == 'length':

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.sqrt(np.sum(flat_data[spect, :]**2))

        elif norm == 'intensity':

            axis = 3
            vel = astrokit.get_axis(axis, cube)

            for spect in range(len(flat_data)):

                normed_flat_data[spect, :] = flat_data[spect, :]/np.trapz(flat_data[spect, :], vel)


        elif norm == 'std':

            for i in range(len(normed_flat_data)):

                normed_flat_data[i,:]= flat_data[i,:]/np.std(flat_data[i,:])

        elif norm == 'z-score':

            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

            #for i in range(len(normed_flat_data)):
            #    normed_flat_data[i,:]= (flat_data[i,:] - np.mean(flat_data[i,:]))/np.std(flat_data[i,:])

        else:

            print('warning: continue without normalization of data')

            normed_flat_data = flat_data

        reduced_data = normed_flat_data

#    if weight:
#
#        #Make the GMM model using pomegranate
#        model = pomegranate.gmm.GeneralMixtureModel.from_samples(
#            pomegranate.MultivariateGaussianDistribution,   #Either single function, or list of functions
#            n_components=n_com,     #Required if single function passed as first arg
#            X = reduced_data,     #data format: each row is a point-coordinate, each column is a dimension
#            init = 'first-k',
#            stop_threshold = 0.001,  #Lower this value to get better fit but take longer.
#            max_iterations = gmm_iter
#            )
#
#        #Force the model to train again, using additional fitting parameters
#        model.fit(
#            X=reduced_data,         #data format: each row is a coordinate, each column is a dimension
#            weights = flat_weight,  #List of weights. One for each point-coordinate
#            stop_threshold = 0.001,  #Lower this value to get better fit but take longer.
#            max_iterations = gmm_iter
#                            #   (sklearn likes better/slower fits than pomegrante by default)
#            )

#    else:

    model = GaussianMixture(

        n_components = n_com,
        covariance_type = 'full',
        random_state = 42,
        tol = threshold,
        max_iter = gmm_iter

    ).fit(reduced_data)


    labels = model.predict(reduced_data)
    probs = model.predict_proba(reduced_data)


    probs_cube = np.zeros([n_com, dim_ax2, dim_ax1])
    domain_map = astrokit.zeros_map(cube)

    domain_map[0].data[:,:] = np.nan

    for label in range(len(labels)):

        domain_map[0].data[int(flat_index[label][1]), int(flat_index[label][0])] = labels[label]
        probs_cube[:, int(flat_index[label][1]), int(flat_index[label][0])] = probs[label, :]

    #mode_signal_mean = []
    #mode_signal_std = []

    #labels_map = labels.reshape(dim_ax2, dim_ax1).transpose()

    #for i in range(n_com):

    #    signal = np.mean(cube[0].data[:,domain_map[0].data==i], axis=1)
#        std = np.std(cube[0].data[:, domain_map[0].data==i], axis=1)

    #    mode_signal_mean.append(signal)
    #    mode_signal_std.append(std)

    #return mode_signal_mean, mode_signal_std, domain_map, probs_cube
    return domain_map, probs_cube


def cluster_average_spectra(
    cube,
    cluster_map,
    weight_map = None,
    weight_cube = None
    ):

    cluster_num = int(np.nanmax(cluster_map[0].data)) +1
    len_ax3 = cube[0].header['NAXIS3']
    cluster_spect = np.zeros([cluster_num, len_ax3])
    cluster_std = np.zeros([cluster_num, len_ax3])


    for cluster in range(cluster_num):

        if weight_map:

            cluster_spect[cluster, :] = np.average(cube[0].data[:, cluster_map[0].data == cluster],
                                                   weights = weight_map[0].data[cluster_map[0].data == cluster], axis = 1)

        elif weight_cube:

            cluster_spect[cluster, :] = np.average(cube[0].data[:, cluster_map[0].data == cluster],
                                                   weights = weight_cube[0].data[:, cluster_map[0].data == cluster], axis = 1)
        else:

            cluster_spect[cluster, :] = np.average(cube[0].data[:, cluster_map[0].data == cluster], axis = 1)

        cluster_std[cluster, :] = np.std(cube[0].data[:, cluster_map[0].data == cluster], axis = 1)


    return cluster_spect, cluster_std

def cluster_average_value(
    input_map,
    cluster_map
    ):

    cluster_num = int(np.nanmax(cluster_map[0].data)) +1

    cluster_value = np.zeros(cluster_num)

    for cluster in range(cluster_num):

        cluster_value[cluster] = np.average(input_map[0].data[cluster_map[0].data == cluster])

    return cluster_value

def multi_gauss_fit(cube,
                    rms_map,
                    gauss_num = 2,
                    sort = 'velocity',
                    guess = None,
                    bound_min = None,
                    bound_max = None,
                    sigma = 3):

    axis = 1
    grid_ax1 = astrokit.get_axis(axis, cube)

    axis = 2
    grid_ax2 = astrokit.get_axis(axis, cube)

    grid_ax3 = np.arange(1,3*gauss_num+1, 1)

    ref_value_ax1 = cube[0].header['CRVAL1']

    ref_value_ax2 = cube[0].header['CRVAL2']

    ref_value_ax3 = 0

    fit_cube =  astrokit.empty_grid(grid_ax1 = grid_ax1,
                                    grid_ax2 = grid_ax2,
                                    grid_ax3 = grid_ax3,
                                    ref_value_ax1 = ref_value_ax1,
                                    ref_value_ax2 = ref_value_ax2,
                                    ref_value_ax3 = ref_value_ax3,
                                    beam_maj = None,
                                    beam_min = None)

    fit_cube_err = astrokit.zeros_cube(fit_cube)

    axis = 3
    vel =  astrokit.get_axis(axis, cube)/1e3

    len_ax1 = cube[0].header['NAXIS1']
    len_ax2 = cube[0].header['NAXIS2']

    for idx_ax1 in range(len_ax1):
        for idx_ax2 in range(len_ax2):

            if (np.max(cube[0].data[:, idx_ax2, idx_ax1]) > (sigma * rms_map[0].data[idx_ax2, idx_ax1])):

                fill_param = True

                try:

                    rms_spect = np.zeros_like(vel)
                    rms_spect[:] = rms_map[0].data[idx_ax2, idx_ax1]

                    popt, pcov=curve_fit(astrokit.gauss_multi,
                                         vel,
                                         cube[0].data[:, idx_ax2, idx_ax1],
                                         sigma = rms_spect,
                                         p0 = guess,
                                         bounds = (bound_min, bound_max),
                                         maxfev = 1000)

                except:

                    fill_param = False
                    fit_cube[0].data[:, idx_ax2, idx_ax1] = np.nan

            else:

                fill_param = False

                fit_cube[0].data[:, idx_ax2, idx_ax1] = np.nan


            if fill_param:

                perr = np.sqrt(np.diag(pcov))

                sort_values = np.zeros(gauss_num)
                sort_popt = np.zeros(gauss_num*3)
                sort_perr = np.zeros(gauss_num*3)

                if sort == 'velocity':

                    idx_param = 1

                elif sort == 'intensity':

                    idx_param = 0

                elif sort == 'width':

                    idx_param = 2

                for gauss in range(gauss_num):

                    sort_values[gauss] = popt[idx_param+gauss*3]

                sort_idx = np.argsort(sort_values)

                for gauss in range(gauss_num):

                    sort_popt[gauss*3] = popt[sort_idx[gauss]*3]
                    sort_popt[gauss*3+1] = popt[sort_idx[gauss]*3+1]
                    sort_popt[gauss*3+2] = popt[sort_idx[gauss]*3+2]

                    sort_perr[gauss*3] = perr[sort_idx[gauss]*3]
                    sort_perr[gauss*3+1] = perr[sort_idx[gauss]*3+1]
                    sort_perr[gauss*3+2] = perr[sort_idx[gauss]*3+2]

                fit_cube[0].data[:, idx_ax2, idx_ax1] = sort_popt[:]

                fit_cube_err[0].data[:, idx_ax2, idx_ax1] = sort_perr[:]

    return fit_cube, fit_cube_err

def sequential_gauss_fit(cube,
                         rms_map,
                         gauss_num,
                         first_guess = None,
                         first_bound_min = None,
                         first_bound_max = None,
                         sigma = 3,
                         fit_const = 0.1):

    axis = 1
    grid_ax1 = astrokit.get_axis(axis, cube)

    axis = 2
    grid_ax2 = astrokit.get_axis(axis, cube)

    grid_ax3 = np.arange(1,3*gauss_num+1, 1)

    ref_value_ax1 = cube[0].header['CRVAL1']

    ref_value_ax2 = cube[0].header['CRVAL2']

    ref_value_ax3 = 0

    fit_cube =  astrokit.empty_grid(grid_ax1 = grid_ax1,
                           grid_ax2 = grid_ax2,
                           grid_ax3 = grid_ax3,
                           ref_value_ax1 = ref_value_ax1,
                           ref_value_ax2 = ref_value_ax2,
                           ref_value_ax3 = ref_value_ax3,
                           beam_maj = None,
                           beam_min = None)

    axis = 3
    vel =  astrokit.get_axis(axis, cube)/1e3

    len_ax1 = cube[0].header['NAXIS1']
    len_ax2 = cube[0].header['NAXIS2']

    for idx_ax1 in range(len_ax1):
        for idx_ax2 in range(len_ax2):

            if (np.max(cube[0].data[:, idx_ax2, idx_ax1]) > (sigma * rms_map[0].data[idx_ax2, idx_ax1])):

                fill_param = True

                try:

                    rms_spect = np.zeros_like(vel)
                    rms_spect[:] = rms_map[0].data[idx_ax2, idx_ax1]

                    guess = np.zeros(3)
                    bound_min = np.zeros(3)
                    bound_max = np.zeros(3)

                    for idx in range(3):
                        guess[idx] = first_guess[idx]
                        bound_min[idx] = first_bound_min[idx]
                        bound_max[idx] = first_bound_max[idx]


                    popt, pcov=curve_fit(astrokit.gauss_multi,
                                         vel,
                                         cube[0].data[:, idx_ax2, idx_ax1],
                                         sigma = rms_spect,
                                         p0 = guess,
                                         bounds = (bound_min, bound_max),
                                         maxfev = 1000)

                except:

                    fill_param = False
                    fit_cube[0].data[:, idx_ax2, idx_ax1] = np.nan

                if ((gauss_num > 1) and fill_param):

                    gauss = 1

                    while ( (gauss < gauss_num)
                    and fill_param ):

                        guess = np.zeros(3*(gauss+1) )
                        bound_min = np.zeros(3*(gauss+1) )
                        bound_max = np.zeros(3*(gauss+1) )

                        for idx in range(3*gauss):

                            guess[idx] = popt[idx]
                            bound_min[idx] = popt[idx] - abs(popt[idx]*fit_const)
                            bound_max[idx] = popt[idx] + abs(popt[idx]*fit_const)

                        first_idx = 0

                        for idx in range(3*gauss, 3*gauss +3):

                            guess[idx] = first_guess[first_idx]
                            bound_min[idx] = first_bound_min[first_idx]
                            bound_max[idx] = first_bound_max[first_idx]

                            first_idx += 1

                        try:

                            popt, pcov=curve_fit(astrokit.gauss_multi,
                                                 vel,
                                                 cube[0].data[:, idx_ax2, idx_ax1],
                                                 sigma = rms_spect,
                                                 p0 = guess,
                                                 bounds = (bound_min, bound_max),
                                                 maxfev = 1000)

                        except:

                            fill_param = False


                        gauss+=1

            else:

                fill_param = False

                fit_cube[0].data[:, idx_ax2, idx_ax1] = np.nan


            if fill_param:

                fit_cube[0].data[:, idx_ax2, idx_ax1] = popt[:]

    return fit_cube

def cluster_size(domain_map,
                 dist,
                 dist_err = 0):

    res_ra = abs(domain_map[0].header["CDELT1"])
    res_dec = abs(domain_map[0].header["CDELT2"])

    pix_size = Angle(res_ra*u.deg).rad * Angle(res_dec*u.deg).rad

    num_cluster = int(np.nanmax(domain_map[0].data)+1)

    cluster_size = np.zeros(num_cluster)
    cluster_size_err = np.zeros(num_cluster)

    for cluster in range(num_cluster):

        num_pix =  np.sum(domain_map[0].data == cluster)

        cluster_size[cluster] = dist**2 * pix_size * num_pix

        cluster_size_err[cluster] = 2. * dist * dist_err * pix_size * num_pix

    return cluster_size, cluster_size_err
