from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import numpy as np
import astrokit
import copy
import pomegranate

def cluster_information(cube,
                        cluster_range,
                        weight=None,
                        methode = 'BIC',
                        reduce_dim = 'pca',
                        norm = 'mean',
                        gmm_iter = 1000,
                        sample = None):

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

    if weight:
        flat_weight_masked = weight[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()
        flat_weight = flat_weight_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    # Consider only the points that are not masked
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked,axis=1))]

    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        if norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'std':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(0.99, whiten=True)

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

    # REMEMBER: things will slightly change if you ask for, say 10000 spectra.

    print('Starting the computation of Bayesian Information Criterion')

    # We compute the BIC on a random sample of the data
    # Consider a random sample

    if sample:

        indices = np.random.choice(reduced_data.shape[0], sample, replace=False)

        sample_data = reduced_data[indices]

        if weight:

            sample_weight = flat_weight[indices]

    else:

        sample_data = reduced_data

        if weight:

            sample_weight = flat_weight



    info_criterion = []

    cluster_list = np.arange(cluster_range[0], cluster_range[1], 1)

    for cluster in cluster_list:

        if weight:

            #Make the GMM model using pomegranate
            model = pomegranate.gmm.GeneralMixtureModel.from_samples(
                pomegranate.MultivariateGaussianDistribution,   #Either single function, or list of functions
                n_components = cluster,     #Required if single function passed as first arg
                X = sample_data,     #data format: each row is a point-coordinate, each column is a dimension
                stop_threshold = 0.01,  #Lower this value to get better fit but take longer.
                max_iterations = 100
                )

            #Force the model to train again, using additional fitting parameters
            model.fit(
                X = sample_data,         #data format: each row is a coordinate, each column is a dimension
                weights = sample_weight,  #List of weights. One for each point-coordinate
                stop_threshold = 0.01,  #Lower this value to get better fit but take longer.
                max_iterations = 100
                            #   (sklearn likes better/slower fits than pomegrante by default)
                )

        else:


            model = GaussianMixture(n_components = cluster,
                                    covariance_type ='full',
                                    max_iter = gmm_iter).fit(sample_data)


        if methode == 'BIC':

            if weight:

                N_k = cluster-1.+cluster*dim_ax3+(cluster*dim_ax3*(dim_ax3-1.))/2.
                cluster_info_criterion = N_k*np.log(len(sample_data))-2.*np.sum(model.log_probability(sample_data))
                print('cluster = '+str(cluster)+' ---> BIC = '+str(cluster_info_criterion))

            else:

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
                       weight = None,
                       gmm_iter = 1000,
                       reduce_dim = 'original',
                       norm = 'mean'):

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

    if weight:
        flat_weight_masked = weight[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()
        flat_weight = flat_weight_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    # Consider only the points that are not masked
    flat_index = flat_index_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked, axis=1))]

    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        if norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif norm == 'z-score':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(0.99, whiten=True)

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

        if weight:

            #Make the GMM model using pomegranate
            model = pomegranate.gmm.GeneralMixtureModel.from_samples(
                pomegranate.MultivariateGaussianDistribution,   #Either single function, or list of functions
                n_components=n_com,     #Required if single function passed as first arg
                X = reduced_data,     #data format: each row is a point-coordinate, each column is a dimension
                init = 'first-k',
                stop_threshold = 0.001,  #Lower this value to get better fit but take longer.
                max_iterations = gmm_iter
                )

            #Force the model to train again, using additional fitting parameters
            model.fit(
                X=reduced_data,         #data format: each row is a coordinate, each column is a dimension
                weights = flat_weight,  #List of weights. One for each point-coordinate
                stop_threshold = 0.001,  #Lower this value to get better fit but take longer.
                max_iterations = gmm_iter
                            #   (sklearn likes better/slower fits than pomegrante by default)
                )

        else:


            model = GaussianMixture(n_components = n_com,
                                    covariance_type='full',
                                    random_state=42,
                                    max_iter = gmm_iter).fit(reduced_data)


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



def cluster_average(cube,
                    cluster_map,
                    weight_map = None):

    cluster_num = int(np.nanmax(cluster_map[0].data)) +1
    len_ax3 = cube[0].header['NAXIS3']
    cluster_spect = np.zeros([cluster_num, len_ax3])


    for cluster in range(cluster_num):

        if weight_map:

            cluster_spect[cluster, :] = np.average(cube[0].data[:, cluster_map[0].data == cluster],
                                                   weights = weight_map[0].data[cluster_map[0].data == cluster], axis = 1)
        else:

            cluster_spect[cluster, :] = np.average(cube[0].data[:, cluster_map[0].data == cluster], axis = 1)

        cluster_std = np.std(cube[0].data[:, cluster_map[0].data == cluster], axis = 1)


    return cluster_spect, cluster_std
