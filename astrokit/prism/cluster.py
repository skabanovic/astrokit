from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import numpy as np
import astrokit
import copy

def cluster_information(cube,
                        cluster_range,
                        methode = 'BIC',
                        reduce_dim = 'pca',
                        pca_norm = 'mean',
                        gmm_iter = 1000,
                        sample = None):

    # Dimension of the input cube
    dim_ax1 = cube[0].header['NAXIS1'] # Points in Y/Dec axis
    dim_ax2 = cube[0].header['NAXIS2'] # Points in X/RA axis
    dim_ax3 = cube[0].header['NAXIS3'] # Points in velocity

    # Check: if a grid point is masked with nan at velocity v it should be masked at all v
    for idx_ax1 in range(dim_ax1):
        for idx_ax2 in range(dim_ax2):
            if np.isnan(np.sum(cube[0].data[:,idx_ax2, idx_ax1])):
                cube[0].data[:,idx_ax2,idx_ax1] = np.nan

    # From a [dimV,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_data_masked = cube[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()

    # Consider only the points that are not masked
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked,axis=1))]

    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        if pca_norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif pca_norm == 'std':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(0.99, whiten=True)

        reduced_data = pca.fit_transform(normed_flat_data)

        print('PCA done')
        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(reduced_data)[1])+' dimensions')

    else:

        reduced_data = flat_data

    # REMEMBER: things will slightly change if you ask for, say 10000 spectra.

    print('Starting the computation of Bayesian Information Criterion')

    # We compute the BIC on a random sample of the data
    # Consider a random sample

    if sample:

        indices = np.random.choice(reduced_data.shape[0], sample, replace=False)

        sample_data = reduced_data[indices]

    else:

        sample_data = reduced_data

    info_criterion = []

    cluster_list = np.arange(cluster_range[0], cluster_range[1], 1)

    for cluster in cluster_list:

        model = GaussianMixture(n_components = cluster,
                                covariance_type='full',
                                max_iter = gmm_iter).fit(sample_data)

        if methode == 'BIC':

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
                       gmm_iter = 1000,
                       reduce_dim = 'original',
                       pca_norm = 'mean'):

    # Dimension of the input cube
    dim_ax1 = cube[0].header['NAXIS1'] # Points in Y/Dec axis
    dim_ax2 = cube[0].header['NAXIS2'] # Points in X/RA axis
    dim_ax3 = cube[0].header['NAXIS3'] # Points in velocity

    # Check: if a grid point is masked with nan at velocity v it should be masked at all v
    for idx_ax1 in range(dim_ax1):
        for idx_ax2 in range(dim_ax2):
            if np.isnan(np.sum(cube[0].data[:,idx_ax2, idx_ax1])):
                cube[0].data[:,idx_ax2,idx_ax1] = np.nan

    # From a [dimV,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_data_masked = cube[0].data.reshape(dim_ax3, dim_ax2*dim_ax1).transpose()

    # Consider only the points that are not masked
    flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked,axis=1))]

    if reduce_dim == 'pca':

        # PCA will assume that the dataset has zero mean
        # We need to standardize each spectra

        if pca_norm == 'mean':
            normed_flat_data = flat_data - np.mean(flat_data, axis = 0)

        elif pca_norm == 'std':
            normed_flat_data = (flat_data - np.mean(flat_data,axis = 0))/np.std(flat_data,axis = 0)

        # we are asking to reduce the dimensionality and still retain 99% of the dataset variance
        pca = PCA(0.99, whiten=True)

        reduced_data = pca.fit_transform(normed_flat_data)

        print('PCA done')
        print('Feature space is reduced from '+str(dim_ax3)+' to '+str(np.shape(reduced_data)[1])+' dimensions')

    elif reduce_dim == 'original':

        reduced_data = flat_data

    gmm = GaussianMixture(n_components = n_com,
                          covariance_type='full',
                          random_state=42,
                          max_iter = gmm_iter).fit(reduced_data)


    labels = gmm.predict(reduced_data)

    # Number of spectra
    n_ts = np.shape(flat_data)[0]

    # Step 2: go from index 1 to index n_com and find the respective nodes
    # initialize
    modes_indices = []

    for i in range(n_com):
        modes_indices.append(duplicates(labels, i))


    mode_signal_mean = []
    mode_signal_std = []
    for i in range(len(modes_indices)):
        # Extract the mode
        extract_mode = np.array(flat_data)[modes_indices[i]]
        # Compute the signal
        signal = np.mean(extract_mode,axis=0)
        # Compute the std
        std = np.std(extract_mode,axis=0)
        # Store the result
        mode_signal_mean.append(signal)
        mode_signal_std.append(std)

    # Find the mapping
    # mapping[i] will tell you the mapping from flat_data to flat_data_masked
    # Specifically:
    # flat_data[i] will correspond to an index mapping[i] = k, where k is such that flat_data_masked[k] == flat_data[i]
    mapping = []

    # From an array to a list
    list_flat_data_masked = flat_data_masked.tolist()
    list_flat_data = flat_data.tolist()

    for i in range(len(flat_data)):
        mapping.append(list_flat_data_masked.index(list_flat_data[i]))


    # Here we store the domains maps (clusters embedded in the grid)
    domains = []
    for i in range(len(modes_indices)):

        # Let's create a temporary spatial grid
        data_grid  = cube[0].data[0,:,:].copy()
        # Let's flatten it
        flat_data_grid = data_grid.flatten()
        # If is not a nan is a zero
        flat_data_grid[~np.isnan(flat_data_grid)] = 0

        for j in range(len(modes_indices[i])):

            # set to 1 the pixel belonging to a domain
            flat_data_grid[mapping[modes_indices[i][j]]] = 1

        domains.append(flat_data_grid)


    #  Gridded domains (just have to reshape the fucker)
    #gridded_domains = np.zeros([dimY,dimX,len(domains)])

    domain_map = []

    for i in range(len(domains)):

        gridded_domains = astrokit.zeros_map(cube)
        gridded_domains[0].data = domains[i].reshape(dim_ax2, dim_ax1)
        domain_map.append(copy.deepcopy(gridded_domains))


    return mode_signal_mean, mode_signal_std, domain_map
