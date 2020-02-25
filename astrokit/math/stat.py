import numpy as np

def mean(values, weight = None, method = 'simple'):

    if method == 'simple':

        mean_value = np.nansum(values)/len(values)

    if method == 'weighted':

        mean_value = np.nansum(values*weight)/sum(weight)

    return mean_value

def variance(values, mean_value = None):

    if mean_value == None:

         mean_value = mean(values)

    var_value = np.nansum((values-mean_value)**2)/len(values)

    return var_value

def rms(values, mean_value = None, method = 'rms'):

    if mean_value == None:

         mean_value = mean(values)

    var_value  = variance(values, mean_value)

    rms_value  = np.sqrt(var_value)

    if method == 'rms':

        return rms_value

    elif method == 'fwhm':

        return rms_value*2.*np.sqrt(2.*np.log(2))
