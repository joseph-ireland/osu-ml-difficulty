"""
Raleigh distribution gives the distance from the mean of a 2d normal distribution
see: https://en.wikipedia.org/wiki/Rayleigh_distribution
"""

import numpy as np


def CDF(x, sigma=1):
    return 1 - np.exp(-x**2 / (2 * sigma**2))

def CDF_inv(z, sigma=1):
    return np.sqrt(-2 * sigma**2 * np.log(1 - z))

def PDF(x, sigma=1):
    return (x / sigma**2) * np.exp(-x**2 / (2*sigma**2))

def mle_sigma(samples):
    return np.sqrt(np.sum(np.square(samples))/(2*samples.shape[0]))
