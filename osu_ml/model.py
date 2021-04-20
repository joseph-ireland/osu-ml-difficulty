import numpy as np

from osu_ml import raleigh


_z0 = raleigh.CDF_inv(0.96)

def hit_probability(throughput, time, distance):
    return raleigh.CDF((_z0 / distance) * (np.exp2(throughput*time)-1))

def error_cfd_fitts(throughput, time, distance, error, toffset=0):
    return raleigh.CDF(error * (_z0 / distance) * (np.exp2(throughput*(time-toffset))-1))

def error_pdf_fitts(throughput, time, distance, error):
    return raleigh.PDF(error * (_z0 / distance) * (np.exp2(throughput*time)-1))

mean_multiplier = np.sqrt(0.5*np.pi)
def throughput_estimate_fitts(time, distance, error):
    return (1/time)*np.log2(1+(distance*mean_multiplier/(error*_z0)))
