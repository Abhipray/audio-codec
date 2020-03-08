""" Functions for SBR """
import numpy as np

SBR_FACTOR = 2

def omitted_bands(sfBands, factor=SBR_FACTOR):
    sbr_thresh = sfBands.upperLine[-1]//factor
    omitted_bands = np.where(sfBands.lowerLine >= sbr_thresh)[0]
    return omitted_bands

def sbr_factor():
    return SBR_FACTOR
