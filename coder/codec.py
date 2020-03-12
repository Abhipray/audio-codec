"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT, IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import *  #allocates bits to scale factor bands given SMRs
from gain_shape_quantize import quantize_gain_shape, dequantize_gain_shape

K_FINE = -2


def Decode(pb, bitAlloc, overallScaleFactor, codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1. * (1 << overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    # vectorizing the Dequantize function call
    # vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN, dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines = codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            # Reconstruct MDCT lines from VQ
            mdctLine[iMant:(iMant + nLines)] = dequantize_gain_shape(
                pb, int(bitAlloc[iBand] * nLines), nLines, k_fine=K_FINE)
        iMant += nLines
    # mdctLine /= rescaleLevel  # put overall gain back to original level

    # IMDCT and window the data for this channel
    data = SineWindow(IMDCT(mdctLine, halfN,
                            halfN))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data, codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""

    bitAlloc = []
    indices = []
    idx_bits = []
    overall_scale = []
    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (b, m, o, s) = EncodeSingleChannel(data[iCh], codingParams)
        bitAlloc.append(b)
        indices.append(m)
        idx_bits.append(o)
        overall_scale.append(s)
    # return results bundled over channels
    return (bitAlloc, indices, idx_bits, overall_scale)


def EncodeSingleChannel(data, codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (
        1 << codingParams.nMantSizeBits
    )  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits > 16:
        maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
    # vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits  # Overall scale factor
    bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less bit allocation bits

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    mdctTimeSamples = SineWindow(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max(np.abs(mdctLines))
    overallScale = ScaleFactor(
        maxLine, nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1 << overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale,
                    codingParams.sampleRate, sfBands)

    mdctLines /= (1 << overallScale)  #Renormalize back to usual

    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines,
                        SMRs)
    # print(bitAlloc)
    # print(np.sum(bitAlloc * sfBands.nLines), bitBudget)
    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands, dtype=np.int32)
    nMant = halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]:
            nMant -= sfBands.nLines[
                iBand]  # account for mantissas not being transmitted

    all_indices = []
    all_idx_bits = []

    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[
            iBand] + 1  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]

        if bitAlloc[iBand]:
            x = mdctLines[lowLine:
                          highLine]  # Input vector for vector quantization
            # perform the vector quantization
            # print('bit alloc: ', bitAlloc[iBand], len(x))
            band_budget = int(bitAlloc[iBand] * nLines)
            indices, bits = quantize_gain_shape(x, band_budget, k_fine=K_FINE)
            all_indices.append(indices)
            all_idx_bits.append(bits)

    # end of loop over scale factor bands

    # return results
    return (bitAlloc, all_indices, all_idx_bits, overallScale)
