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
from sbr import *
from scipy.ndimage import gaussian_filter1d
from math import floor, ceil
import matplotlib.pyplot as plt

def Decode(scaleFactor, bitAlloc, mantissa, overallScaleFactor, codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1. * (1 << overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    # vectorizing the Dequantize function call
    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN, dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines = codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant + nLines)] = vDequantize(
                scaleFactor[iBand], mantissa[iMant:(iMant + nLines)],
                codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level

    # IMDCT and window the data for this channel
    data = SineWindow(IMDCT(mdctLine, halfN,
                            halfN))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data

def Decode_SBR(scaleFactor, bitAlloc, mantissa, overallScaleFactor, codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object
    Accounts for spectral band replication via params in codingParams"""

    rescaleLevel = 1. * (1 << overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    # vectorizing the Dequantize function call
    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN, dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines = codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            if iBand in codingParams.omittedBands:

                # setting the whole band to a scalar value
                mdctLine[iMant:(iMant + nLines)] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:(iMant + 1)],
                    codingParams.nScaleBits, bitAlloc[iBand])
                iMant += 1
            else:
                mdctLine[iMant:(iMant + nLines)] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:(iMant + nLines)],
                    codingParams.nScaleBits, bitAlloc[iBand])
                iMant += nLines
    
    # currently, mdctLine in the omittedBands is storing the envelope, which is a step function
    # we can smooth this step function of the envelope with a Gaussian kernel to generate a nicer step function
    # the sigma function here controls the smoothing--probably needs some tuning
    omit_cutoff = codingParams.sfBands.lowerLine[codingParams.omittedBands[0]]
    envelope = mdctLine[omit_cutoff:]
    smoothed_envelope =  gaussian_filter1d(envelope, sigma=100)

    # now, replicate and then multiply by the smoothed envelope
    num_omitted = len(mdctLine)-omit_cutoff
    factor = len(mdctLine)/num_omitted
    factor = floor(factor)

    # this works i promise
    pieces = mdctLine[omit_cutoff//factor:len(mdctLine)//factor+1]
    filers =  np.repeat(pieces, factor)

    # if the factor does not perfectly divide omit_cutoff, we would be introducing an offset unless we accounted for this
    num_pre_pieces = factor - omit_cutoff % factor
    filers = filers[num_pre_pieces:]

    mdctLine[omit_cutoff:] =filers[:num_omitted]

    #plt.figure(figsize=(12, 8))
    #plt.title('Envelope')
    #plt.semilogx(np.arange(num_omitted), smoothed_envelope)
    #plt.show()


    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines without envelope adjustment')
    #plt.semilogx(np.arange(num_omitted), np.abs(mdctLine[omit_cutoff:])**2)
    #plt.show()

    # adjust amplitudes by envelope (make zero-safe)
    for iBand in codingParams.omittedBands:
        if np.max(np.abs(mdctLine[codingParams.sfBands.lowerLine[iBand]:codingParams.sfBands.upperLine[iBand]+1])) > 0:
            mdctLine[codingParams.sfBands.lowerLine[iBand]:codingParams.sfBands.upperLine[iBand]+1] *= \
                smoothed_envelope[codingParams.sfBands.lowerLine[iBand]-omit_cutoff:codingParams.sfBands.upperLine[iBand]+1-omit_cutoff]/np.max(np.abs(mdctLine[codingParams.sfBands.lowerLine[iBand]:codingParams.sfBands.upperLine[iBand]+1]))

    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines with envelope adjustment')
    #plt.semilogx(np.arange(num_omitted), np.abs(mdctLine[omit_cutoff:])**2)
    #plt.show()

    # add some noise to these high frequency components to sound a bit more natural
    # need to tune scale, which is the standard deviation of each sample
    # right now, the standard deviation of the noise depends on the amplitude of the frequency line (should make sense, right?)
    mdctLine[omit_cutoff:] += np.random.normal(loc=np.zeros(num_omitted), scale=abs(mdctLine[omit_cutoff:])/100, size=num_omitted)
    
    
    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines with envelope adjustment and noise')
    #plt.semilogx(np.arange(num_omitted), np.abs(mdctLine[omit_cutoff:])**2)
    #plt.show()

    # additional smoothing??
    # print('mdct line', mdctLine[np.where(mdctLine > 0)[0]])
    
    mdctLine /= rescaleLevel  # put overall gain back to original level

    # IMDCT and window the data for this channel
    data = SineWindow(IMDCT(mdctLine, halfN,
                            halfN))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data

def Encode(data, codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s, b, m, o) = EncodeSingleChannel(data[iCh], codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor, bitAlloc, mantissa, overallScaleFactor)


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
    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits * (
        sfBands.nBands + 1
    )  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less mantissa bit allocation bits

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

    # peaks_in_bands = np.zeros(sfBands.nBands)

    # for band in range(sfBands.nBands):
    #     lower_limit = sfBands.lowerLine[band]
    #     upper_limit = sfBands.upperLine[band] + 1
    #     peaks_in_bands[band] = np.amax(mdct_spl[lower_limit:upper_limit])
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
    mantissa = np.empty(nMant, dtype=np.int32)
    iMant = 0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[
            iBand] + 1  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]
        scaleLine = np.max(np.abs(mdctLines[lowLine:highLine]))
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits,
                                         bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant + nLines] = vMantissa(
                mdctLines[lowLine:highLine], scaleFactor[iBand], nScaleBits,
                bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)

def Encode_SBR(data, codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s, b, m, o) = EncodeSingleChannel_SBR(data[iCh], codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor, bitAlloc, mantissa, overallScaleFactor)


def EncodeSingleChannel_SBR(data, codingParams):
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
    omittedBands = codingParams.omittedBands
    # vectorizing the Mantissa function call
    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits * (
        sfBands.nBands + 1
    )  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less mantissa bit allocation bits

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

    # perform bit allocation using SMR results
    bitAlloc = BitAlloc_SBR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines,
                        SMRs, omittedBands)
    # print(bitAlloc)
    # print(np.sum(bitAlloc * sfBands.nLines), bitBudget)
    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands, dtype=np.int32)
    nMant = halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]:
            nMant -= sfBands.nLines[
                iBand]  # account for mantissas not being transmitted
        elif iBand in omittedBands:
            nMant -= (sfBands.nLines[iBand] - 1)
    mantissa = np.empty(nMant, dtype=np.int32)
    iMant = 0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[
            iBand] + 1  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]
        
        scaleLine = np.max(np.abs(mdctLines[lowLine:highLine]))

        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits,
                                         bitAlloc[iBand])
        if bitAlloc[iBand]:
            if iBand in omittedBands:
                mantissa[iMant] = MantissaFP(scaleLine, scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                iMant += 1
            else:
                mantissa[iMant:iMant + nLines] = vMantissa(
                    mdctLines[lowLine:highLine], scaleFactor[iBand], nScaleBits,
                    bitAlloc[iBand])
                iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)
