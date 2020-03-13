"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow, StartWindow, StartStopWindow, StopWindow, TestShortWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT, IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import *  #allocates bits to scale factor bands given SMRs
from sbr import *
from scipy.ndimage import gaussian_filter1d
from math import floor, ceil
import matplotlib.pyplot as plt
from scipy import interpolate
from gain_shape_quantize import quantize_gain_shape, dequantize_gain_shape, mu_law_fn, inv_mu_law_fn

K_FINE = 0
SHORT = 256


def getCorrectWindow(lastTrans, curTrans, nextTrans, Nlong=2048):
    if curTrans:
        return lambda x: SineWindow(
            x)  # lambda x: TestShortWindow(x, Nlong, SHORT)

    if lastTrans and nextTrans:
        return lambda x: StartStopWindow(x, Nlong, SHORT)

    if lastTrans:
        return lambda x: StopWindow(x, Nlong, SHORT)

    if nextTrans:
        return lambda x: StartWindow(x, Nlong, SHORT)

    return lambda x: SineWindow(x)


def Decode(scaleFactor,
           bitAlloc,
           mantissa,
           overallScaleFactor,
           pb,
           codingParams,
           lastTrans=False,
           curTrans=False,
           nextTrans=False):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    if curTrans:
        sfBands = codingParams.sfBandsShort
    else:
        sfBands = codingParams.sfBands

    rescaleLevel = 1. * (1 << overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    # vectorizing the Dequantize function call
    # vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN, dtype=np.float64)
    iMant = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            if codingParams.useVQ:
                # Reconstruct MDCT lines from VQ
                mdctLine[iMant:(iMant + nLines)] = dequantize_gain_shape(
                    pb, int(bitAlloc[iBand] * nLines), nLines, k_fine=K_FINE)
            else:
                mdctLine[iMant:(iMant + nLines)] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:(iMant + nLines)],
                    codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level

    # IMDCT and window the data for this channel
    window = getCorrectWindow(lastTrans, curTrans, nextTrans, N)
    data = window(IMDCT(mdctLine, halfN, halfN))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Decode_SBR(scaleFactor,
               bitAlloc,
               mantissa,
               overallScaleFactor,
               pb,
               codingParams,
               lastTrans=False,
               curTrans=False,
               nextTrans=False):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object
    Accounts for spectral band replication via params in codingParams"""

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

        if iBand in codingParams.omittedBands:
            nLines = 1

        if bitAlloc[iBand]:
            if codingParams.useVQ:
                # Reconstruct MDCT lines from VQ
                mdctLine[iMant:(iMant + nLines)] = dequantize_gain_shape(
                    pb, int(bitAlloc[iBand] * nLines), nLines, k_fine=K_FINE)
            else:
                mdctLine[iMant:(iMant + nLines)] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:(iMant + nLines)],
                    codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines

    # we can smooth this step function of the envelope with a Gaussian kernel to generate a nicer step function
    # the sigma function here controls the smoothing--probably needs some tuning
    omit_cutoff = codingParams.sfBands.lowerLine[codingParams.omittedBands[0]]
    num_omitted = len(mdctLine) - omit_cutoff
    envelope = mdctLine[omit_cutoff:]
    envelope = np.append(envelope, np.zeros_like(envelope))

    #plt.figure(figsize=(12, 8))
    #plt.title('Envelope')
    #plt.semilogx(np.arange(2*num_omitted), envelope)
    #plt.show()

    smoothed_envelope = gaussian_filter1d(envelope, sigma=200)

    #plt.figure(figsize=(12, 8))
    #plt.title('Smoothed Envelope')
    #plt.semilogx(np.arange(2*num_omitted), smoothed_envelope)
    #plt.show()

    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines before transposition')
    #plt.semilogx(np.arange(len(mdctLine)), np.abs(mdctLine)**2)
    #plt.axvline(x=omit_cutoff, color='r')
    #plt.show()

    # now, replicate and then multiply by the smoothed envelope
    factor = len(mdctLine) / num_omitted
    factor = floor(factor)
    up_factor = factor

    mdct_spacing = codingParams.sampleRate / (2 * codingParams.nMDCTLines)
    mdct_freqs = (np.arange(codingParams.nMDCTLines) + 1 / 2) * (mdct_spacing)

    # this works i promise. This interpolates the lower frequencies in order to transpose upward
    # This I expect to be helpful because mdct has an offset we need to acct for
    interp_indices = np.arange(omit_cutoff // up_factor - 1,
                               len(mdctLine) // up_factor + 1)
    interp_x = mdct_freqs[interp_indices]
    interp_y = mdctLine[interp_indices]

    # for now, do a piecewise linear interpolation. Could try doing quadratic or cubic too
    interp_fn = interpolate.interp1d(interp_x, interp_y, kind='slinear')
    new_freqs = mdct_freqs[omit_cutoff:] / up_factor

    filers = interp_fn(new_freqs)

    mdctLine[omit_cutoff:] = filers

    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines without envelope adjustment')
    #plt.semilogx(np.arange(len(mdctLine)), np.abs(mdctLine)**2)
    #plt.axvline(x=omit_cutoff, color='r')
    #plt.show()

    # adjust amplitudes by envelope (make zero-safe)
    for iBand in codingParams.omittedBands:
        lowLine = codingParams.sfBands.lowerLine[iBand]
        highLine = codingParams.sfBands.upperLine[iBand] + 1
        if np.max(np.abs(mdctLine[lowLine:highLine])) > 0:
            mdctLine[lowLine:highLine] *= \
                    smoothed_envelope[lowLine-omit_cutoff:highLine-omit_cutoff]/np.mean(np.abs(mdctLine[lowLine:highLine]))

    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines with envelope adjustment')
    #plt.semilogx(np.arange(len(mdctLine)), np.abs(mdctLine)**2)
    #plt.axvline(x=omit_cutoff, color='r')
    #plt.show()

    # add some noise to these high frequency components to sound a bit more natural
    # need to tune scale, which is the standard deviation of each sample
    # right now, the standard deviation of the noise depends on the amplitude of the frequency line (should make sense, right?)
    # mdctLine[omit_cutoff:] += np.random.normal(loc=np.zeros(num_omitted), scale=abs(mdctLine[omit_cutoff:])/10, size=num_omitted)

    #plt.figure(figsize=(12, 8))
    #plt.title('MDCT Lines with envelope adjustment and noise')
    #plt.semilogx(np.arange(num_omitted), np.abs(mdctLine[omit_cutoff:])**2)
    #plt.show()

    # additional smoothing??
    # print('mdct line', mdctLine[np.where(mdctLine > 0)[0]])

    mdctLine /= rescaleLevel  # put overall gain back to original level

    # IMDCT and window the data for this channel
    window = getCorrectWindow(lastTrans, curTrans, nextTrans, N)
    data = window(IMDCT(mdctLine, halfN, halfN))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,
           codingParams,
           lastTrans=False,
           curTrans=False,
           nextTrans=False):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    if codingParams.useVQ:
        bitAlloc = []
        indices = []
        idx_bits = []
        overall_scale = []

        # loop over channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (b, m, o, s) = EncodeSingleChannel(data[iCh], codingParams,
                                               lastTrans, curTrans, nextTrans)
            bitAlloc.append(b)
            indices.append(m)
            idx_bits.append(o)
            overall_scale.append(s)
        # return results bundled over channels
        return (bitAlloc, indices, idx_bits, overall_scale)

    else:
        scaleFactor = []
        bitAlloc = []
        mantissa = []
        overallScaleFactor = []

        # loop over channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (s, b, m, o) = EncodeSingleChannel(data[iCh], codingParams,
                                               lastTrans, curTrans, nextTrans)
            scaleFactor.append(s)
            bitAlloc.append(b)
            mantissa.append(m)
            overallScaleFactor.append(o)
        # return results bundled over channels
        return (scaleFactor, bitAlloc, mantissa, overallScaleFactor)


def EncodeSingleChannel(data,
                        codingParams,
                        lastTrans=False,
                        curTrans=False,
                        nextTrans=False):
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
    if curTrans:
        sfBands = codingParams.sfBandsShort
    else:
        sfBands = codingParams.sfBands  # vectorizing the Mantissa function call
    # vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    NforBitBudget = int(1.45 * halfN) if curTrans else halfN
    if lastTrans or nextTrans:
        NforBitBudget = int(0.85 * NforBitBudget)
    bitBudget = codingParams.targetBitsPerSample * NforBitBudget  # this is overall target bit rate
    if codingParams.useVQ:
        bitBudget -= nScaleBits  # Overall scale factor
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less bit allocation bits
    else:
        bitBudget -= nScaleBits * (
            sfBands.nBands + 1
        )  # less scale factor bits (including overall scale factor)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less mantissa bit allocation bits

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    window = getCorrectWindow(lastTrans, curTrans, nextTrans, N)
    mdctTimeSamples = window(data)
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

    # mdct_spl = SPL(4.0 * ((mdctLines / (2**overallScale))**2.0))
    # peaks_in_bands = np.zeros(sfBands.nBands)
    # if codingParams.useVQ:
    #     mdctLines /= (1 << overallScale)

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

    if codingParams.useVQ:
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
                band_budget = int(bitAlloc[iBand] * nLines)
                indices, bits = quantize_gain_shape(x,
                                                    band_budget,
                                                    k_fine=K_FINE)
                if sum(bits) == 0:
                    bitAlloc[iBand] = 0
                else:
                    all_indices.append(indices)
                    all_idx_bits.append(bits)
        # end of loop over scale factor bands

        # return results
        return (bitAlloc, all_indices, all_idx_bits, overallScale)
    else:
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
                    mdctLines[lowLine:highLine], scaleFactor[iBand],
                    nScaleBits, bitAlloc[iBand])
                iMant += nLines
        # end of loop over scale factor bands

        # return results
        return (scaleFactor, bitAlloc, mantissa, overallScale)


def Encode_SBR(data,
               codingParams,
               lastTrans=False,
               curTrans=False,
               nextTrans=False):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    if codingParams.useVQ:
        bitAlloc = []
        indices = []
        idx_bits = []
        overall_scale = []

        # loop over channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (b, m, o, s) = EncodeSingleChannel_SBR(data[iCh], codingParams,
                                                   lastTrans, curTrans,
                                                   nextTrans)
            bitAlloc.append(b)
            indices.append(m)
            idx_bits.append(o)
            overall_scale.append(s)
        # return results bundled over channels
        return (bitAlloc, indices, idx_bits, overall_scale)

    else:
        scaleFactor = []
        bitAlloc = []
        mantissa = []
        overallScaleFactor = []

        # loop over channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (s, b, m, o) = EncodeSingleChannel_SBR(data[iCh], codingParams,
                                                   lastTrans, curTrans,
                                                   nextTrans)
            scaleFactor.append(s)
            bitAlloc.append(b)
            mantissa.append(m)
            overallScaleFactor.append(o)
        # return results bundled over channels
        return (scaleFactor, bitAlloc, mantissa, overallScaleFactor)


def EncodeSingleChannel_SBR(data,
                            codingParams,
                            lastTrans=False,
                            curTrans=False,
                            nextTrans=False):
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
    window = getCorrectWindow(lastTrans, curTrans, nextTrans, N)
    mdctTimeSamples = window(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]
    fftLines = np.abs(
        np.fft.rfft(HanningWindow(data))
    ) / codingParams.nMDCTLines  # compute fft for transmitting the envelope of the signal
    maxFFTLine = np.max(fftLines)

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max(np.abs(mdctLines))
    maxLine = max(
        maxLine,
        maxFFTLine)  # make sure we are all in the range (-1, 1) FFT and MDCT
    overallScale = ScaleFactor(
        maxLine, nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1 << overallScale)
    fftLines *= (1 << overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale,
                    codingParams.sampleRate, sfBands)

    # perform bit allocation using SMR results
    bitAlloc = BitAlloc_SBR(bitBudget, maxMantBits, sfBands.nBands,
                            sfBands.nLines, SMRs, omittedBands)
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
    if codingParams.useVQ:
        all_indices = []
        all_idx_bits = []
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[
                iBand] + 1  # extra value is because slices don't include last value
            nLines = sfBands.nLines[iBand]
            if iBand in omittedBands:
                nLines = 1

            if bitAlloc[iBand]:
                x = mdctLines[lowLine:
                              highLine]  # Input vector for vector quantization
                if iBand in omittedBands:
                    x = fftLines[lowLine:highLine]
                    x = np.array([np.mean(np.abs(x))])

                # perform the vector quantization
                band_budget = int(bitAlloc[iBand] * nLines)
                indices, bits = quantize_gain_shape(x,
                                                    band_budget,
                                                    k_fine=K_FINE)
                if sum(bits) == 0:
                    bitAlloc[iBand] = 0
                else:
                    all_indices.append(indices)
                    all_idx_bits.append(bits)
                if sum(bits) < bitAlloc[iBand] and iBand < sfBands.nBands - 2:
                    # Distribute unused bands to subsequent band
                    bitAlloc[iBand + 1] += bitAlloc[iBand] - sum(bits)
        # end of loop over scale factor bands

        # return results
        return (bitAlloc, all_indices, all_idx_bits, overallScale)
    else:
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
                    scaleLine = np.mean(fftLines[lowLine:highLine])
                    scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits,
                                                     bitAlloc[iBand])
                    mantissa[iMant] = vMantissa(scaleLine, scaleFactor[iBand],
                                                nScaleBits, bitAlloc[iBand])
                    iMant += 1
                else:
                    mantissa[iMant:iMant + nLines] = vMantissa(
                        mdctLines[lowLine:highLine], scaleFactor[iBand],
                        nScaleBits, bitAlloc[iBand])
                    iMant += nLines

        # return results
        return (scaleFactor, bitAlloc, mantissa, overallScale)
