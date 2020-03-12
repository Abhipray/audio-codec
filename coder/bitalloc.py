import numpy as np
import matplotlib.pyplot as plt
from mdct import *
from window import *
from psychoac import *
from quantize import *
from sbr import *

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    avg_bits_per_line = bitBudget / np.sum(nLines)
    bits = []
    for n in range(nBands):
        bits.append(min([np.floor(avg_bits_per_line * nLines[n]), maxMantBits]))
    return bits

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    bits = []
    for n in range(nBands):
        nBits = min([max([np.floor((peakSPL[n] - 6) / 6), 0]), maxMantBits, bitBudget])
        bitBudget -= nBits
        bits.append(nBits)
    return bits

def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    bits = []
    for n in range(nBands):
        if SMR[n] < -6:
            bits.append(0)
            continue
    
        bits_wanted = (SMR[n] + 6) / 6
        bits.append(min([np.floor(bits_wanted), maxMantBits]))
        bitBudget -= bits[-1]

    return list(np.asarray(bits, dtype=np.int16))


# Question 1.c)
DBTOBITS=6.2
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band    
    """
    # sanity check
    if maxMantBits > 16: maxMantBits = 16

    bits = np.zeros(nBands, dtype=int)
    flagged = np.zeros(nBands, dtype=bool)
    Nflip = 0 # index to find needle for round
    level_flip = 0. # needle for round
    count = 0
    while True:
        valid = np.logical_not(flagged)
        N = np.sum(nLines[valid]) # consider only lines that will be encoded
        if N == 0:
            N += 1e-12
            
        # optimal allocation
        Ropt = bitBudget / N + \
            (1.0/ DBTOBITS) * (SMR[valid] - np.sum(nLines[valid]*SMR[valid]) / N)
        Dopt = np.sort((Ropt - np.floor(Ropt) -0.5)[Ropt - np.floor(Ropt) - 0.5 > 0])
        
        if Nflip > len (Dopt):
            Nflip -= 1
            Ropt = np.floor(Ropt)
        else:
            if Nflip == 0:
                level_flip = 0.
            else:
                level_flip = Dopt[Nflip -1]
            bits[valid] = np.round(Ropt - level_flip)

        # sanity check
        bits [bits > maxMantBits] = maxMantBits
        flagged = bits < 2 # lonely and negative bits are flagged
        bits [flagged] = 0
        # import pdb; pdb.set_trace()

        if(np.logical_xor(flagged, valid).all() & (np.sum(np.multiply(bits, nLines)) <= bitBudget)):
            break
        if(np.logical_xor(flagged, valid).all() & (np.sum(np.multiply(bits, nLines)) > bitBudget)):
            Nflip += 1
        
        count += 1
        if count > 200:
            # print('stuck in loop!')
            break

#     print('spent vs budget', np.sum(bits * nLines) - bitBudget)
    return bits

# Question 1.c)
#def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
#    """
#    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum
#
#       Arguments:
#           bitBudget is total number of mantissa bits to allocate
#           maxMantBits is max mantissa bits that can be allocated per line
#           nBands is total number of scale factor bands
#           nLines[nBands] is number of lines in each scale factor band
#           SMR[nBands] is signal-to-mask ratio in each scale factor band
#
#        Returns:
#            bits[nBands] is number of bits allocated to each scale factor band
#
#        Logic:
#           Maximizing SMR over blook gives optimization result that:
#               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
#           where P is the pool of bits for mantissas and N is number of bands
#           This result needs to be adjusted if any R(i) goes below 2 (in which
#           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
#           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
#           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
#           We will not bother to worry about slight variations in bit budget due
#           rounding of the above equation to integer values of R(i).
#        
#    """
#    # compute integer rounding of naive formula
#    avg_smr = np.mean(SMR)
#    naive_formula = (bitBudget/nBands + (SMR-avg_smr)/DBTOBITS)
#    bits = np.around(naive_formula/nLines).astype(np.int32)
#
#    # adjust for values that are too high and too low
#    bits[bits>maxMantBits] = maxMantBits
#    ones = np.where(bits==1)[0]
#    bits[bits<2] = 0
#
#    # calculate remaining bits, and assign to where we previously had 1 bit allocated if we can
#    remaining = bitBudget - np.sum(bits*nLines)
#
#    while remaining < 0:
#        bit_allocs = sorted(np.arange(nBands), key=lambda x: bits[x], reverse=True)
#        if bits[bit_allocs[0]] != 2:
#            bits[bit_allocs[0]] -= 1
#        else:
#            bits[bit_allocs[0]] -= 2
#
#        remaining = bitBudget - np.sum(bits*nLines)
#
#    # sort ones by SMRs
#    ones = sorted(ones, key=lambda x:naive_formula[x], reverse=True)
#
#    for idx in ones:
#        if nLines[idx]*2 <= remaining:
#            bits[idx] += 2
#            remaining -= nLines[idx]*2
#
#    # if there are still remaining bits, allocate by SMR
#    valid_positions = np.where(bits<maxMantBits)[0]
#    sorted_SMRs = sorted(valid_positions, key=lambda x:SMR[x], reverse=True)
#
#    for idx in sorted_SMRs:
#        if nLines[idx] <= remaining:
#            bits[idx] += 1
#            remaining -= nLines[idx]
#
#    return bits

def BitAlloc_SBR(bitBudget, maxMantBits, nBands, nLines, SMR, omittedBands):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum
    Accounts for spectral band replication by applying maxMantBits for each band over
    the SBR threshold. This is because for each of these bands, we will only send one
    floating point number to encapsulate the spectral envelope. 

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band
           omittedBands is the list of band indices which are not being sent b/c they will be reconstructed via SBR

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band    
    """
    # sanity check

    # we are sending LINES_PER_OMIT lines, instead of the original nLines, for all the SBR'ed bands
    for b in omittedBands:
        nLines[b] = LINES_PER_OMIT

    return BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR)


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    Fs = 48000
    N = 1024
    freqs = np.array([440, 550, 660, 880, 4400, 8800])
    amps = np.array([0.43, 0.24, 0.15, 0.09, 0.05, 0.04])

    x = np.sum(np.array([amps[i] * np.cos(2 * np.pi * freqs[i] * np.arange(N) / Fs)
        for i in range(6)]), axis=0)

    avg_power = lambda sig : (1/N) * np.sum(sig**2)
    X = MDCT(KBDWindow(x), N//2, N//2)
    intensity = (8 / (N**2 * avg_power(KBDWindow(np.ones(N))))) * np.abs(X)**2
    X_spl = SPL(intensity)
    dct_freqs = np.fft.rfftfreq(N-1, d=1/Fs)

    barks = Bark(dct_freqs)
    peak_spls = np.zeros(N//2)
    peaks_dumb = []
    cur_peak = -100
    cur_bark = 0
    last_bark_ind = 0
    for i, b in enumerate(barks):
        if np.floor(b) > cur_bark:
            cur_bark = np.floor(b)
            peak_spls[last_bark_ind:i] = cur_peak
            peaks_dumb.append(cur_peak)

            cur_peak = -100
            last_bark_ind = i
        
        if X_spl[i] > cur_peak:
            cur_peak = X_spl[i]
    
    peak_spls[last_bark_ind:] = cur_peak
    peaks_dumb.append(cur_peak)

    mask = getMaskedThreshold(x, X, 0, Fs, None)[:-1]
    mask_bark = []
    cur_mask = mask[0]
    cur_bark = 0
    for i, b in enumerate(barks):
        if np.floor(b) > cur_bark:
            cur_bark = np.floor(b)
            mask_bark.append(cur_mask)
            cur_mask = mask[i]
        
        if mask[i] > cur_mask:
            cur_mask = mask[i]
    mask_bark.append(cur_mask)

    # Problem 1.c
    budget = 1711 # 128 kb/s/ch
    nLines = AssignMDCTLinesFromFreqLimits(N//2, Fs)
    sfBands = ScaleFactorBands(nLines)
    smr = np.asarray(peaks_dumb) - np.asarray(mask_bark)
    # smr = CalcSMRs(x, X, 0, Fs, sfBands)

    bAllocUniform = { 'bits': BitAllocUniform(budget, 16, 25, nLines), 'title': 'Noise Floor (128 kb/s, Uniform Allocation)' }
    bAllocSNR = { 'bits': BitAllocConstSNR(budget, 16, 25, nLines, peaks_dumb), 'title': 'Noise Floor (128 kb/s, Const SNR Allocation)' }
    bAllocNMR = { 'bits': BitAllocConstNMR(budget, 16, 25, nLines, smr), 'title': 'Noise Floor (128 kb/s, Const NMR Allocation)' }

    def plotNoiseFloor(floor, title=''):
        plt.figure()
        plt.semilogx(dct_freqs,X_spl, label='Signal')
        # plt.semilogx(dct_freqs,peak_spls)
        plt.semilogx(dct_freqs,floor, label='Noise Floor')
        plt.semilogx(dct_freqs,mask, label='Mask Threshold')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('SPL')
        plt.title(title)
        plt.ylim(-30, 100)

    def doForStuff(listOfStuff):
        for bAlloc in listOfStuff:
            print(bAlloc['title'])
            print(bAlloc['bits'])

            noise_floor = np.zeros(N//2)
            last_bark_ind = 0
            cur_bark = 0
            for i, b in enumerate(barks):
                if np.floor(b) > cur_bark:
                    cur_bark = int(np.floor(b))
                    noise_floor[last_bark_ind:i] = peak_spls[i-1] - 6 * bAlloc['bits'][cur_bark-1]
                    last_bark_ind = i

            noise_floor[last_bark_ind:] = peak_spls[-1]
            plotNoiseFloor(noise_floor, bAlloc['title'])

    # doForStuff([bAllocUniform, bAllocSNR, bAllocNMR])
    doForStuff([bAllocNMR])

    # Problem 1.d
    budget = 1852 # 196 kb/s/ch

    bAllocUniform = { 'bits': BitAllocUniform(budget, 16, 25, nLines), 'title': 'Noise Floor (192 kb/s, Uniform Allocation)' }
    bAllocSNR = { 'bits': BitAllocConstSNR(budget, 16, 25, nLines, peaks_dumb), 'title': 'Noise Floor (192 kb/s, Const SNR Allocation)' }
    bAllocNMR = { 'bits': BitAllocConstNMR(budget, 16, 25, nLines, smr), 'title': 'Noise Floor (192 kb/s, Const NMR Allocation)' }

    # doForStuff([bAllocUniform, bAllocSNR, bAllocNMR])
    doForStuff([bAllocNMR])


    # Problem 2.a
    bAllocReal = { 'bits': BitAlloc(budget, 16, 25, nLines, smr), 'title': 'Real BitAlloc (192 kb/s, Uniform Allocation)' }
    # doForStuff([bAllocReal])

    # ScaleFactor
    # vMantissa()
    # vDequantize
    
    plt.show()

    pass # TO REPLACE WITH YOUR CODE
