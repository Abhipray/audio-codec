import numpy as np
import matplotlib.pyplot as plt
from mdct import *
from window import *
from psychoac import *
from quantize import *

SBR_FACTOR = 2

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
    while True:
        valid = np.logical_not(flagged)
        N = np.sum(nLines[valid]) # consider only lines that will be encoded
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

    return bits

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
    if maxMantBits > 16: maxMantBits = 16

    bitBudget -= len(omittedBands)*maxMantBits
    remaining_bands = nBands - len(omittedBands)
    bits_normal = BitAlloc(bitBudget, maxMantBits, remaining_bands, nLines[:remaining_bands], SMR[:remaining_bands])
    bits_extend = np.ones(len(omittedBands))*maxMantBits
    return np.concatenate((bits_normal, bits_extend)).astype(np.int32)

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

    # Problem 1.c
    budget = 1711 # 128 kb/s/ch
    nLines = AssignMDCTLinesFromFreqLimits(N//2, Fs)
    sfBands = ScaleFactorBands(nLines)
    smr = CalcSMRs(x, X, 0, Fs, sfBands)

    bAllocUniform = { 'bits': BitAllocUniform(budget, 16, 25, nLines), 'title': 'Noise Floor (128 kb/s, Uniform Allocation)' }
    bAllocSNR = { 'bits': BitAllocConstSNR(budget, 16, 25, nLines, smr), 'title': 'Noise Floor (128 kb/s, Const SNR Allocation)' }
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

    # Problem 1.d
    budget = 1711 # 196 kb/s/ch

    bAllocUniform = { 'bits': BitAllocUniform(budget, 16, 25, nLines), 'title': 'Noise Floor (192 kb/s, Uniform Allocation)' }
    bAllocSNR = { 'bits': BitAllocConstSNR(budget, 16, 25, nLines, smr), 'title': 'Noise Floor (192 kb/s, Const SNR Allocation)' }
    bAllocNMR = { 'bits': BitAllocConstNMR(budget, 16, 25, nLines, smr), 'title': 'Noise Floor (192 kb/s, Const NMR Allocation)' }

    # doForStuff([bAllocUniform, bAllocSNR, bAllocNMR])


    # Problem 2.a
    bAllocReal = { 'bits': BitAlloc_SBR(budget, 16, 25, nLines, smr), 'title': 'Real BitAlloc (192 kb/s, Uniform Allocation)' }
    # doForStuff([bAllocReal])
    print(bAllocReal)

    # ScaleFactor
    # vMantissa()
    # vDequantize
    
    plt.show()

    pass # TO REPLACE WITH YOUR CODE
