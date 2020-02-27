import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from window import *
from scipy.signal import find_peaks

test_smr = False


def SPL(intensity):
    """
    Returns the SPL corresponding to intensity 
    """
    spl = 96 + 10 * np.log10(abs(intensity))
    if type(intensity) is np.ndarray:
        spl[spl < -30] = -30
    elif spl < -30:
        spl = -30
    return spl


def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """
    return 10**((spl - 96) / 10)


def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    f[f < 10] = 10
    thresh = 3.64 * (f / 1000)**(-0.8) - 6.5 * np.exp(
        -0.6 * (f / 1000 - 3.3)**2) + 10**(-3) * (f / 1000)**4
    return thresh


def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    bark = 13.0 * np.arctan(0.76 * f / 1000.0) + 3.5 * np.arctan(
        (f / 7500.0)**2)

    return bark


class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """
    def __init__(self, f, SPL, isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.SPL = SPL
        self.isTonal = isTonal
        self.bark = Bark(f)
        if isTonal:
            self.drop = 16
        else:
            self.drop = 3

    def IntensityAtFreq(self, freq):
        """The intensity at frequency freq"""
        return self.vIntensityAtBark(Bark(freq))

    def IntensityAtBark(self, z):
        """The intensity at Bark location z"""

        dz = z - self.bark
        gain = 0
        if abs(dz) <= 0.5:
            gain = 0
        elif dz < 0.5:
            gain = (-27 * (abs(dz) - 0.5))
        else:
            gain = (-27 + 0.367 * max(self.SPL - 40, 0)) * (abs(dz) - 0.5)
        spl = self.SPL + gain - self.drop
        return Intensity(spl)

    def vIntensityAtBark(self, zVec):
        """The intensity at vector of Bark locations zVec"""
        dz = zVec - self.bark
        gain = np.zeros_like(zVec)
        gain[dz < -0.5] = -27 * (abs(dz[dz < -0.5]) - 0.5)
        gain[dz > 0.5] = (-27 + 0.367 * max(self.SPL - 40, 0)) * (
            abs(dz[dz > 0.5]) - 0.5)
        spl = self.SPL + gain - self.drop
        return Intensity(spl)


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = np.array([
    100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320,
    2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 24000
])


def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit=cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    bin_width = sampleRate / (2 * nMDCTLines)
    mdct_bin_centers = np.floor((flimit / bin_width - 0.5))

    assignments = mdct_bin_centers - np.concatenate([[-1],
                                                     mdct_bin_centers[:-1]])
    # Remove bins above Nyquist
    for i in range(len(assignments)):
        if flimit[i] > sampleRate / 2:
            assignments[i] = nMDCTLines - np.sum(assignments[0:i])
            assignments[i + 1:] = 0
            break
    return assignments


class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """
    def __init__(self, nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        self.nLines = np.array(nLines, dtype=np.int)
        self.nBands = len(nLines)

        self.lowerLine = np.zeros((self.nBands, ), dtype=np.int)
        self.upperLine = np.zeros((self.nBands, ), dtype=np.int)

        for i in range(1, len(nLines)):
            self.lowerLine[i] = self.lowerLine[i - 1] + nLines[i - 1]

        self.upperLine = (self.lowerLine + nLines - 1).astype(np.int)


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    # Take the fft and calculate SPL and then peaks
    N = len(data)
    window_power = np.mean(np.hanning(N)**2)
    norm = 4 / (N**2 * window_power)

    x_fft = np.fft.rfft(HanningWindow(data))
    x_fftfreq = np.fft.rfftfreq(N, d=1 / sampleRate)

    # Find the peaks
    peaks, spls = estimate_peaks(norm * abs(x_fft)**2, x_fftfreq)

    # Compute the MDCT frequencies
    n_mdct_lines = len(MDCTdata)
    mdct_freq_spacing = sampleRate / (2 * n_mdct_lines)
    mdct_freqs = mdct_freq_spacing * (np.arange(n_mdct_lines) + 0.5)

    # Add tonal maskers
    complete_mask = []
    tonal_bands = []
    for f, spl in zip(peaks, spls):
        masker = Masker(f, spl, True)
        masking_curve = masker.IntensityAtFreq(mdct_freqs)
        masking_spl = SPL(masking_curve)
        complete_mask.append(masking_spl)
        tonal_bands.append(int(Bark(f)))

    # Add noise maskers based on critical bands
    for z in range(sfBands.nBands):
        if z not in tonal_bands:
            lower = sfBands.lowerLine[z]
            upper = sfBands.upperLine[z] + 1
            intensities = norm * abs(x_fft[lower:upper])**2
            spl = SPL(intensities.sum())
            avg_freq = np.sum(
                intensities * x_fftfreq[lower:upper]) / intensities.sum()
            masker = Masker(avg_freq, spl, isTonal=False)
            masking_curve = masker.IntensityAtFreq(mdct_freqs)
            masking_spl = SPL(masking_curve)

    # Add masking threshold
    complete_mask.append(Thresh(mdct_freqs))
    complete_mask = np.amax(complete_mask, axis=0)

    return complete_mask


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  corresponds to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations. 
				Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    masked_threshold = getMaskedThreshold(data, MDCTdata, MDCTscale,
                                          sampleRate, sfBands)

    # scale down mdct
    mdct_data = MDCTdata / 2**MDCTscale
    sine_power = 1 / 2
    norm = 2 / (sine_power)
    mdct_intensity = mdct_data**2 * norm
    mdct_spl = SPL(mdct_intensity)

    if test_smr:
        n_mdct_lines = len(MDCTdata)
        mdct_freq_spacing = sampleRate / (2 * n_mdct_lines)
        mdct_freqs = mdct_freq_spacing * (np.arange(n_mdct_lines) + 0.5)

        x_fft = np.fft.rfft(HanningWindow(data))
        x_fftfreq = np.fft.rfftfreq(len(data), d=1 / sampleRate)

        plt.figure()
        plt.semilogx(mdct_freqs, masked_threshold, label='masked threshold')
        plt.semilogx(mdct_freqs, mdct_spl, label='MDCT SPL')
        plt.semilogx(x_fftfreq,
                     SPL(abs(x_fft)**2 * 8 / 3 * 4 / len(data)**2),
                     label='FFT')
        plt.ylim([0, 96])
        plt.legend()
        # Add bark lines
        bark_freqs = np.array([
            0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
            2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
            15500
        ])
        for f in bark_freqs:
            plt.axvline(f, linewidth=0.5)
        plt.savefig('./test_smr.png', dpi=200)

    # Then determines the maximum signal-to-mask ratio within
    # each critical band and returns that result in the SMR[] array.
    smr = np.zeros((sfBands.nBands, ))
    for i in range(sfBands.nBands):
        band_start = sfBands.lowerLine[i]
        band_end = sfBands.upperLine[i] + 1
        smr[i] = np.amax(mdct_spl[band_start:band_end] -
                         masked_threshold[band_start:band_end])

    return smr


#-----------------------------------------------------------------------------


def generate_sine_wave(N, As, f0s, Fs=48e3, hann=True):
    # Windowed by hanning
    t = np.arange(0, N)
    x = np.zeros((N, ))
    for i in range(len(As)):
        x += As[i] * np.cos(2 * np.pi * f0s[i] * t / Fs)
    if hann:
        x = HanningWindow(x)
    return x


def estimate_peaks(x_fft_intensity, x_fftfreq):
    # Estimate the peaks and their frequencies
    peaks = []
    nbrs = 1
    for i in range(nbrs, len(x_fft_intensity)):
        if np.sum(x_fft_intensity[i] > x_fft_intensity[i - nbrs:i]) == len(
                x_fft_intensity[i - nbrs:i]) and np.sum(
                    x_fft_intensity[i] > x_fft_intensity[i + 1:i + nbrs + 1]
                ) == len(x_fft_intensity[i + 1:i + nbrs + 1]):
            peaks.append(i)

    frequencies = []
    spls = []
    for f in peaks:
        # Find the intensity and center frequency by using the neighboring two bins
        intensity = np.sum(x_fft_intensity[f - 1:f + 1])
        spl = SPL(intensity)
        peak_en = x_fft_intensity[f - 1:f + 1]
        avg_freq = np.sum(x_fftfreq[f - 1:f + 1] * peak_en) / np.sum(peak_en)
        frequencies.append(avg_freq)
        spls.append(spl)
    return frequencies, spls


#Testing code
if __name__ == "__main__":

    # Block size comparison
    Ns = [512, 1024, 2048]
    Fs = 48000
    As = [0.43, 0.24, 0.15, 0.09, 0.05, 0.04]
    f0s = [440, 550, 660, 880, 4400, 8800]
    true_spls = 10 * np.log10(np.array(As)**2) + 96
    for N in Ns:
        print(f"N={N}")
        window_power = np.mean(np.hanning(N)**2)
        norm = 4 / (N**2 * window_power)
        x = generate_sine_wave(N, As, f0s, Fs)
        x_fft = np.fft.rfft(x)
        x_fftfreq = np.fft.rfftfreq(N, d=1 / Fs)
        x_fft_spl = SPL(norm * abs(x_fft)**2)
        plt.semilogx(x_fftfreq, x_fft_spl, label=f"Block size={N}")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('dB SPL')

        freqs, spls = estimate_peaks(norm * abs(x_fft)**2, x_fftfreq)
        df = pd.DataFrame(list(zip(f0s, freqs, true_spls, spls)),
                          columns=[
                              'Actual frequency (Hz)',
                              'Estimated Frequency (Hz)', 'Actual SPL',
                              'Estimated SPL'
                          ])
        print(df.to_latex())

    plt.legend()
    plt.savefig("./spl_block_sizes.png", dpi=200)
    plt.figure()

    # Threshold in quiet
    N = 1024
    window_power = np.mean(np.hanning(N)**2)
    norm = 4 / (N**2 * window_power)

    x = generate_sine_wave(N, As, f0s, Fs)
    x_fft = np.fft.rfft(x)
    x_fftfreq = np.fft.rfftfreq(N, d=1 / Fs)
    x_fft_intensity = norm * abs(x_fft)**2
    x_fft_spl = SPL(x_fft_intensity)
    plt.semilogx(x_fftfreq, x_fft_spl, label=f"X[k]")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB SPL')
    plt.semilogx(x_fftfreq, Thresh(x_fftfreq), label='Hearing threshold')
    plt.ylim([-10, 96])
    plt.legend()
    plt.savefig("./spl_hearing_threshold.png", dpi=200)

    # Bark testing
    print("Bark scale")
    bark_freqs = np.array([
        0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
        15500
    ])
    barks = Bark(bark_freqs)
    print(
        pd.DataFrame(zip(bark_freqs, barks), columns=['Frequency(Hz)',
                                                      'Bark']).to_latex())

    # Maskers test (1e)
    freqs, spls = estimate_peaks(x_fft_intensity, x_fftfreq)

    complete_mask = []
    for f, spl in zip(freqs, spls):
        masker = Masker(f, spl, True)
        print(f, spl)
        masking_curve = masker.IntensityAtFreq(x_fftfreq)
        masking_spl = SPL(masking_curve)
        plt.semilogx(x_fftfreq, masking_spl, label=f'Masker at {f:0.1f}')
        complete_mask.append(masking_spl)
    plt.legend()
    plt.savefig('./masking_curves.png', dpi=200)

    # Add bark lines
    for f in bark_freqs:
        plt.axvline(f, linewidth=0.5)

    # Add masking threshold
    complete_mask.append(Thresh(x_fftfreq))
    complete_mask = np.amax(complete_mask, axis=0)
    plt.semilogx(x_fftfreq,
                 complete_mask,
                 linestyle='--',
                 linewidth=3,
                 label='Complete mask')
    plt.legend()
    plt.savefig('./masking_curves_bark.png', dpi=200)

    # Test MDCT lines assignment
    mdct_lines = AssignMDCTLinesFromFreqLimits(2560,
                                               16000,
                                               flimit=cbFreqLimits)

    # 1f

    scale_factor_bands = ScaleFactorBands(
        AssignMDCTLinesFromFreqLimits(512, 48e3))
    print(
        "Scale factor bands lower line and upper line:",
        list(zip(scale_factor_bands.lowerLine, scale_factor_bands.upperLine)))
    print(scale_factor_bands.nBands, scale_factor_bands.nLines)

    # 1g
    N = 1024
    x = generate_sine_wave(N, As, f0s, Fs, hann=False)
    scale_factor = 6
    nLines = AssignMDCTLinesFromFreqLimits(N // 2, Fs)
    sfBands = ScaleFactorBands(nLines)
    x_mdct = MDCT(SineWindow(x), N // 2, N // 2) * (2**scale_factor)
    smr_mdct = CalcSMRs(x, x_mdct, scale_factor, Fs, sfBands)
    print(pd.DataFrame(smr_mdct, columns=['SMR']).to_latex())
