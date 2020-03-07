"""
window.py -- Defines functions to window an array of data samples
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import matplotlib.pyplot as plt

from mdct import MDCT


### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N)
    window = np.sin(np.pi * (n + 0.5) / N)
    return window * dataSampleArray
    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N)
    window = 0.5 * (1 - np.cos(2 * np.pi * (n + 0.5) / N))
    return window * dataSampleArray
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray, alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the 
	Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    n = np.arange(N)
    numerator = np.i0(np.pi * alpha * np.sqrt(1 - ((2 * n + 1) / N - 1)**2))
    denominator = np.i0(np.pi * alpha)
    window = numerator / denominator
    return dataSampleArray * window
    ### YOUR CODE ENDS HERE ###

def StartWindow(dataSampleArray, N_long, N_short):
    """
    Returns a copy of dataSampleArray Sine-Windowed
    with the correct window to start an Edler Block
    Switch
    """
    pad = N_long // 4 - N_short // 4
    window = np.concatenate((SineWindow(np.ones(N_long))[:N_long//2], np.ones(pad),
        SineWindow(np.ones(N_short))[N_short//2:], np.zeros(pad)))

    return window * dataSampleArray

def StopWindow(dataSampleArray, N_long, N_short):
    """
    Returns a copy of dataSampleArray Sine-Windowed
    with the correct window to stop an Edler Block
    Switch
    """
    window = np.flip(StartWindow(np.ones(N_long), N_long, N_short))
    return window * dataSampleArray

def StartStopWindow(dataSampleArray, N_long, N_short):
    """
    Returns a copy of dataSampleArray Sine-Windowed
    with the correct window to transition between two
    "short window" blocks
    """
    pad = N_long // 4 - N_short // 4
    window = np.concatenate((np.zeros(pad), SineWindow(np.ones(N_short))[:N_short//2],
        np.ones(2*pad), SineWindow(np.ones(N_short))[N_short//2:], np.zeros(pad)))

    return window * dataSampleArray

def TestShortWindow(dataSampleArray, N_long, N_short):
    """
    Returns a copy of dataSampleArray Sine-Windowed
    with a window instead of the short windows to be used
    later
    """
    zeroN = N_long - N_short
    window = np.concatenate((SineWindow(np.ones(N_short))[:N_short//2], np.zeros(zeroN), SineWindow(np.ones(N_short))[N_short//2:]))
    return window * dataSampleArray

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    N = 1024
    n = np.arange(N)
    x = np.cos(2 * np.pi * 3000 * n / 44100)
    sine_window = SineWindow(np.ones((N, )))
    x_sine = SineWindow(x)
    x_sine_fft = np.fft.rfft(x_sine)
    x_sine_fft_spl = 96 + 10 * np.log10(
        4 / (N**2 * np.mean(sine_window**2)) * abs(x_sine_fft)**2)
    print(np.amax(x_sine_fft_spl))
    x_sine_mdct = MDCT(x_sine, 512, 512)
    x_sine_mdct_spl = 96 + 10 * np.log10(
        2 / (np.mean(sine_window**2)) * abs(x_sine_mdct)**2)

    x_hann = HanningWindow(x)
    x_hann_fft = np.fft.rfft(x_hann)
    hann = HanningWindow(np.ones((N, )))
    x_hann_spl = 96 + 10 * np.log10(
        4 / (N**2 * np.mean(hann**2)) * abs(x_hann_fft)**2)

    kbd = KBDWindow(np.ones((N, )))
    x_kbd = KBDWindow(x)
    x_kbd_mdct = MDCT(x_kbd, N // 2, N // 2)
    x_kbd_spl = 96 + 10 * np.log10(2 / (np.mean(kbd**2)) * abs(x_kbd_mdct)**2)

    plt.plot(x_sine, label='Sine window')
    plt.plot(x_hann, label='Hanning window')
    plt.plot(x_kbd, label='Kaiser Bessel Derived window')
    plt.legend()
    plt.ylabel("Sine wave")
    plt.xlabel("Sample index")
    plt.savefig('./windowed_cosine.png', dpi=200)

    plt.figure()
    fft_freqs = np.fft.rfftfreq(N, 1 / 44100)
    plt.plot(fft_freqs, x_sine_fft_spl, label='Sine window + FFT')
    plt.plot(fft_freqs[:len(x_sine_mdct_spl)],
             x_sine_mdct_spl,
             label='Sine window + MDCT')
    plt.plot(fft_freqs, x_hann_spl, label='Hanning window + FFT')
    plt.plot(fft_freqs[:len(x_kbd_spl)], x_kbd_spl, label='KBD window + MDCT')
    plt.axhline(96)
    plt.axvline(3000, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlabel('Frequency (Hz) or k for MDCT')
    plt.ylabel('SPL (dB)')
    plt.savefig('./SPL_freq.png', dpi=200)
    ### YOUR TESTING CODE ENDS HERE ###
