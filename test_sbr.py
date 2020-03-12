import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import wavfile
import scipy.signal as signal

fs, x2 = wavfile.read('sbr_test_sigma10_20linespband/harpsichord_96.wav')
fs, y = wavfile.read('test_signals/harpsichord.wav')

def specgram_test(x_list):
    for x in x_list:

        x = x[:,0] / 2**15
        f, t, Zxx = signal.stft(x, fs, nperseg=1024)
        plt.figure()
        plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
        plt.title('X STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

def waveform_test(x):
    plt.plot(x[:150000])
    plt.xlim(100000, 110000)
    # plt.plot(x)
    plt.axvline(1024*101, color='r')
    plt.axvline(1024*102, color='r')
    plt.axvline(1024*103, color='r')

# window_test()
specgram_test([x2, y])
# waveform_test(y)
# waveform_test(x)
# waveform_test(xz)

plt.show()
