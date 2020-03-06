import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import wavfile
import scipy.signal as signal

# fs, xz = wavfile.read('test_decoded2/castanet_128zz.wav')
fs, x = wavfile.read('test_decoded/castanet_128.wav')
fs, x2 = wavfile.read('test_decoded2/castanet_128.wav')
fs, y = wavfile.read('test_signals/castanet.wav')

x = x[:,0] / 2**15
x2 = x2[:,0] / 2**15
y = y[:,0] / 2**15

def specgram_test(x, y):
    f, t, Zxx = signal.stft(x, fs, nperseg=1024)
    plt.figure()
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
    plt.title('X STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    f, t, Zyy = signal.stft(y, fs, nperseg=1024)
    plt.figure()
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Zyy)))
    plt.title('STFT Magnitude')
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
specgram_test(x, y)
specgram_test(x2, y)
# waveform_test(y)
# waveform_test(x)
# waveform_test(xz)

plt.show()
