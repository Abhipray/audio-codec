import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import wavfile
import scipy.signal as signal
from window import SineWindow, HanningWindow, StartWindow, StartStopWindow, StopWindow, TestShortWindow

fs, x = wavfile.read('test_decoded2/castanet_128.wav')
fs, y = wavfile.read('test_signals/castanet.wav')

x = x[:,0] / 2**15
y = y[:,0] / 2**15

x = np.zeros(2048*3)
x[:2048] += StartWindow(StartWindow(np.ones(2048), 2048, 256), 2048, 256)

for n in range(1024, 1024 + 2048 - 128, 128):
    x[n:n+256] += SineWindow(SineWindow(np.ones(256)))

x[2048:2048*2] += StopWindow(StopWindow(np.ones(2048), 2048, 256), 2048, 256)

plt.plot(x)

# f, t, Zxx = signal.stft(x, fs, nperseg=1024)

# plt.figure()
# plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
# plt.title('X STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')

# f, t, Zyy = signal.stft(y, fs, nperseg=1024)

# plt.figure()
# plt.pcolormesh(t, f, 20*np.log10(np.abs(Zyy)))
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')

plt.show()
