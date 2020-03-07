import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def parTransientDetect(block, thresh=6.4, axis=1):
    """
    Use simple peak-to-avg ratio method to detect transients
    """
    avg = np.mean(np.abs(block), axis=axis)
    max = np.max(np.abs(block), axis=axis)

    if any(avg == 0):
        return 0

    par = max / avg # peak to avg ratio
    if any(par > thresh):
        return True
    return False

if __name__ == "__main__":
    fs, x = wavfile.read('test_signals/castanet.wav')
    x = x[25000:225000,:] / 2**15

    plt.plot(x[:,0])

    block_size = 2048
    ts = np.zeros_like(x)
    thresh = 15

    for n in range(0, len(x), block_size):
        block = x[n:n+block_size]
        ts[n:n+block_size] = float(parTransientDetect(block, axis=0))

    plt.plot(ts)

    plt.show()
