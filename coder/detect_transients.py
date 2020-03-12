import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def parTransientDetect(block, thresh=4.5, axis=1):
    """
    Use simple peak-to-avg ratio method to detect transients
    """
    mymax = np.max(np.abs(block), axis=axis)
    argmax = np.argmax(np.abs(block), axis=axis) + 500
    idxs = np.arange(min(max(argmax), np.shape(block)[axis]))
    if len(idxs) != 0:
        avg = np.mean(np.abs(np.take(block, idxs, axis=axis)))
    else:
        avg = np.mean(np.abs(block))

    if np.any(avg == 0):
        return 0

    par = mymax / avg # peak to avg ratio
    if np.any(par > thresh):
        return True
    return False

if __name__ == "__main__":
    fs, x = wavfile.read('../test_signals/castanet.wav')
    x = x[20000:500000,:] / 2**15

    plt.plot(x[:,0])

    block_size = 2048
    ts = np.zeros_like(x)

    for n in range(0, len(x), block_size):
        block = x[n:n+block_size]

        ts[n:n+block_size] = float(parTransientDetect(block, thresh=4.5, axis=0))

    plt.plot(ts)

    plt.show()
