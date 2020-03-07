"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import time


### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = (a + b)
    mdct = []
    n0 = (b + 1) / 2.0
    if isInverse:
        for n in range(N):
            coeff = np.sum(data * np.cos(2 * np.pi / N * (n + n0) *
                                         (np.arange(N // 2) + 0.5)))
            coeff *= 2
            mdct.append(coeff)
    else:
        for k in range(N // 2):
            coeff = np.sum(data * np.cos(2 * np.pi / N * (np.arange(N) + n0) *
                                         (k + 0.5)))
            coeff *= 2 / N
            mdct.append(coeff)
    mdct = np.array(mdct)
    return mdct
    ### YOUR CODE ENDS HERE ###


### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    n0 = (b + 1) / 2
    mdct = []
    if isInverse:
        pretwiddle = data * np.exp(1j * 2 * np.pi * np.arange(N // 2) * n0 / N)
        ifft = np.fft.ifft(pretwiddle, N)
        posttwiddle = (ifft * np.exp(1j * 2 * np.pi * (np.arange(N) + n0) /
                                     (2 * N))).real
        posttwiddle *= 2 * N
        return posttwiddle
    else:
        pretwiddle = data * np.exp(-1j * 2 * np.pi * np.arange(N) / (2 * N))
        fft = np.fft.fft(pretwiddle, N)[:N // 2]
        posttwiddle = (fft * np.exp(-1j * 2 * np.pi * n0 *
                                    (np.arange(N // 2) + 0.5) / N)).real
        posttwiddle *= 2 / N
        return posttwiddle
    ### YOUR CODE ENDS HERE ###


def IMDCT(data, a, b):

    ### YOUR CODE STARTS HERE ###
    return MDCT(data, a, b, isInverse=True)
    ### YOUR CODE ENDS HERE ###


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    x = np.array([0, 1, 2, 3, 4, 4, 4, 4, 3, 1, -1, -3])
    frame_size = 8
    noverlap = 4
    x = np.concatenate([np.zeros(noverlap), x, np.zeros(noverlap)])
    x_hat = np.zeros_like(x)
    n_samples = len(x)
    for i in range(0, n_samples - noverlap, noverlap):
        data = x[i:i + frame_size]
        mdct = MDCTslow(data, 4, 4, isInverse=False)
        imdct = MDCTslow(mdct, 4, 4, isInverse=True) / 2
        print(i, mdct, imdct)
        x_hat[i:i + frame_size] += imdct
    assert np.allclose(x, x_hat)

    # Test MDCT implementation
    a = b = len(x) // 2
    mdct_a = MDCTslow(x, a, b, isInverse=False)
    mdct_b = MDCT(x, a, b, isInverse=False)
    assert np.allclose(mdct_a, mdct_b)
    imdct_a = MDCTslow(mdct_a, a, b, isInverse=True)
    imdct_b = MDCT(mdct_b, a, b, isInverse=True)
    assert np.allclose(imdct_a, imdct_b)

    # Time the functions
    n_trials = 100
    funcs = {'MDCTSlow': MDCTslow, 'MDCTFast': MDCT}
    x = np.random.randn(2048)
    a = b = 1024
    for f in funcs:
        start = time.time_ns()
        for i in range(n_trials):
            mdct_a = funcs[f](x, a, b, isInverse=False)
            imdct_a = funcs[f](mdct_a, a, b, isInverse=True)
        end = time.time_ns()
        print(
            f"{f} took on average {(end-start)*1e-6/n_trials} milliseconds for {n_trials} trials"
        )

    ### YOUR TESTING CODE ENDS HERE ###
