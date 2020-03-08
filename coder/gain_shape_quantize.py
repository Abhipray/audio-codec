"""Functions and classes to perform gain-shape quantization of a vector.

The gain is quantized using mu-law companding followed by uniform quantization.
The shape is quantized using a normalized pyramid vector quantization codebook (PVQ).

@author Abhipray Sahoo (abhiprays@gmail.com)
"""

import numpy as np


def pvq_search(x: np.array, k: int):
    """Given a unit vector x of N dimension, returns closest PVQ S(N, k) codebook vector.
    It projects the point from the hypersphere to the hyperpyramid.
    
    Args:
        x (np.array): input vector
        k (int): number of "pulses"
    
    Returns:
        [type]: [description]
    """
    x_l1 = np.sum(np.abs(x))
    x_hat = k * x / x_l1
    return x_hat, np.round(x_hat)


def gain_shape_alloc(R, N, k_fine):
    """Given a total number of bits, calculate the number of bits for the gain-shape quantization"""
    R_shape = R / N + 0.5 * np.log2(N) - k_fine
    R_shape = np.round(R_shape)
    R_gain = R - R_shape
    return R_gain, R_shape


def pvq_codebook_size(L, K, N=None):
    """Computes a table N(L,K) that contains the size of the codebook described by the PVQ S(L,K)
    
    Args:
        L ([int]): Number of dimensions of the vector
        K ([int]): Number of "pulses"
    
    Returns:
        [np.array]: table N(L,K)
    """
    if N is None:
        N = np.ones((L + 1, K + 1), dtype=np.int) * -1
    if L >= 0 and K == 0:
        N[L][K] = 1
        return N
    if K >= 1 and L == 0:
        N[L][K] = 0
        return N
    if N[L - 1][K] == -1:
        pvq_codebook_size(L - 1, K, N)
    if N[L - 1][K - 1] == -1:
        pvq_codebook_size(L - 1, K - 1, N)
    if N[L][K - 1] == -1:
        pvq_codebook_size(L, K - 1, N)
    N[L][K] = N[L - 1][K] + N[L - 1][K - 1] + N[L][K - 1]
    return N


# Encode a pyramid vq codebook vector into an integer index
def encode_pvq_vector(x, K, N):
    b = 0
    k = K
    L = len(x)
    l = L

    for x_i in x:
        if abs(x_i) == 0:
            b += 0
        if abs(x_i) == 1:
            b += N[l - 1][k] + (0.5 * (1 - np.sign(x_i))) * N[l - 1][k - 1]
        if abs(x_i) > 1:
            b += N[l - 1][k] + 2 * np.sum([
                N[l - 1][k - j] for j in range(1, abs(x_i))
            ]) + 0.5 * (1 - np.sign(x_i)) * N[l - 1][k - abs(x_i)]
        k -= abs(x_i)
        l -= 1
        if k == 0:
            break
    return int(b)


# Decode a pyramid vq codebook vector index into the codebook vector
def decode_pvq_vector(b, L, K, N):
    x = np.zeros((L, ), dtype=np.int)
    xb = 0
    k = K
    l = L
    i = 0
    while i < L:
        step_2, step_3, step_4, step_5 = True, True, True, True
        if b == xb:
            # step 1
            x[i] = 0
            step_2, step_3, step_4 = False, False, False
        # step 2
        if step_2:
            if b < xb + N[l - 1][k]:
                x[i] = 0
                step_3 = False
            else:
                xb += N[l - 1][k]
                j = 1
        # step 3
        if step_3:
            while b >= xb + 2 * N[l - 1][k - j]:
                xb += 2 * N[l - 1][k - j]
                if j < k:
                    j += 1
                else:
                    break
            if b < xb + 2 * N[l - 1][k - j]:
                if xb <= b < xb + N[l - 1][k - j]:
                    x[i] = j
                else:
                    x[i] = -j
        # step 4
        if step_4:
            k -= abs(x[i])
            l -= 1
            i += 1
            if k > 0:
                step_5 = False
        # step 5
        if step_5:
            if k > 0:
                x[L - 1] = k - abs(x[i])
            break
    return x


def pvq_compute_k(L: int, R: int):
    """Given a rate R (number of bits), find the largest K (number of pulses) that satisfies log2(N(L, K)) <= R
    
    Args:
        L ([int]): 
        R ([int]): maximum number of bits
    """
    largest_codebook_size = 2**R
    k = 1
    while True:
        N = pvq_codebook_size(L, k)
        if N[L][k] > largest_codebook_size:
            return k - 1
        k += 1
