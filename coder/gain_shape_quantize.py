"""Functions and classes to perform gain-shape quantization of a vector.
@author Abhipray Sahoo (abhiprays@gmail.com)

The gain is quantized using mu-law companding followed by uniform quantization.
The shape is quantized using a normalized pyramid vector quantization codebook (PVQ).

The PVQ codebook is defined as set of vectors y: {y: sum(y_i) = K}. It can be thought of
as placing K unit-pulses at different dimensions of the vector y to achieve the sum K.

Symbols and definitions:
L = number of dimensions of vector
K = number of pulses
N(L,K) = number of vectors in PVQ codebook for given vector dimension and K
S(L,K) = set of PVQ codebook vector for given vector dimension L and K
R = rate (or number of bits) per dimensions

"""

import numpy as np
import logging
from quantize import *
from gain_shape_encoding import GainShapeEncoding
from bitpack import PackedBits

log = logging.getLogger(__name__)

SPLIT_BITS = 32


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
    x_hat = np.abs(k * x / x_l1)
    # Round down and then do a greedy optimization to fit the remaining pulses
    pre_search = np.floor(x_hat)
    # Count remaining pulses
    remaining_pulses = k - np.sum(abs(pre_search))
    while remaining_pulses > 0:
        # log.debug(f'Remaining pulses: {remaining_pulses}')
        # distribute pulses to the components that got rounded down
        i = np.argmax(abs(x_hat) - pre_search)
        pre_search[i] += 1
        remaining_pulses -= 1
    pre_search *= np.sign(x)
    return x_hat * np.sign(x), pre_search.astype(np.int)


def gain_shape_alloc(R, L, k_fine):
    """Given a total number of bits, calculate the number of bits for the gain-shape quantization"""
    R_gain = R / L + 0.5 * np.log2(L) - k_fine
    R_gain = np.floor(R_gain)
    R_shape = max(R - R_gain, 0)
    return int(R_gain), int(R_shape)


cache = {}


# @np_cache(maxsize=int(1e6))
def pvq_codebook_size(L, K, N=None):
    """Computes a table N(L,K) that contains the size of the codebook described by the PVQ S(L,K)
    
    Args:
        L ([int]): Number of dimensions of the vector
        K ([int]): Number of "pulses"
    
    Returns:
        [np.array]: table N(L,K)
    """
    global cache
    # print(L, K)
    if N is None:
        N = np.ones((L + 1, K + 1), dtype=np.int) * -1
    if (L, K) in cache:
        # print('cache hit', (L, K), cache.keys())
        N[:L + 1, :K + 1] = cache[(L, K)][:L + 1, :K + 1]
        return N
    if L >= 0 and K == 0:
        N[:L + 1, K] = 1
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
    cache[(L, K)] = N[:L + 1, :K + 1]
    return N


# Encode a pyramid vq codebook vector into an integer index
def encode_pvq_vector(x, K, N):
    b = 0
    k = K
    L = len(x)
    l = L

    for x_i in x:
        if x_i == 0:
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


class decoder:
    def __init__(self, b, L, K, N):
        self.x = np.zeros((L, ), dtype=np.int)
        self.i = 0
        self.xb = 0
        self.k = K
        self.l = L
        self.b = b
        self.L = L
        self.N = N

    def step_5(self):
        # global k, L, x
        if self.k > 0:
            self.x[self.L - 1] = self.k - abs(self.x[self.i])

    def step_1(self):
        if self.b == self.xb:
            self.x[self.i] = 0
            self.step_5()
        else:
            self.step_2()
        return self.x

    def step_4(self):
        self.k = self.k - abs(self.x[self.i])
        self.l -= 1
        self.i += 1
        if self.k > 0:
            self.step_1()
        else:
            self.step_5()

    def step_2(self):
        if self.b < (self.xb + self.N[self.l - 1][self.k]):
            self.x[self.i] = 0
            self.step_4()
        else:
            self.xb += self.N[self.l - 1][self.k]
            self.j = 1
            self.step_3()

    def step_3(self):
        self.j = min(self.j, self.k)
        if self.b < (self.xb +
                     2 * self.N[self.l - 1][max(self.k - self.j, 0)]):
            if self.b >= self.xb + self.N[self.l - 1][self.k - self.j]:
                self.x[self.i] = -self.j
            elif self.xb <= self.b < self.xb + self.N[self.l - 1][self.k -
                                                                  self.j]:
                self.x[self.i] = self.j
            self.step_4()
        else:
            self.xb += 2 * self.N[self.l - 1][self.k - self.j]
            if self.j < self.k:
                self.j += 1
                self.step_3()
            else:
                self.step_1()
            # self.j += 1


# Decode a pyramid vq codebook vector index into the codebook vector
def decode_pvq_vector(b, L, K, N):
    return decoder(b, L, K, N).step_1()
    x = np.zeros((L, ), dtype=np.int)
    xb = 0
    k = K
    l = L
    i = 0
    while True:
        step_2, step_3, step_4, step_5 = True, True, True, True
        if b == xb:
            # step 1
            # log.debug('step1')
            x[i] = 0
            step_2, step_3, step_4 = False, False, False
        # step 2
        if step_2:
            # log.debug('step2')
            if b < (xb + N[l - 1][k]):
                x[i] = 0
                step_3 = False
            else:
                xb += N[l - 1][k]
                j = 1
        # step 3
        did_break = False
        if step_3:
            # log.debug('step3')
            while b >= (xb + 2 * N[l - 1][k - j]):
                xb += 2 * N[l - 1][k - j]
                if j < k:
                    j += 1
                else:
                    break

            if b < (xb + 2 * N[l - 1][k - j]):
                if b >= (xb + N[l - 1][k - j]):
                    x[i] = -j
                elif xb <= b < (xb + N[l - 1][k - j]):
                    x[i] = j
                else:
                    log.debug("Oh")
        # step 4
        if step_4:
            # log.debug('step4')
            k -= abs(x[i])
            l -= 1
            i += 1
            if k > 0:
                step_5 = False
                log.debug('skipping step5', k, x[i])
        # step 5
        if step_5:
            # log.debug('step5')
            if k > 0:
                x[L - 1] = k - abs(x[i])
            break
    return x


def quantize_pvq(x, num_bits):
    """Vector quantize a unit-vector x with num_bits"""
    x_norm = np.linalg.norm(x)
    assert np.allclose(x_norm, 1.0) or np.allclose(
        x_norm, 0.0), f"x is not a unit-vector, norm={x_norm}"
    L = len(x)
    k, actual_bits_used, N = pvq_compute_k_for_R(L, num_bits)

    # PVQ the shape
    _, codebook_vector = pvq_search(x, k)
    codebook_idx = encode_pvq_vector(codebook_vector, k, N)
    return codebook_idx, actual_bits_used


def dequantize_pvq(cb_idx, L, k, N=None):
    """Find the codebook vector corresponding to cb_idx"""
    if N is None:
        N = pvq_codebook_size(L, k)
    x = decode_pvq_vector(cb_idx, L, k, N)
    x = x.astype(np.float)
    x_l2 = np.linalg.norm(x)
    if x_l2 != 0:
        x = x / np.linalg.norm(x)
    return x


def pvq_compute_k_for_R(L: int, max_bits: int):
    """Given a number of bits, find the largest K (number of pulses) that satisfies log2(N(L, K)) <= max_bits
    
    Args:
        L ([int]): 
        R ([int]): maximum number of bits
    """
    largest_codebook_size = 2**max_bits
    k = 1
    while True:
        N = pvq_codebook_size(L, k)
        if N[L][k] > largest_codebook_size:
            return k - 1, np.ceil(np.log2(N[L][k - 1] +
                                          np.finfo(float).eps)).astype(
                                              np.int), N[:, :k]

        k += 1


def mu_law_fn(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)


def inv_mu_law_fn(y, mu=255):
    return np.sign(y) / mu * ((1 + mu)**np.abs(y) - 1)


def bit_allocation_ms(a_mid_side, theta, length, k_fine=0):
    if theta == 0:
        a_mid = 0
    else:
        a_mid = np.floor(
            (a_mid_side -
             (length - 1) * np.log2(np.tan(abs(theta)) + np.finfo(float).eps))
            / 2).astype(np.int)
    a_mid = min(max(a_mid, 0), a_mid_side)
    a_side = max(a_mid_side - a_mid, 0)
    return a_mid, a_side


def split_band_encode(x, bit_alloc, k_fine=0):
    if bit_alloc > SPLIT_BITS:
        log.debug(f'Splitting because bit alloc is {bit_alloc}')
        # MS coding with split bands
        # Split the band in half
        band_size = len(x)
        mid = band_size // 2

        half_band = int(np.ceil(band_size / 2))
        if half_band > band_size // 2:
            # Band has odd number of elements, pad the left half
            left = np.concatenate([x[:mid], [0]])
        else:
            left = x[:mid]
        right = x[mid:]
        mid = half_band

        # Calculate M and S
        M = (left + right) / 2
        S = (left - right) / 2
        M_l2 = np.linalg.norm(M)
        S_l2 = np.linalg.norm(S)
        if M_l2 != 0:
            m = M / M_l2
        else:
            m = M
        if S_l2 != 0:
            s = S / S_l2
        else:
            s = S
        # theta captures distribution of energy between bands
        if M_l2 == 0:
            theta = 0
        else:
            theta = np.arctan(S_l2 / M_l2)

        # scalar quantize theta
        log.debug(f'Encoded theta: {theta}')
        theta_normalized = theta / (np.pi / 2)
        # Find bit allocation for theta
        a_theta, a_mid_side = gain_shape_alloc(bit_alloc, mid, k_fine)
        theta_idx = QuantizeUniform(theta_normalized, a_theta)
        theta_quantized = DequantizeUniform(theta_idx, a_theta) * (np.pi / 2)

        # Find bit allocations for m, s
        a_mid, a_side = bit_allocation_ms(a_mid_side, theta_quantized, mid,
                                          k_fine)
        log.debug(
            f'Encoded bitalloc theta: {a_theta}, mid: {a_mid}, side: {a_side} L:{mid} theta_quantized:{theta_quantized}'
        )

        # PVQ m and s
        indices = [theta_idx]
        bits = [a_theta]

        if a_mid > SPLIT_BITS:
            indices_new, bits_new = split_band_encode(m, a_mid, k_fine)
            assert sum(bits_new
                       ) <= a_mid, f"busted {sum(bits_new)} {a_mid} {bits_new}"
            indices += indices_new
            bits += bits_new
        elif a_mid > 0:
            # Base case
            mid_idx, mid_actual_bits = quantize_pvq(m, a_mid)
            # print(f'Adding {mid_actual_bits} bits for mid')
            indices += [mid_idx]
            bits += [mid_actual_bits]

        if a_side > SPLIT_BITS:
            indices_new, bits_new = split_band_encode(s, a_side, k_fine)
            assert sum(bits_new) <= a_side
            indices += indices_new
            bits += bits_new
        elif a_side > 0:
            side_idx, side_actual_bits = quantize_pvq(s, a_side)
            # print(f'Adding {side_actual_bits} bits for side')
            indices += [side_idx]
            bits += [side_actual_bits]

        # log.debug(
        #     f'Encoded actual bits used for mid: {mid_actual_bits} and side: {side_actual_bits}'
        # )
        return indices, bits
    else:
        # Vector quantize directly but set max number of bits to 32
        bit_alloc = min(32, bit_alloc)
        idx, actual_bits = quantize_pvq(x, bit_alloc)
        log.debug(f'No split Encode actual bits used: {actual_bits}')
        return [idx], [actual_bits]


def split_band_decode(pb, num_bits, band_size, k_fine=0):
    if num_bits > SPLIT_BITS:
        half_band = int(np.ceil(band_size / 2))
        a_theta, a_mid_side = gain_shape_alloc(num_bits, half_band, k_fine)
        log.debug(f'bitalloc theta: {a_theta}')
        theta_idx = pb.ReadBits(a_theta)
        theta_hat = DequantizeUniform(theta_idx, a_theta) * (np.pi / 2)
        log.debug(f'decoded theta: {theta_hat}')
        # Decode pvq indices
        # Find bit allocations for m and s
        a_mid, a_side = bit_allocation_ms(a_mid_side, theta_hat, half_band,
                                          k_fine)
        log.debug(f'bitalloc theta: {a_theta}, {a_mid}, {a_side}')

        total_used_bits = a_theta

        if a_mid > SPLIT_BITS:
            mid_hat, used_bits = split_band_decode(pb, a_mid, half_band,
                                                   k_fine)
            total_used_bits += used_bits
        elif a_mid > 0:
            # decode mid here
            k, mid_actual_bits, N = pvq_compute_k_for_R(half_band, a_mid)
            log.debug(f'mid_actual_bits: {mid_actual_bits}')
            mid_idx = pb.ReadBits(mid_actual_bits)
            mid_hat = dequantize_pvq(mid_idx, half_band, k, N)
            total_used_bits += mid_actual_bits
        else:
            mid_hat = np.zeros((half_band, ))
        if a_side > SPLIT_BITS:
            side_hat, used_bits = split_band_decode(pb, a_side, half_band,
                                                    k_fine)
            total_used_bits += used_bits
        elif a_side > 0:
            # decode right here
            k, side_actual_bits, N = pvq_compute_k_for_R(half_band, a_side)
            log.debug(f'side_actual_bits: {side_actual_bits}')
            side_idx = pb.ReadBits(side_actual_bits)
            side_hat = dequantize_pvq(side_idx, half_band, k, N)
            total_used_bits += side_actual_bits
        else:
            side_hat = np.zeros((half_band, ))

        left = mid_hat * np.cos(theta_hat) + side_hat * np.sin(theta_hat)
        right = mid_hat * np.cos(theta_hat) - side_hat * np.sin(theta_hat)

        left /= np.sqrt(2)
        right /= np.sqrt(2)

        if half_band > band_size // 2:
            # Band has odd number of elements, drop the last element in left
            left = left[:len(left) - 1]

        x = np.concatenate([left, right])
        x_l2 = np.linalg.norm(x)
        if x_l2 != 0:
            x /= x_l2
        return x, total_used_bits
    else:
        bit_alloc = min(32, num_bits)
        k, actual_bits_used, N = pvq_compute_k_for_R(band_size, num_bits)
        idx = pb.ReadBits(actual_bits_used)

        x = dequantize_pvq(idx, band_size, k, N)
        x_l2 = np.linalg.norm(x)
        if x_l2 != 0:
            x /= x_l2
        return x, actual_bits_used


def quantize_gain_shape(x, bit_alloc, k_fine=0):
    # Allocate bits for gain and shape
    L = len(x)
    bits_gain, bits_shape = gain_shape_alloc(bit_alloc, L, k_fine)

    log.debug(f'Original bits_gain: {bits_gain} bits_shape {bits_shape}')

    # Separate gain and shape
    gain = np.linalg.norm(x)

    if gain != 0:
        if bits_shape != 0:
            # Encode the shape of the band using split_encode
            shape = x / gain
            log.debug(shape.tolist())
            indices, bits = split_band_encode(shape, bits_shape, k_fine)
            total_used_bits = sum(bits)
            assert total_used_bits <= bits_shape, "Used more bits than allocated in quantize"
            # Mu-law scalar quantize gain
            bits_gain += bits_shape - total_used_bits
            log.debug(
                f'Actual bits_gain: {bits_gain} bits_shape {total_used_bits}')

        else:
            indices = []
            bits = []
        gain = mu_law_fn(gain / L)
        log.debug(f"original gain: {gain}")
        if bits_gain < 0:
            bits_gain = 0
        gain_idx = QuantizeUniform(gain, bits_gain)

        indices = indices + [gain_idx]
        bits = bits + [bits_gain]
        return indices, bits
    else:
        return [0], [0]


def dequantize_gain_shape(pb, bit_alloc, L, k_fine=0):

    bits_gain, bits_shape = gain_shape_alloc(bit_alloc, L, k_fine)
    log.debug(f'bits_gain: {bits_gain} bits_shape {bits_shape}')

    if bits_shape != 0:
        # Find k that satisfies R_shape
        shape, bits_used = split_band_decode(pb, bits_shape, L, k_fine)
        assert bits_used <= bits_shape, "Used more bits than allocated in dequantize"
        # Reconstruct the gain
        # Mu-law scalar dequantize gain
        bits_gain += (bits_shape - bits_used)
    else:
        shape = np.ones((L, ))
        bits_used = bits_shape

    log.debug(f'Gain bits: {bits_gain}, {bits_used}')
    gain_idx = pb.ReadBits(bits_gain)

    gain_unquantized = DequantizeUniform(gain_idx, bits_gain)
    gain = inv_mu_law_fn(gain_unquantized) * L
    log.debug(f"decoded gain:{gain}")

    return gain * shape


if __name__ == '__main__':
    L = 163
    bit_alloc = 489
    k_fine = 0

    x = np.random.rand(L) * 2 - 1
    # test quantize_gain_shape
    indices, bits = quantize_gain_shape(x, bit_alloc, k_fine)
    print(indices, bits, sum(bits))

    # pack bits
    pb = PackedBits()
    pb.Size(np.ceil(sum(bits) / 8).astype(np.int))
    for idx, nBits in zip(indices, bits):
        pb.WriteBits(idx, nBits)  # encode indices

    pb.ResetPointers()
    x_hat = dequantize_gain_shape(pb, bit_alloc, L, k_fine)
    error = 10 * np.log10(np.mean((x - x_hat)**2))

    expected_snr = 6.02 * bit_alloc / L
    print(list(zip(x, x_hat)))
    scale_bits = 3
    mantissa_bits = int((bit_alloc - scale_bits) // L)
    scale = ScaleFactor(np.max(x), scale_bits, mantissa_bits)
    mantissa = vMantissa(x, scale, scale_bits, mantissa_bits)
    fp_quant = vDequantize(scale, mantissa, scale_bits, mantissa_bits)
    fp_error = 10 * np.log10(np.mean((x - fp_quant)**2))

    print(
        f'error: {error}dB expected: -{expected_snr} fp_error: {fp_error}dB {np.linalg.norm(x)} {np.linalg.norm(x_hat)}'
    )
