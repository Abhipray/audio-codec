"""
Class to encode/decode the gain shapes returned by vector quantization methods
"""

from bitpack import *
from audiofile import *

class GainShapeEncoding():
    def __init__(self, gain_idx, gain_bits, mid_idx, mid_bits,
        side_idx, side_bits, theta_idx, theta_bits, codingParams):

        self.idxs = [gain_idx, mid_idx, side_idx, theta_idx]
        self.bits = [gain_bits, mid_bits, side_bits, theta_bits]
        self.baBits = codingParams.nMantSizeBits # num of bits used to store bit allocation info

    def get_bits_needed(self):
        return sum(self.bits)

    def encode(self, pb):
        pb.WriteBits(self.get_bits_needed(), self.baBits) # encode bit allocation sum
        for idx, nBits in zip(self.idxs, self.bits):
            pb.WriteBits(idx, nBits) # encode indices

    def decode(self, pb):
        sum_bits = pb.ReadBits(self.baBits)
        # @TODO: how to calculate these from the sum??
        gain_bits = 0
        mid_bits = 0
        side_bits = 0
        theta_bits = 0

        self.bits = [gain_bits, mid_bits, side_bits, theta_bits]
        for idx, nBits in zip(self.idxs, self.bits):
            idx = pb.ReadBits(nBits) # decode indices


if __name__ == "__main__":
    cp = CodingParams()
    cp.nMantSizeBits = 16

    gse = GainShapeEncoding(123, 1, 456, 2, 789, 3, 10, 4, cp)
    print(gse.get_bits_needed())

    pb = PackedBits()
    pb.Size(gse.get_bits_needed() // 8 + 1)
    gse.encode(pb)
