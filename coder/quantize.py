"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import pandas as pd
from collections import defaultdict


### Problem 1.a.i ###
def QuantizeUniform(aNum, nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    sign = np.sign(aNum)
    if sign >= 0:
        s = 0
    else:
        s = 1
    if abs(aNum) >= 1:
        code = 2**(nBits - 1) - 1
    else:
        code = int(((2**nBits - 1) * abs(aNum) + 1) // 2)
    aQuantizedNum = (s << (nBits - 1)) + code
    ### YOUR CODE ENDS HERE ###

    return int(aQuantizedNum)


### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum, nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    ### YOUR CODE STARTS HERE ###
    s = aQuantizedNum & (1 << (nBits - 1))
    code = aQuantizedNum & (2**(nBits - 1) - 1)
    if s == 0:
        sign = 1
    else:
        sign = -1
    aNum = sign * 2 * code / (2**nBits - 1)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """

    #Notes:
    #Make sure to vectorize properly your function as specified in the homework instructions

    ### YOUR CODE STARTS HERE ###
    aNumVec = np.array(aNumVec)
    s = np.zeros_like(aNumVec)
    s[aNumVec < 0] = 1 << (nBits - 1)
    aQuantizedNumVec = (((2**nBits - 1) * abs(aNumVec) + 1) // 2).astype(int)
    aQuantizedNumVec[abs(aNumVec) >= 1] = 2**(nBits - 1) - 1  # clip
    aQuantizedNumVec = s + abs(aQuantizedNumVec)
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec.astype(np.int)


### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """

    ### YOUR CODE STARTS HERE ###
    s = np.bitwise_and(aQuantizedNumVec, (1 << (nBits - 1)))
    code = np.bitwise_and(aQuantizedNumVec, (2**(nBits - 1) - 1))
    sign = np.ones_like(aQuantizedNumVec)
    sign[s != 0] = -1
    aNumVec = sign * 2 * (code) / (2**nBits - 1)
    ### YOUR CODE ENDS HERE ###

    return aNumVec


### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    s_code = QuantizeUniform(aNum, R)
    code = s_code & (2**(R - 1) - 1)
    # First bit is sign bit
    mask = 1 << (R - 2)
    zeros = 0
    while mask:
        if mask & code == 0:
            zeros += 1
            mask >>= 1
        else:
            break
    if zeros < 2**nScaleBits - 1:
        scale = zeros
    else:
        scale = 2**nScaleBits - 1
    ### YOUR CODE ENDS HERE ###

    return int(scale)


### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    s_code = QuantizeUniform(aNum, R)
    s = s_code & (1 << (R - 1))
    if s > 0:
        s = 1 << (nMantBits - 1)
    code = s_code & (2**(R - 1) - 1)
    if scale == (2**nScaleBits - 1):
        mantissa = s + (code & (2**(nMantBits - 1) - 1))
    else:
        mantissa = s + ((code >> (R - scale - nMantBits - 1)) &
                        ((2**(nMantBits - 1) - 1)))
    ### YOUR CODE ENDS HERE ###

    return int(mantissa)


### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    aNum = 0
    s = mantissa & (1 << (nMantBits - 1))
    if s > 0:
        s = 1 << (R - 1)
    code = mantissa & (2**(nMantBits - 1) - 1)
    aNum += s
    aNum += code << max(R - scale - nMantBits - 1, 0)
    if scale != (2**nScaleBits - 1):
        aNum += 1 << (R - scale - 2)
    shift = R - scale - nMantBits - 2
    if shift > 0:
        aNum += 1 << shift
    aNum = DequantizeUniform(aNum, R)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    s_code = QuantizeUniform(aNum, R)
    s = s_code & (1 << (R - 1))
    if s > 0:
        s = 1 << (nMantBits - 1)
    code = s_code & (2**(R - 1) - 1)
    if scale == (2**nScaleBits - 1):
        mantissa = s + (code & (2**(nMantBits - 1) - 1))
    else:
        mantissa = s + ((code >>
                         (R - scale - nMantBits)) & ((2**(nMantBits - 1) - 1)))
    ### YOUR CODE ENDS HERE ###

    return int(mantissa)


### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    aNum = 0.0  # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    aNum = 0
    s = mantissa & (1 << (nMantBits - 1))
    if s > 0:
        s = 1 << (R - 1)
    code = mantissa & (2**(nMantBits - 1) - 1)
    aNum += s
    aNum += code << max(R - scale - nMantBits, 0)

    if scale < (2**nScaleBits - 1):
        if code > 0:
            shift = R - scale - nMantBits - 1
            aNum += 1 << shift
    aNum = DequantizeUniform(aNum, R)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    mantissaVec = np.zeros_like(
        aNumVec, dtype=int)  # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    s_code = vQuantizeUniform(aNumVec, R)
    s = np.bitwise_and(s_code, (1 << (R - 1)))
    s[s > 0] = 1 << (nMantBits - 1)
    code = np.bitwise_and(s_code, (2**(R - 1) - 1))
    if scale == (2**nScaleBits - 1):
        mantissaVec = s + (code & (2**(nMantBits - 1) - 1))
    else:
        mantissaVec = s + ((code >> (R - scale - nMantBits)) &
                           ((2**(nMantBits - 1) - 1)))
    ### YOUR CODE ENDS HERE ###

    return mantissaVec


### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    R = 2**nScaleBits - 1 + nMantBits
    aNumVec = np.zeros_like(mantissaVec, dtype=int)
    s = np.bitwise_and(mantissaVec, (1 << (nMantBits - 1)))
    s[s > 0] = 1 << (R - 1)
    code = np.bitwise_and(mantissaVec, (2**(nMantBits - 1) - 1))
    aNumVec += s
    aNumVec += np.left_shift(code, max(R - scale - nMantBits, 0))
    if scale < (2**nScaleBits - 1):
        shift = R - scale - nMantBits - 1
        aNumVec[code > 0] += (1 << shift)
    aNumVec = vDequantizeUniform(aNumVec, R)

    ### YOUR CODE ENDS HERE ###

    return aNumVec


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    inputs = np.array(
        [-0.99, -0.39, -.08, -0.001, 0, 0.01, 0.29, 0.68, 0.99, 1.0])
    bitdepths = [8, 12]
    bitdepths_names = ['8 bit midtread', '12 bit midtread']
    print('Uniform Quantization')
    df_dict = defaultdict(list)
    for i in inputs:
        df_dict['Input'].append(i)
    for k, b in enumerate(bitdepths):
        for i in inputs:
            quantized = QuantizeUniform(i, b)
            dequantized = DequantizeUniform(quantized, b)
            df_dict[bitdepths_names[k]].append(dequantized)

    # Test vector quantization
    for k, b in enumerate(bitdepths):
        quantized = vQuantizeUniform(inputs, b)
        dequantized = vDequantizeUniform(quantized, b)
        print(f"Comparing vectorized to normal for bitdepth {b}: ",
              np.allclose(dequantized, df_dict[bitdepths_names[k]]))

    for i in inputs:
        # FP
        scale = ScaleFactor(i)
        mantissa = MantissaFP(i, scale)
        value = DequantizeFP(scale, mantissa)
        df_dict['3s5m FP'].append(value)

        # BFP
        mantissa_bfp = Mantissa(i, scale)
        value_bfp = Dequantize(scale, mantissa_bfp)
        df_dict['3s5m BFP N=1'].append(value_bfp)

        # Test the vector implementation of BFP
        mantissa_vbfp = vMantissa(np.array([i]), scale)
        value_vbfp = vDequantize(scale, mantissa_vbfp)
        assert abs(value_vbfp[0] - value_bfp) < 1e-5

    df = pd.DataFrame(df_dict)
    print(df)
    # print(df.to_latex())

    ### YOUR TESTING CODE ENDS HERE ###
