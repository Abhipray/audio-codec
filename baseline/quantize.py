"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    if nBits <= 0: return 0

    # constants
    signBit = (1 << (nBits-1))
    multFac = (signBit << 1) - 1

    # get sign and convert num to abs val
    val = aNum
    if val < 0:
        sign = 1
        val *= -1
    else: sign = 0

    # create quantization code
    if val >= 1: # check for overload
        code = signBit - 1
    else:
        code = int((val*multFac + 1.0) / 2.0)
    
    # Add sign bit
    if sign:
        code += signBit

    return code

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """
    if nBits <= 0: return 0

    # constants
    signBit = (1 << (nBits - 1))
    multFac = (signBit << 1) - 1

    # extract sign info
    sign = 0
    if aQuantizedNum & signBit:
        sign = 1
        aQuantizedNum -= signBit

    # get value
    val = 2.0 * aQuantizedNum / multFac

    # add sign
    if sign: val *= -1

    return val

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    N = len(aNumVec)

    if nBits <= 0:
        return np.zeros(N, dtype=np.uint64)
    
    # constants
    signBit = (1 << (nBits - 1))
    multFac = (signBit << 1) - 1

    # get signs, convert nums to abs vals
    vals = np.copy(aNumVec)
    signs = np.signbit(vals)
    vals[signs] = -vals[signs]

    # create quantization codes
    code = np.empty(N, dtype=np.uint64)

    code[vals>=1] = signBit - 1 # overloads
    code[vals<1] = ((vals[vals<1] * multFac + 1.0) / 2.0).astype(np.uint64)
    code[signs] +=  signBit

    return code


### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """
    N = len(aQuantizedNumVec)

    if nBits <= 0:
        return np.zeros(N, dtype=np.float64)

    # constants
    signBit = (1 << (nBits-1))
    multFac = (signBit << 1) - 1
    
    # extract sign info
    code = np.copy(aQuantizedNumVec)
    signs = np.zeros(N, dtype=np.bool)
    negs = (code & signBit) == signBit
    signs[negs] = True
    code[negs] -= signBit

    # compute values
    vals = 2.0 * code / multFac

    # add sign
    vals[signs] = -vals[signs]

    return vals

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    # check for reasonable inputs
    if nScaleBits < 0: nScaleBits = 0
    if nMantBits <= 0: return 0

    # set up constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (maxBits-1))

    # uniformly quantize magnitude using maxBits
    code = QuantizeUniform(abs(aNum), maxBits)

    # left-shift away the sign bit
    code <<= 1

    # Get scale factor by shifting left until you hit a 1
    scale = 0
    while scale < maxScale and (signBit & code)==0:
        code <<= 1
        scale += 1

    return scale


### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    # Check inputs
    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    #  constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits-1))

    # extract sign, and absval
    sign = 0
    if aNum < 0:
        sign = 1
        aNum *= -1

    # compute unsigned code with uniform quantization
    code = QuantizeUniform(aNum, maxBits)

    # extract mantissa (shift left by scale factor and sign bit)
    code <<= (scale + 1)

    # remove leading 1 (if these)
    if scale < maxScale:
        code -= (1 << (maxBits - 1))
        code <<= 1

    # move bits down to lowest nMantBits-1 bits
    code >>= (maxBits - nMantBits + 1)

    # add sign
    if sign: code += signBit

    return code
    

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """
    # check inputs
    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    # constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits-1))

    # determine sign bit and remove from mantissa
    if mantissa & signBit:
        sign = 1
        mantissa -= signBit
    else: sign = 0

    # build uniformly quantized code (add leading 1 if needed)
    if scale < maxScale:
        mantissa = mantissa + (1 << (nMantBits-1))

    # add trailing 1 if scale is less thatn maxScale-1 and shift-left
    if scale < (maxScale-1):
        mantissa = (mantissa << 1) + 1
        mantissa <<= (maxScale - scale - 2)

    # add signBit
    if sign:
        signBit = (1 << (maxBits-1))
        mantissa += signBit
    
    return DequantizeUniform(mantissa, maxBits)


### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    # check inputs
    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    # constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits - 1))

    # extract sign and absval
    sign = 0
    if aNum < 0:
        sign = 1
        aNum *= -1

    # compute unsigned code and remove signbit
    code = QuantizeUniform(aNum, maxBits)
    code <<= 1

    # shift left by scale factor and remove trailing zeros
    code <<= scale
    code >>= (maxBits - nMantBits + 1)

    # add sign to front of code
    if sign: code += signBit

    return code


### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """
    # check inputs
    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    # constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits - 1))

    # determine sign bit and remove from mantissa
    if mantissa & signBit:
        sign = 1
        mantissa -= signBit
    else: sign = 0

    # build code as mantissa shifted leftto maxBits - scale
    code = mantissa << (maxScale - scale)

    # add trailing 1 if needed
    if (scale < maxScale and mantissa > 0):
        code += 1 << (maxScale - scale - 1)

    # add signbit
    if sign:
        signBit = (1 << (maxBits-1))
        code += signBit

    return DequantizeUniform(code, maxBits)


### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    N = len(aNumVec)
    # check inputs
    if nMantBits <= 0: return np.zeros(N, dtype=np.uint64)
    if nScaleBits < 0: nScaleBits = 0

    # constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits-1))

    # extract signs
    vals = np.copy(aNumVec)
    signs = np.signbit(vals)
    vals[signs] = -vals[signs]

    # compute unsigned code
    code = vQuantizeUniform(vals, maxBits)

    # extract mantissas
    code <<= (scale + 1) # shift left by scale factor and sign bit (remove leading zeros)
    code >>= (maxBits - nMantBits + 1) # move bits down to lowest nMantBits-1

    # add sign to front of code
    code[signs] += signBit

    return code


### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """
    N = len(mantissaVec)
    # check inputs
    if nMantBits <= 0: return np.zeros(N, dtype=np.float64)
    if nScaleBits < 0: nScaleBits = 0

    # constants
    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1 << (nMantBits-1))

    mantissa = np.copy(mantissaVec)

    # determine sign bits and remove from mantissa
    negs = (mantissa & signBit) == signBit
    mantissa[negs] -= signBit

    # build code as mantissa shifted left to maxBits - scale
    code = mantissa << (maxScale - scale)

    # add trailing 1 if needed
    if scale < maxScale:
        code[mantissa>0] += 1 << (maxScale - scale - 1)

    # add signbit
    signBit == (1 << (maxBits-1))
    code[negs] += signBit

    return vDequantizeUniform(code, maxBits)


# #-----------------------------------------------------------------------------

# #Testing code
# if __name__ == "__main__":

#     ### YOUR TESTING CODE STARTS HERE ###

#     num = np.array([-0.99, -0.39, -0.08, -0.001, 0.0, 0.01, 0.29, 0.68, 0.99, 1.0])
#     nScale = 3
#     nMant = 5

#     print('8-bit mitread')
#     res = vDequantizeUniform(vQuantizeUniform(num, 8), 8)
#     print(res)

#     print('12-bit mitread')
#     res = vDequantizeUniform(vQuantizeUniform(num, 12), 12)
#     print(res)

#     print('3s5m FP')
#     res = []
#     for x in num:
#         scale = ScaleFactor(x)
#         mant = MantissaFP(x, scale)
#         res.append(DequantizeFP(scale, mant))
#     print(res)

#     print('3s5m BFP')
#     res = []
#     for x in num:
#         scale = ScaleFactor(x)
#         mant = Mantissa(x, scale)
#         res.append(Dequantize(scale, mant))
#     print(res)

#     print('3s5m BFP, N=10')
#     scale = ScaleFactor(np.max(np.abs(num)))
#     mant = vMantissa(num, scale)
#     res = vDequantize(scale, mant)
#     print(res)
#     ### YOUR TESTING CODE ENDS HERE ###

