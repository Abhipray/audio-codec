"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import *  # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec  # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from detect_transients import parTransientDetect
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sbr import *

import numpy as np  # to allow conversion of data blocks to numpy's array object
MAX16BITS = 32767
SBR_FACTOR = 2

class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag = b'PAC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag = self.fp.read(4)
        if tag != self.tag:
            raise RuntimeError(
                "Tried to read a non-PAC file into a PACFile object")
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits, useSBR) \
                 = unpack('<LHLLHHH',self.fp.read(calcsize('<LHLLHHH')))
        nBands = unpack('<L', self.fp.read(calcsize('<L')))[0]
        nLines = unpack('<' + str(nBands) + 'H',
                        self.fp.read(calcsize('<' + str(nBands) + 'H')))
        sfBands = ScaleFactorBands(nLines)
        sfBandsShort = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(128, sampleRate))
        omittedBands = omitted_bands(sfBands)
        # load up a CodingParams object with the header data
        myParams = CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBands = sfBands
        myParams.sfBandsShort = sfBandsShort
        myParams.useSBR = bool(useSBR)
        myParams.omittedBands = omittedBands if myParams.useSBR else []
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels):
            overlapAndAdd.append(np.zeros(nMDCTLines, dtype=np.float64))
        myParams.overlapAndAdd = overlapAndAdd
        return myParams

    def getDecodedBlock(self, pb, codingParams, lastTrans, curTrans, nextTrans):
        if curTrans:
            sfBands = codingParams.sfBandsShort
        else:
            sfBands = codingParams.sfBands

        # extract the data from the PackedBits object
        overallScaleFactor = pb.ReadBits(
            codingParams.nScaleBits)  # overall scale factor
        scaleFactor = []
        bitAlloc = []
        mantissa = np.zeros(codingParams.nMDCTLines,
                            np.int32)  # start w/ all mantissas zero
        for iBand in range(sfBands.nBands):  # loop over each scale factor band to pack its data
            ba = pb.ReadBits(codingParams.nMantSizeBits)
            if ba:
                ba += 1  # no bit allocation of 1 so ba of 2 and up stored as one less
            bitAlloc.append(ba)  # bit allocation for this band
            scaleFactor.append(pb.ReadBits(
                codingParams.nScaleBits))  # scale factor for this band
            if bitAlloc[iBand]:
                # if bits allocated, extract those mantissas and put in correct location in matnissa array
                if iBand in codingParams.omittedBands and not curTrans:
                        m = pb.ReadBits(bitAlloc[iBand])
                else:
                    m = np.empty(sfBands.nLines[iBand], np.int32)
                    for j in range(sfBands.nLines[iBand]):
                        m[j] = pb.ReadBits(
                            bitAlloc[iBand]
                        )  # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                mantissa[sfBands.lowerLine[iBand]:(sfBands.upperLine[iBand] + 1)] = m
        # done unpacking data (end loop over scale factor bands)

        # CUSTOM DATA:
        # < now can unpack any custom data passed in the nBytes of data >

        # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
        decodedData = self.Decode(scaleFactor, bitAlloc, mantissa,
                                  overallScaleFactor, codingParams,
                                  lastTrans, curTrans, nextTrans)
        
        return decodedData


    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # loop over channels (whose coded data are stored separately) and read in each data block
        data = []
        for iCh in range(codingParams.nChannels):
            data.append(np.array(
                [], dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s = self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd = codingParams.overlapAndAdd
                    codingParams.overlapAndAdd = 0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",
                            s)[0]  # read it as a little-endian unsigned long
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData(
                self.fp.read(nBytes)
            )  # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:
                raise "Only read a partial block of coded PACFile data"

            # Read block switching info
            lastTrans = pb.ReadBits(1)
            curTrans  = pb.ReadBits(1)
            nextTrans = pb.ReadBits(1)

            if not curTrans: # normal sized block
                decodedData = self.getDecodedBlock(pb, codingParams, lastTrans, curTrans, nextTrans)
            
            else: # short blocks
                long = codingParams.nSamplesPerBlock
                short = 128
                codingParams.nSamplesPerBlock = short
                codingParams.nMDCTLines = short

                decodedData = np.zeros(2*long)
                pad = long // 2 - short // 2
                for n in range(pad, 2*long-short-pad, short):
                    dataShort = self.getDecodedBlock(pb, codingParams, lastTrans, curTrans, nextTrans)
                    decodedData[n:n+2*short] += dataShort

                codingParams.nSamplesPerBlock = long
                codingParams.nMDCTLines = long

            data[iCh] = np.concatenate(
                (data[iCh],
                 np.add(codingParams.overlapAndAdd[iCh],
                        decodedData[:codingParams.nMDCTLines])
                 ))  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[
                codingParams.nMDCTLines:]  # save other half for next pass

        # end loop over channels, return signed-fraction samples for this block
        return data

    def WriteFileHeader(self, codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples % codingParams.nMDCTLines:
            codingParams.numSamples += (
                codingParams.nMDCTLines -
                codingParams.numSamples % codingParams.nMDCTLines
            )  # zero padding for partial final PCM block
        # also add in the delay block for the second pass w/ the last half-block
        codingParams.numSamples += codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block
        # write the coded file attributes
        self.fp.write(
            pack('<LHLLHHH', codingParams.sampleRate, codingParams.nChannels,
                 codingParams.numSamples, codingParams.nMDCTLines,
                 codingParams.nScaleBits, codingParams.nMantSizeBits,
                 codingParams.useSBR))
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands = ScaleFactorBands(
            AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                          codingParams.sampleRate))
        codingParams.sfBands = sfBands
        codingParams.sfBandsShort = ScaleFactorBands(
            AssignMDCTLinesFromFreqLimits(128, codingParams.sampleRate))
        omittedBands = omitted_bands(sfBands)
        codingParams.omittedBands = omittedBands if codingParams.useSBR else []
        self.fp.write(pack('<L', sfBands.nBands))
        self.fp.write(
            pack('<' + str(sfBands.nBands) + 'H', *(sfBands.nLines.tolist())))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(
                np.zeros(codingParams.nMDCTLines, dtype=np.float64))
        codingParams.priorBlock = priorBlock
        return


    def getNumBytesNeeded(self, codingParams, bitAlloc, iCh, curTrans=False):
        nBytes = codingParams.nScaleBits  # bits for overall scale factor
        if curTrans:
            sfBands = codingParams.sfBandsShort
        else:
            sfBands = codingParams.sfBands

        for iBand in range(sfBands.nBands):  # loop over each scale factor band to get its bits
            nBytes += codingParams.nMantSizeBits + codingParams.nScaleBits  # mantissa bit allocation and scale factor for that sf band
            if bitAlloc[iCh][iBand]:
                # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                if iBand in codingParams.omittedBands and not curTrans:
                    nBytes += bitAlloc[iCh][iBand]
                else:
                    nBytes += bitAlloc[iCh][iBand] * sfBands.nLines[
                        iBand]  # no bit alloc = 1 so actuall alloc is one higher
        
        return nBytes

    def writeEncodedBits(self, pb, codingParams, iCh, overallScaleFactor, bitAlloc, scaleFactor, mantissa, curTrans=False):
        if curTrans:
            sfBands = codingParams.sfBandsShort
        else:
            sfBands = codingParams.sfBands

        # now pack the nBytes of data into the PackedBits object
        pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)  # overall scale factor
        iMant = 0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
        for iBand in range(sfBands.nBands):  # loop over each scale factor band to pack its data
            ba = bitAlloc[iCh][iBand]
            if ba:
                ba -= 1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
            pb.WriteBits(ba, codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
            pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
            if bitAlloc[iCh][iBand]:
                if iBand in codingParams.omittedBands and not curTrans:
                    pb.WriteBits(
                        mantissa[iCh][iMant], bitAlloc[iCh][iBand]
                    )  # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                    iMant += 1
                else:
                    for j in range(sfBands.nLines[iBand]):
                        pb.WriteBits(mantissa[iCh][iMant + j], bitAlloc[iCh][iBand])  # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                    iMant += sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
        # done packing (end loop over scale factor bands)

    def WriteDataBlock(self, data, codingParams, lastTrans=False, curTrans=False, nextTrans=False):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData = []
        for iCh in range(codingParams.nChannels):
            fullBlockData.append(
                np.concatenate((codingParams.priorBlock[iCh], data[iCh])))
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        if not curTrans:
            # (ENCODE HERE) Encode the full block of multi=channel data
            (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(
                fullBlockData, codingParams, lastTrans=lastTrans,
                curTrans=curTrans, nextTrans=nextTrans
            )  # returns a tuple with all the block-specific info not in the file header
        else:
            long = codingParams.nSamplesPerBlock
            short = 128
            codingParams.nSamplesPerBlock = short
            codingParams.nMDCTLines = short

            scaleFactors = []
            bitAllocs = []
            mantissas = []
            overallScaleFactors = []
            pad = long // 2 - short // 2
            for n in range(pad, 2*long-short-pad, short):
                shortBlock = []
                for iCh in range(codingParams.nChannels):
                    if np.all(fullBlockData[iCh][n:n+2*short] == 0):
                        codingParams.nSamplesPerBlock = long
                        codingParams.nMDCTLines = long
                        return

                    shortBlock.append(fullBlockData[iCh][n:n+2*short])

                (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(
                    shortBlock, codingParams, lastTrans=lastTrans, curTrans=curTrans, nextTrans=nextTrans)
                
                scaleFactors.append(scaleFactor)
                bitAllocs.append(bitAlloc)
                mantissas.append(mantissa)
                overallScaleFactors.append(overallScaleFactor)

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):
            if not curTrans:
                nBytes = self.getNumBytesNeeded(codingParams, bitAlloc, iCh)
            else:
                nBytes = 0
                for bitAlloc in bitAllocs:
                    nBytes += self.getNumBytesNeeded(codingParams, bitAlloc, iCh, True)

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>
            nBytes += 4 # for block switching

            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes % BYTESIZE == 0: nBytes //= BYTESIZE
            else: nBytes = nBytes // BYTESIZE + 1
            self.fp.write(pack(
                "<L",
                int(nBytes)))  # stores size as a little-endian unsigned long

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # Write block switching info
            pb.WriteBits(lastTrans, 1)
            pb.WriteBits(curTrans, 1)
            pb.WriteBits(nextTrans, 1)

            if not curTrans:
                self.writeEncodedBits(pb, codingParams, iCh, overallScaleFactor, bitAlloc, scaleFactor, mantissa)
            else:
                for n in range(len(bitAllocs)):
                    self.writeEncodedBits(pb, codingParams, iCh, overallScaleFactors[n],
                        bitAllocs[n], scaleFactors[n], mantissas[n], True)
                
                codingParams.nSamplesPerBlock = long
                codingParams.nMDCTLines = long

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return

    def Close(self, codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [
                np.zeros(codingParams.nMDCTLines, dtype=np.float),
                np.zeros(codingParams.nMDCTLines, dtype=np.float)
            ]
            self.WriteDataBlock(data, codingParams)
        self.fp.close()

    def Encode(self, data, codingParams, lastTrans=False, curTrans=False, nextTrans=False):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        if codingParams.useSBR and not curTrans:
            return codec.Encode_SBR(data, codingParams, lastTrans, curTrans, nextTrans)
        
        return codec.Encode(data, codingParams, lastTrans, curTrans, nextTrans)

    def Decode(self, scaleFactor, bitAlloc, mantissa, overallScaleFactor,
               codingParams, lastTrans=False, curTrans=False, nextTrans=False):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        if codingParams.useSBR and not curTrans:
            return codec.Decode_SBR(scaleFactor, bitAlloc, mantissa,
                            overallScaleFactor, codingParams,
                            lastTrans, curTrans, nextTrans)
        
        return codec.Decode(scaleFactor, bitAlloc, mantissa,
                            overallScaleFactor, codingParams,
                            lastTrans, curTrans, nextTrans)


#-----------------------------------------------------------------------------

input_dir = Path('../test_signals')
output_dir = Path('../test_decoded96')
bitrates = [96]
os.makedirs(output_dir, exist_ok=True)

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__ == "__main__":

    print("\nTesting the PAC coder (input.wav -> coded.pac -> output.wav):")
    import time
    from pcmfile import *  # to get access to WAV file handling
    elapsed = time.time()

    for data_rate in bitrates:
        for in_file in input_dir.glob('*.wav'):
            for Direction in ("Encode", "Decode"):
                #    for Direction in ("Decode"):
                print(f'Processing {in_file} at {data_rate}kbps')
                # create the audio file objects
                if Direction == "Encode":
                    print("\n\tEncoding input PCM file...", )
                    inFile = PCMFile(str(in_file))
                    outFile = PACFile(
                        f"{output_dir}/{in_file.stem}_coded_{data_rate}.pac")
                else:  # "Decode"
                    print("\n\tDecoding coded PAC file...", )
                    inFile = PACFile(
                        f"{output_dir}/{in_file.stem}_coded_{data_rate}.pac")
                    outFile = PCMFile(
                        f"{output_dir}/{in_file.stem}_{data_rate}.wav")
                # only difference is file names and type of AudioFile object

                # open input file
                codingParams = inFile.OpenForReading(
                )  # (includes reading header)

                # pass parameters to the output file
                if Direction == "Encode":
                    # set additional parameters that are needed for PAC file
                    # (beyond those set by the PCM file on open)
                    codingParams.nMDCTLines = 1024
                    codingParams.nScaleBits = 4
                    codingParams.nMantSizeBits = 16
                    codingParams.targetBitsPerSample = data_rate / (codingParams.sampleRate / 1000)
                    codingParams.useSBR = True if data_rate < 128 else False
                    # tell the PCM file how large the block size is
                    codingParams.nSamplesPerBlock = codingParams.nMDCTLines
                else:  # "Decode"
                    # set PCM parameters (the rest is same as set by PAC file on open)
                    codingParams.bitsPerSample = 16
                # only difference is in setting up the output file parameters

                # open the output file
                outFile.OpenForWriting(
                    codingParams)  # (includes writing header)

                # Read the input file and pass its data to the output file to be written
                lookahead = np.zeros((codingParams.nChannels, 2*codingParams.nSamplesPerBlock))
                curBlockHasTransient = False
                lastBlockHadTransient = False
                while True:
                    if Direction == "Encode":
                        data = inFile.ReadDataBlock(codingParams)

                        if not data:
                            nextBlockHasTransient = False
                        else:
                            lookahead = np.concatenate((np.copy(data), lookahead[:,codingParams.nSamplesPerBlock:]), axis=1)
                            nextBlockHasTransient = parTransientDetect(lookahead)

                        outFile.WriteDataBlock(lookahead[:,:codingParams.nSamplesPerBlock], codingParams,
                                               lastTrans=lastBlockHadTransient, curTrans=curBlockHasTransient, nextTrans=nextBlockHasTransient)
                        lastBlockHadTransient = curBlockHasTransient
                        curBlockHasTransient = nextBlockHasTransient

                        if not data: break

                    else:
                        data = inFile.ReadDataBlock(codingParams)
                        if not data: break  # we hit the end of the input file
                        outFile.WriteDataBlock(data, codingParams)
                    
                    print(
                        ".",
                        end="")  # just to signal how far we've gotten to user
                # end loop over reading/writing the blocks

                # close the files
                inFile.Close(codingParams)
                outFile.Close(codingParams)
            # end of loop over Encode/Decode

            elapsed = time.time() - elapsed
            print("\nDone with Encode/Decode test\n")
            print(elapsed, " seconds elapsed")
