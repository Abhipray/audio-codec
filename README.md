# audio-codec
stanford music422 final project 

The `baseline` branch contains the baseline coder.
The `master` branch contains the most up-to-date coder.

In each branch:
  - The `coder/` directory holds the coder files
  - The `test_decoded/` directory holds the coded files for that coder

## SBR logic
### Encode
For some number of high frequency critical bands (what I call "omitted bands"), don't encode every frequency line. Instead just send one mantissa per band, representing the maximum amplitude of the signal in that band. This represents the envelope of the signal. As a heuristic, I use maxMantissaBits to encode this one value (can be changed, or we could send multiple frequency lines per band to be more precise about the envelope). 

### Decode
Transpose lower frequencies onto these higher frequencies (see Decode_SBR in codec.py). Then adjust by the envelope that was sent. I smooth out the envelope in order for it to be more graceful. Also should probably add some noise at these high frequencies, which I do to a small degree. 

### Current status
Higher frequencies sound really bad at 128 kb/s bit rate. Have not tried lower bitrates where maybe this is more competitive relative to the baseline coder. Marina had suggested only to use at lower bit rates, so maybe it's just not supposed to sound good at this bit rate.

