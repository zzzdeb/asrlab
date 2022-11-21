1.d.
  - What portions of the signal are silence?
    Columns that are almost white are silence (almost because of noise).
  - What portions are (un-)voiced speech?
    Portions (columns) that do not have clear peaks, and look more like
    noise. Peaks mean we have on that frequency clear peak sound.
1.e. 
    Extract the spectra for the time frames 25, 105, and 410.
    Plot these three in one graph. Describe the significant differences
    between these three spectra.
    - spectrum_25105405.png illustrates spectrums in those three time frames.
    - 25 is silence, since it is mostly white.
    - 105 is voiced sound with clear frequencies in lower frequency.
    - 405 is unvoiced sound because spectrum is evenly noisy.

2.
  - Plotting shows first two dimension of mixture 44
2.2.a
  - Complexity: O(n^3)
  - Linear alignment took 49.7405 seconds
2.2.c
  - Complexity: O(n^2)
  - Linear alignment took 7.70498 seconds
